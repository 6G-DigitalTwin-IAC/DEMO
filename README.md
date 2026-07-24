# LEO Digital Twin — Bricks 1 & 2

**Status:** 37/37 physics checks passing.

## Run it

```
pip install numpy networkx
python verify.py
```

## The build order (each brick needs the last)

| Brick | What | Status |
|---|---|---|
| 1 | Real network: orbits, ISLs, rotating Earth | done |
| 2 | Traffic and congestion | done |
| 3 | Digital twin (delayed telemetry) | next |
| 4 | Energy model | |
| 5 | Control gate | |

## The constellation

**Walker Delta 53.0: 264/12/1** — 264 sats, 12 planes, 22 per plane,
53° inclination, 550 km, 95.5 min period, 7.589 km/s.

Same inclination and altitude as Starlink shell 1 (`53.0: 1584/72/39`).

### Why 264 and not 40

We tried 40 (5 planes × 8). **It's geometrically impossible.** 8 sats per
ring sit 5,297 km apart; 5 planes leave an 8,136 km equatorial gap. Both
exceed any laser terminal's reach. A 40-sat shell at 550 km cannot form a
+Grid at all.

The old code hid this by linking anything within 2,600 km — which is why it
appeared to work at 100 satellites while being physically impossible.

22 sats/plane (1,970 km) and 12 planes (2,315–3,680 km) both close. That's
why Starlink uses 22 per plane.

## Two findings for the paper

### 1. M/M/1 collapses at LEO ISL speeds

The textbook queueing model gives `Wq = (1/μ)·ρ/(1-ρ)`. At 100 Gbps with
1500-byte packets, μ = 8,333,333 packets/s — the link swallows a packet in
**0.12 microseconds**. Light needs **6.57 ms** to cross a 1,970 km ISL.

**The queue is 54,758× faster than light.**

| ρ | M/M/1 queue | vs one hop |
|---|---|---|
| 0.50 | 0.0001 ms | 0.00% |
| 0.90 | 0.0011 ms | 0.02% |
| 0.99 | 0.0119 ms | **0.18%** |

A link at 99% utilisation — effectively dead — adds 0.18% to a hop. With
M/M/1, congestion is invisible, routes never change, the twin has nothing
to predict, and the gate has nothing to filter. **No paper.**

Not an arithmetic slip. M/M/1 is right for a slow link and wrong for a
100 Gbps laser spanning 2,000 km of vacuum.

### 2. Bufferbloat is what actually happens

M/M/1 assumes Poisson arrivals. Real traffic is bursty and real routers have
deep buffers. Under sustained load the buffer fills and stays full.

**Measured** (arXiv:2310.09242): Starlink RTT inflates 2–4× under load,
reaching 400–500 ms, against a ~33 ms baseline. Four orders of magnitude
above M/M/1.

Our model: `delay = BUFFER_DELAY_MS · ρ^exponent`

| ρ | queue delay | vs one hop |
|---|---|---|
| 0.30 | 18.0 ms | 2.7× |
| 0.90 | 162.0 ms | 24.7× |

Now congestion is on the same scale as geometry, so detouring is worth it.

**Note:** SpaceX has since right-sized buffers (median 48.5→33 ms, p99
>150→<65 ms). We model the measured inflation regime; operators are
engineering it away.

## Verified results

```
IST → NYC over 1 hour:
                   min      mean      max
   propagation   31.84ms   45.73ms  117.85ms
   total delay  144.94ms  277.94ms  541.61ms
   hops: 5 to 14

   straight line through Earth: 25.15 ms  (PHYSICAL FLOOR)
   terrestrial fibre:           ~40 ms
   our propagation:              45.73 ms
   our total:                   277.94 ms  (6.1x propagation)
```

## Files

| File | Job |
|---|---|
| `config.py` | Every number, tagged [PHYSICS]/[SOURCED]/[DERIVED]/[SWEPT]/[ASSUMED] |
| `orbits.py` | Where satellites are at time t |
| `isl.py` | +Grid topology |
| `ground.py` | Ground stations on a rotating Earth |
| `traffic.py` | Congestion model + background load |
| `network.py` | Assemble into a routable graph |
| `verify.py` | Prove it obeys physics |

## What was wrong with the old version

- **ISL topology.** Linked anything within 2,600 km. A laser terminal is a
  physical telescope on a gimbal — you get 4, and you don't re-aim (lock
  acquisition takes seconds to minutes). Verified: intra-plane distance
  varies **0.000000 km** over a full orbit; inter-plane swings 1,365 km.
- **Fixed Earth.** Istanbul moves at 350.6 m/s in the inertial frame.
- **No time units.** `OMEGA = 0.015 rad/frame` but delays reported in
  seconds. A "frame" wasn't a unit of anything. Now 1 tick = 1 second.
- **Fake orbits.** A rotation trick, not orbital mechanics.

## Known limitations (state in the paper)

- **No J2 perturbation.** RAAN drifts ~5°/day at 550 km; over 1 hour that's
  ~0.2°, affecting all planes near-equally, so relative topology barely moves.
- **No drag, SRP, or station-keeping. Circular orbits only.**
- **Polar cutoff never fires at 53°.** Satellites never exceed 53° latitude,
  so the 70° check is dead code *at this inclination*. Correct and ready for
  a polar shell. Deliberately NOT lowered to force link breaks — that would
  fabricate a physical effect.
- **Bufferbloat model is phenomenological.** Reproduces the order of
  magnitude of published Starlink behaviour. Not fitted to a dataset.
  `BUFFER_DELAY_MS` and `BUFFER_EXPONENT` must be SWEPT.
- **4 laser terminals assumed.** Literature says 4; current V2 Mini hardware
  reportedly has 3. Footnote it.
- **GSL capacity (4 Gbps) is assumed.** Public per-gateway figures aren't clean.
- **One flow, one ground-station pair.** Multi-flow is future work.

## Known issue

The autocorrelation test measures over only 600 samples, so the estimate
wobbles (0.070 at 30s, 0.148 at 60s — should decay smoothly). The physics is
right; the measurement is noisy. Needs a longer series before it goes in a
figure.

## Next: Brick 3

The twin doesn't see true congestion. It sees a telemetry report that left
the satellite hundreds of milliseconds ago. **That staleness — not invented
random noise — is the error the gate protects against.**

`BackgroundTraffic.snapshot()` already exists for exactly this: it's the
report the twin will receive, late.

The key parameter is already in config: `BACKGROUND_CORRELATION_TIME_S = 30`.
If load were white noise, no twin could ever predict it and the paper's
question would be rigged. Verified autocorrelation: 0.95 at 1s, 0.41 at 15s,
~0.08 at 120s. Predictable but imperfect — the regime worth studying.

---

## Brick 3: The Digital Twin (done)

`twin.py` + `verify_twin.py`. 6/6 checks passing.

### The idea

Two things now exist and they are not the same:

- **Reality** — `BackgroundTraffic`. What is truly on every link, right now.
- **The twin** — a ground-based model that sees telemetry reports which
  arrived *late*, and between reports holds its last picture.

The twin is always looking at a slightly old photo of the network. It's
wrong because **information takes time to travel** — not because we added
random noise. Kill the delay and the error vanishes. Nobody can argue with
the speed of light.

The twin exposes the same `.utilisation(u, v, kind)` interface as reality,
so it drops into `annotate_graph()` unchanged. The routing code can't tell
whether it's using truth or belief — exactly like a real router.

### Verified: the twin is wrong for the RIGHT reason

| telemetry delay | mean belief age | mean error |
|---|---|---|
| 0 s | 2 s | 0.038 |
| 5 s | 7 s | 0.078 |
| 10 s | 12 s | 0.098 |
| 40 s | 42 s | 0.148 |

- **Error grows with delay** — staler belief, wronger twin.
- **Error vanishes at zero delay** (0.0000) — proof it's staleness, not noise.
- **Error is bounded** (~0.156 even at 300s) by how far reality drifts.
  A stale snapshot was *true once*; it can only be as wrong as reality has
  moved. The old ±15% noise had no such ceiling.

### The result that justifies the gate

**At 10s telemetry delay, the twin disagrees with reality about the best
route 45.7% of the time — and when it does, its route is genuinely worse,
costing 43.6 ms more on average (max 207 ms).**

That is the gate's reason to exist, quantified. Nearly half the twin's
proposed switches would move you to a worse route because it's acting on an
old photo. The gate (Brick 5) filters those.

### New config parameters (both SWEPT)

- `TELEMETRY_DELAY_S = 5.0` — how old the twin's info is. The headline swept
  parameter: the paper's key figure is "gate benefit vs. telemetry delay".
- `TELEMETRY_UPDATE_INTERVAL_S = 5.0` — how often a fresh report arrives.

### Honest flag

45.7% disagreement at 10s delay is high. Either a strong signal for the gate
to work with, or a sign the twin/telemetry parameters need tuning so the twin
isn't hopeless. The gate run will tell us. Noted, not yet resolved.

### Next: Brick 4 (energy), then Brick 5 (the gate — the actual contribution)

---

## Brick 4: Energy (done)

`energy.py` + `verify_energy.py`. 8/8 checks passing.

### The claim and how we keep it honest

Claim: **"unnecessary reroutes waste energy."** For that to mean anything,
*changing* a route has to cost something.

The old code faked this with `ENERGY_PER_SWITCH = 5e-3` — an invented
constant whose ratio to path cost *was* the entire result. Unfalsifiable.

Two kinds of energy now:

- **Transmission** — carrying the data. ~power × time × hops. Varies with
  route length but it's the same data either way; not the switching cost.
- **Reconfiguration** — *changing* the route. `nodes_changed × FLOPs_per_node
  × J_per_FLOP`. **Zero if you don't switch.** This is the avoidable cost of
  an unnecessary reroute, and it's what the gate saves.

We count **nodes changed** (symmetric difference + next-hop changes), not
whole-path identity — so swapping one hop is cheap and rerouting everything
is dear. More honest than the old whole-path counting.

### The number problem, and why the paper is safe anyway

Reconfiguration energy = `nodes × FLOPs_per_node × J_per_FLOP`.

- `J_per_FLOP = 5×10⁻⁹` is **sourced** (4–7×10⁻⁹ range for space processors).
- `FLOPs_per_node` we **cannot** source precisely — it depends on table size
  and implementation. So it's the **swept** parameter.

At a low FLOPs value, reconfiguration is a rounding error next to
transmission. At a high value, it dominates. **We refuse to pick the value
that makes the result look good** — that's the old cheat.

The fix: we never claim an absolute joule figure or compare reconfiguration
to transmission. We report the **percentage reduction in reroute energy,
gated vs. ungated**. Verified: that percentage is **identical (50.0%) at
10⁶, 10⁹, and 10¹² FLOPs/node** — the uncertain number cancels out of the
ratio. The result cannot be attacked by disputing the FLOPs figure.

### Sources (verify before citing)

- **J/FLOP:** Veeravalli, arXiv:2601.06706 (2026), adopts 5×10⁻⁹, range
  4–7×10⁻⁹ from its refs [23,24]. **This paper post-dates our prep — read it
  and cite its primary refs [23,24], not the secondary.**
- **Independent evidence reconfiguration isn't free:** same source reports
  dynamic reconfiguration incurs 200–500 ms overhead (time/signalling, not
  joules) — citable support for the premise.
- **Sanity:** modern LEO sats carry 10–50 GFLOPS, so a ms-scale recompute is
  tens of millions of FLOPs — inside our swept range.

### Limitation named for the paper

The field ultimately frames energy as **battery cycle aging → satellite
lifetime** (Yan JSAC 2016; SGION JOCN 2025). We model the direct joules only
and name the battery/lifetime mapping as future work. Our claim holds at the
joule level; the lifetime claim would need a battery + eclipse model we don't
build.

### Status: Bricks 1–4 done. Next: Brick 5, the gate — the actual contribution.

---

## Brick 5: The Control Gate (done) — THE CONTRIBUTION

`gate.py`, `simulate.py`, `sweep.py`. The gate filters the twin's proposed
switches: allow a switch only if `twin_pred(new) < threshold × twin_pred(current)`.
The gate is as blind as the twin (it can only see the twin's view), which is
the honest real-world constraint.

Three policies run on the SAME reality, scored on reality:
- **direct** — oracle, always true best route. A delay *ceiling* (impossible
  to reach). Note it also switches the MOST (chases every change), so it's not
  a switch baseline.
- **twin_nogate** — obeys the twin every tick. What prior work does. Thrashes.
- **twin_gate** — twin proposes, gate filters. **Our method.**

### The headline result (5 seeds × 500 ticks, telemetry delay 5s, paired)

| threshold | switch cut | energy cut | delay cost |
|---|---|---|---|
| 0.95 | 28.3% | 26.4% | 0.2% |
| 0.90 | 46.1% | 38.7% | 0.5% |
| **0.80** | **65.4%** | **58.3%** | **2.2%** |
| 0.70 | 73.3% | 67.6% | 4.5% |
| 0.60 | 79.1% | 73.4% | 6.4% |

All vs. twin-no-gate (same twin, gate off). Multi-seed with std shown, e.g.
switches at 0.80 = 13.2±2.4 — the effect is real, not one lucky seed.

**This reproduces the abstract's ~65% switch-reduction claim from an HONEST
model**, and the energy cut (58%) is far better than the abstract's ~15%.
The delay stays near-optimal (2.2% at the 65% point), exactly as claimed.

### The curve has a sweet spot (the paper's story)

- **0.95–0.90:** cut 28–46% of switches for ~0% delay — nearly free.
- **0.80:** cut 65% for 2.2% — the knee, the recommended operating point.
- **Below 0.70:** diminishing returns — delay climbs faster than switches fall.

This is the tradeoff curve the project was always chasing. Not one magic
number: the whole response of delay/switches/energy to the threshold knob.

### Honest caveats
- Runs are 500 ticks × 5 seeds for speed. For the paper: longer runs
  (1800+ ticks), more seeds (10–20), and check the curve is stable.
- Graph is rebuilt every tick, which is slow (~0.9s/100 ticks). A caching
  refactor would let much bigger sweeps run. Optimization, not correctness.
- `direct` switches more than the twin policies — it's a delay ceiling, not
  a switch baseline. Compare gate against twin-no-gate, not against direct.
- The threshold `0.80` sweet spot is at THIS telemetry delay (5s). The paper's
  key second figure is how the sweet spot moves as telemetry delay changes —
  that sweep is the natural next experiment.

### Status: Bricks 1–5 DONE. The simulation is complete and produces the result.
Next: bigger sweeps, the telemetry-delay figure, then the paper and the demo.

---

## CORRECTION (important — supersedes earlier text)

**An earlier version of this code wrongly disabled 22 inter-plane links,
claiming Walker Delta constellations have a "seam." That was false.**

**Seams are a Walker STAR feature, not Walker Delta.** The two families
differ in how they spread their orbital planes:

- **Walker STAR** spreads ascending nodes over **180°**. Wrapping around that
  half-circle puts you next to planes travelling the *opposite* direction
  (ascending vs descending). Relative velocity ~2× orbital speed — far too
  fast for a laser to track. That counter-rotating boundary *is* the seam.
  **Iridium and OneWeb are Walker Star — they have seams.**
- **Walker DELTA** spreads ascending nodes over the full **360°**. Every plane
  runs the *same* direction. No counter-rotation, no seam.

**Ours is Walker Delta**, so `ISL_DISABLE_SEAM = False` and all 264 satellites
have their full 4 ISLs (2 intra-plane + 2 inter-plane = 528 links total).
Previously 44 satellites were artificially limited to 3 links.

**Sources:**
- IETF `draft-piraux-space-constellation-code-00`: for Walker Delta, "there is
  no seam effect as in the Walker Star pattern. Instead, each orbit progresses
  in the same direction and crosses paths twice with every other orbit."
- MATLAB `walkerDelta` documentation: Delta distributes ascending nodes across
  360°; Star distributes across 180°.
- arXiv:2209.05984: describes seams as arising from "adjacent counterrotating
  planes" in Walker *Star* polar constellations.

The flag is kept (not deleted) so a Walker Star variant can be simulated later.

### Did the fix change the result? No.

| threshold | switch cut | energy cut | delay cost |
|---|---|---|---|
| 0.90 | 46.3% | 39.2% | 0.3% |
| **0.80** | **61.1%** | **58.0%** | **2.1%** |
| 0.70 | 72.2% | 70.1% | 3.4% |

Before the fix, threshold 0.80 gave 65.4% / 58.3% / 2.2%. **Essentially
unchanged.** The result comes from the gate logic, not from a topology
artifact — which is exactly what you want to see when you fix a bug.

### Also fixed: the autocorrelation measurement

The Brick 2 autocorrelation test was measured over only 600 samples from one
link, producing non-monotonic nonsense (0.07 at 30s bouncing back to 0.15 at
60s). Now averaged over 40 links × 4000 samples, it decays cleanly as an
Ornstein-Uhlenbeck process should:

`0.966 (1s) → 0.843 (5s) → 0.599 (15s) → 0.359 (30s) → 0.124 (60s) → 0.016 (120s)`

The physics was always right; the measurement was too noisy. Now closed.
