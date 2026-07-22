"""
simulate.py -- run reality + twin + the three policies through time.

This is the loop that produces the paper's numbers. Every tick:

  1. reality advances (traffic drifts, satellites move, Earth rotates)
  2. the twin observes reality (late) and updates its stale belief
  3. the real graph and the twin graph are built
  4. each policy decides its route (on the graph it's allowed to see)
  5. we measure each policy's TRUE delay (priced on the real graph --
     because what you actually experience is real, even if you chose
     your route from a stale twin) and its energy.

The key honesty point: policies CHOOSE using whatever view they're
entitled to (direct sees truth, twin policies see the twin), but they're
all SCORED on reality. You live with the real consequences of your choice.
"""

import numpy as np

import config
import orbits, isl, ground, network, traffic, energy
from twin import DigitalTwin
from gate import DirectPolicy, TwinNoGatePolicy, TwinGatePolicy


def run(threshold, seed=0, duration_s=None, warmup_s=120,
        telemetry_delay_s=None, telemetry_interval_s=None):
    """
    Run one simulation. Returns a metrics dict per policy.

    threshold : the gate's strictness (only affects TWIN_GATE)
    seed      : random seed for background traffic
    All three policies see the SAME reality and the SAME twin, so the
    only difference between them is their decision rule.
    """
    duration_s = config.SIM_DURATION_S if duration_s is None else duration_s

    sats = orbits.build_constellation()
    gs = ground.build_ground_stations()
    plan = isl.build_isl_plan()
    rng = np.random.default_rng(seed)
    bg = traffic.BackgroundTraffic(plan, [g["name"] for g in gs], rng)
    tw = DigitalTwin(delay_s=telemetry_delay_s, interval_s=telemetry_interval_s)

    policies = [DirectPolicy(), TwinNoGatePolicy(), TwinGatePolicy(threshold)]

    # per-policy accumulators
    stats = {p.name: {"true_delay_ms": [], "switches": 0,
                      "reconfig_energy_j": 0.0, "hops": []}
             for p in policies}

    t = 0.0
    n_ticks = int(duration_s)
    for i in range(n_ticks):
        # 1. reality advances
        bg.step(1.0)
        orbits.propagate(sats, t)
        ground.update_ground_stations(gs, t)

        # 2. twin observes (late)
        tw.observe(bg, t)

        # 3. build both graphs
        G_real = network.build_graph(sats, gs, plan)
        traffic.annotate_graph(G_real, bg)          # true delays
        G_twin = network.build_graph(sats, gs, plan)
        traffic.annotate_graph(G_twin, tw)          # twin-believed delays

        # skip the warmup period in measurement (twin still filling up,
        # policies still settling on a first route)
        measuring = t >= warmup_s

        # 4 + 5. each policy decides, then is scored on REALITY
        for p in policies:
            before = p.current_path
            path = p.decide(G_real, G_twin, "IST", "NYC")

            if not measuring or path is None:
                continue

            true_delay = network.path_delay_s(G_real, path) * 1000  # ms
            stats[p.name]["true_delay_ms"].append(true_delay)
            stats[p.name]["hops"].append(
                sum(1 for a, b in zip(path[:-1], path[1:])
                    if isinstance(a, int) and isinstance(b, int)))

            if p.switched_this_tick:
                stats[p.name]["switches"] += 1
                stats[p.name]["reconfig_energy_j"] += energy.reconfig_energy_j(
                    before, path)

        t += 1.0

    # summarise
    out = {}
    for name, s in stats.items():
        d = np.array(s["true_delay_ms"])
        out[name] = {
            "mean_delay_ms": float(d.mean()) if len(d) else float("nan"),
            "p95_delay_ms": float(np.percentile(d, 95)) if len(d) else float("nan"),
            "switches": s["switches"],
            "reconfig_energy_j": s["reconfig_energy_j"],
            "mean_hops": float(np.mean(s["hops"])) if s["hops"] else float("nan"),
            "ticks_measured": len(d),
        }
    return out


if __name__ == "__main__":
    # QUICK 1-SEED LOOK. Just the shape. Is the gate doing anything?
    import sys

    print("=" * 74)
    print("QUICK 1-SEED SANITY LOOK -- is the threshold doing anything?")
    print("=" * 74)
    print(f"\nTelemetry delay: {config.TELEMETRY_DELAY_S}s, "
          f"seed 0, {int(config.SIM_DURATION_S)}s run\n")

    # First: the two fixed references (they don't depend on threshold).
    ref = run(threshold=0.90, seed=0)
    print("REFERENCE POLICIES (threshold-independent):")
    print(f"  {'policy':14} {'mean delay':>11} {'p95':>9} "
          f"{'switches':>9} {'energy(J)':>11}")
    for name in ["direct", "twin_nogate"]:
        r = ref[name]
        print(f"  {name:14} {r['mean_delay_ms']:>9.1f}ms "
              f"{r['p95_delay_ms']:>7.1f}ms {r['switches']:>9} "
              f"{r['reconfig_energy_j']:>11.4f}")

    print("\nGATE, swept across thresholds:")
    print(f"  {'threshold':>10} {'mean delay':>11} {'p95':>9} "
          f"{'switches':>9} {'energy(J)':>11}")

    direct_sw = ref["direct"]["switches"]
    nogate_sw = ref["twin_nogate"]["switches"]

    for th in config.GATE_SWEEP_VALUES:
        r = run(threshold=th, seed=0)["twin_gate"]
        print(f"  {th:>10.2f} {r['mean_delay_ms']:>9.1f}ms "
              f"{r['p95_delay_ms']:>7.1f}ms {r['switches']:>9} "
              f"{r['reconfig_energy_j']:>11.4f}")

    print(f"\n  (direct oracle made {direct_sw} switches; "
          f"twin-no-gate made {nogate_sw})")
    print("\nWhat to look for:")
    print("  - switches should DROP as threshold tightens (1.00 -> 0.50)")
    print("  - delay should RISE as threshold tightens (fewer good switches)")
    print("  - if switches are flat across thresholds, the gate isn't biting")
