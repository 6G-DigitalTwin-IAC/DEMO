"""
test_adaptive.py -- THE END-TO-END TEST.

Runs reality + twin with FOUR controllers side by side, on the same
reality, and compares them:

  direct              -- oracle, sees truth. The ceiling.
  twin_nogate         -- obeys the stale twin. Thrashes. The "before".
  twin_gate_fixed     -- our gate at a single fixed threshold.
  twin_gate_adaptive  -- our gate, threshold chosen live by the kNN from
                         what the twin can see. THE AUTONOMOUS SYSTEM.

The question: does the adaptive gate, choosing its own threshold blind
from the twin, perform as well as the best fixed threshold -- WITHOUT us
hand-tuning it per scenario? If yes, the adaptive system works.
"""

import sys
import numpy as np

import config
import orbits, isl, ground, network, traffic, energy
from twin import DigitalTwin
from gate import (DirectPolicy, TwinNoGatePolicy, TwinGatePolicy,
                  TwinGateAdaptivePolicy)
from adaptive import AdaptiveThresholdKNN


def run_scenario(delay, sigma, knn, fixed_threshold=0.80,
                 seed=0, duration_s=500, warmup_s=120):
    config.TELEMETRY_DELAY_S = delay
    config.BACKGROUND_LOAD_SIGMA = sigma

    sats = orbits.build_constellation()
    gs = ground.build_ground_stations()
    plan = isl.build_isl_plan()
    rng = np.random.default_rng(seed)
    bg = traffic.BackgroundTraffic(plan, [g["name"] for g in gs], rng)
    tw = DigitalTwin(delay_s=delay, interval_s=config.TELEMETRY_UPDATE_INTERVAL_S)

    policies = [
        DirectPolicy(),
        TwinNoGatePolicy(),
        TwinGatePolicy(fixed_threshold),
        TwinGateAdaptivePolicy(knn, delay, tw, recompute_every=10),
    ]
    stats = {p.name: {"delay": [], "switches": 0, "energy": 0.0} for p in policies}

    t = 0.0
    for i in range(int(duration_s)):
        bg.step(1.0)
        orbits.propagate(sats, t)
        ground.update_ground_stations(gs, t)
        tw.observe(bg, t)

        G_real = network.build_graph(sats, gs, plan); traffic.annotate_graph(G_real, bg)
        G_twin = network.build_graph(sats, gs, plan); traffic.annotate_graph(G_twin, tw)

        measuring = t >= warmup_s
        for p in policies:
            before = p.current_path
            path = p.decide(G_real, G_twin, "IST", "NYC")
            if not measuring or path is None:
                continue
            stats[p.name]["delay"].append(network.path_delay_s(G_real, path) * 1000)
            if p.switched_this_tick:
                stats[p.name]["switches"] += 1
                stats[p.name]["energy"] += energy.reconfig_energy_j(before, path)
        t += 1.0

    out = {}
    for name, s in stats.items():
        d = np.array(s["delay"])
        out[name] = {"delay": float(d.mean()) if len(d) else float("nan"),
                     "switches": s["switches"], "energy": s["energy"]}
    # also report what thresholds the adaptive one chose
    ad = [p for p in policies if p.name == "twin_gate_adaptive"][0]
    out["_adaptive_thresholds"] = ad.threshold_history
    return out


def average_over_seeds(delay, sigma, knn, n_seeds, duration_s):
    """
    Run the same scenario over several seeds and average. Returns, for
    each policy, the mean and std of delay / switches / energy across
    seeds -- so we report a stable number with its spread, not one run.
    """
    acc = {"twin_nogate": {"delay": [], "switches": [], "energy": []},
           "twin_gate": {"delay": [], "switches": [], "energy": []},
           "twin_gate_adaptive": {"delay": [], "switches": [], "energy": []},
           "direct": {"delay": [], "switches": [], "energy": []}}
    thr_all = []
    for sd in range(n_seeds):
        r = run_scenario(delay, sigma, knn, seed=sd, duration_s=duration_s)
        for name in acc:
            acc[name]["delay"].append(r[name]["delay"])
            acc[name]["switches"].append(r[name]["switches"])
            acc[name]["energy"].append(r[name]["energy"])
        if r["_adaptive_thresholds"]:
            thr_all.append(np.mean(r["_adaptive_thresholds"]))

    summ = {}
    for name, s in acc.items():
        summ[name] = {
            "delay": np.mean(s["delay"]),   "delay_sd": np.std(s["delay"]),
            "switches": np.mean(s["switches"]), "switches_sd": np.std(s["switches"]),
            "energy": np.mean(s["energy"]), "energy_sd": np.std(s["energy"]),
        }
    summ["_thr"] = np.mean(thr_all) if thr_all else float("nan")
    return summ


if __name__ == "__main__":
    import sys
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

    model = config.TRAFFIC_MODEL
    table = f"training_table_{model}_merged.csv"
    knn = AdaptiveThresholdKNN(table, k=3)

    header = (f"Adaptive gate end-to-end test  (traffic={model})\n"
              f"{n_seeds} seeds x {duration} ticks per scenario, "
              f"averaged (mean +/- std across seeds)\n")
    print(header)

    scenarios = [
        (2.0,  0.10),   # fresh, even   -> easy
        (5.0,  0.30),   # middle
        (10.0, 0.50),   # stale-ish, uneven
        (20.0, 0.65),   # hard
        (30.0, 0.70),   # very hard
    ]

    lines = [header]
    for delay, sigma in scenarios:
        s = average_over_seeds(delay, sigma, knn, n_seeds, duration)
        ng, fx, ad = s["twin_nogate"], s["twin_gate"], s["twin_gate_adaptive"]
        block = (
            f"\n=== scenario: telemetry_delay={delay}s, load_sigma={sigma} ===\n"
            f"  {'policy':<20}{'delay ms':>16}{'switches':>16}{'energy J':>16}\n"
            f"  {'no-gate':<20}"
            f"{ng['delay']:>8.0f}±{ng['delay_sd']:<6.0f}"
            f"{ng['switches']:>8.0f}±{ng['switches_sd']:<6.0f}"
            f"{ng['energy']:>8.2f}±{ng['energy_sd']:<6.2f}\n"
            f"  {'fixed gate (0.80)':<20}"
            f"{fx['delay']:>8.0f}±{fx['delay_sd']:<6.0f}"
            f"{fx['switches']:>8.0f}±{fx['switches_sd']:<6.0f}"
            f"{fx['energy']:>8.2f}±{fx['energy_sd']:<6.2f}\n"
            f"  {'ADAPTIVE (kNN)':<20}"
            f"{ad['delay']:>8.0f}±{ad['delay_sd']:<6.0f}"
            f"{ad['switches']:>8.0f}±{ad['switches_sd']:<6.0f}"
            f"{ad['energy']:>8.2f}±{ad['energy_sd']:<6.2f}\n"
            f"  (adaptive gate chose avg threshold {s['_thr']:.2f})\n"
        )
        print(block)
        lines.append(block)

    # save to file for the paper
    with open(f"RESULTS_{model}_multiseed.txt", "w") as f:
        f.write("".join(lines))
    print(f"\nsaved RESULTS_{model}_multiseed.txt")
