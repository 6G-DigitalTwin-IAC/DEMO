"""
sweep.py -- the paper's headline experiment.

Runs the three policies across a range of gate thresholds, over MULTIPLE
SEEDS, and reports the mean and spread of each metric. This is what
produces the tradeoff curve: delay, switches, and energy vs. threshold.

WHY MULTI-SEED: the background traffic is random. One seed is one sample
of a random process -- reporting it as "the" answer is the mistake that
produced the shaky number in the original abstract. Every point here is a
mean over seeds, with the spread shown, so the curve is a real trend and
not noise.

WHY SHARED SEEDS ACROSS POLICIES: all three policies run on the SAME seed
(the same congestion patterns), so differences between them are the
policy's doing, not luck. Paired comparison.

Usage:
    python sweep.py                # default: quick, few seeds, short runs
    python sweep.py 10 1800        # 10 seeds, 1800-tick runs
"""

import sys
import json
import numpy as np

import config
import simulate


def sweep(seeds, duration_s, thresholds=None, telemetry_delay_s=None):
    thresholds = thresholds or config.GATE_SWEEP_VALUES

    # results[threshold][policy][metric] = list over seeds
    results = {}

    # reference policies don't depend on threshold, so run them once per seed
    ref = {"direct": {"mean_delay_ms": [], "p95_delay_ms": [],
                      "switches": [], "reconfig_energy_j": []},
           "twin_nogate": {"mean_delay_ms": [], "p95_delay_ms": [],
                           "switches": [], "reconfig_energy_j": []}}

    print(f"seeds={seeds}, duration={duration_s}s, "
          f"telemetry_delay={telemetry_delay_s or config.TELEMETRY_DELAY_S}s")
    print(f"thresholds={thresholds}\n")

    for th in thresholds:
        pol = {"mean_delay_ms": [], "p95_delay_ms": [],
               "switches": [], "reconfig_energy_j": []}
        for sd in range(seeds):
            r = simulate.run(threshold=th, seed=sd, duration_s=duration_s,
                             telemetry_delay_s=telemetry_delay_s)
            g = r["twin_gate"]
            for k in pol:
                pol[k].append(g[k])
            # collect references on the first threshold pass only
            if th == thresholds[0]:
                for name in ref:
                    for k in ref[name]:
                        ref[name][k].append(r[name][k])
        results[th] = pol
        print(f"  threshold {th:.2f} done")

    return ref, results


def summarize(ref, results):
    def ms(lst):
        a = np.array(lst, dtype=float)
        return f"{a.mean():7.1f} ± {a.std():4.1f}"

    print("\n" + "=" * 74)
    print("SWEEP RESULTS (mean ± std over seeds)")
    print("=" * 74)

    print("\nReference policies:")
    print(f"  {'policy':14} {'delay ms':>16} {'switches':>14} {'energy J':>16}")
    for name in ["direct", "twin_nogate"]:
        r = ref[name]
        print(f"  {name:14} {ms(r['mean_delay_ms']):>16} "
              f"{ms(r['switches']):>14} {ms(r['reconfig_energy_j']):>16}")

    print("\nGate, by threshold:")
    print(f"  {'thresh':>7} {'delay ms':>16} {'switches':>14} {'energy J':>16}")
    for th in sorted(results.keys(), reverse=True):
        p = results[th]
        print(f"  {th:>7.2f} {ms(p['mean_delay_ms']):>16} "
              f"{ms(p['switches']):>14} {ms(p['reconfig_energy_j']):>16}")

    # headline numbers vs no-gate baseline
    nogate_sw = np.mean(ref["twin_nogate"]["switches"])
    nogate_e = np.mean(ref["twin_nogate"]["reconfig_energy_j"])
    nogate_d = np.mean(ref["twin_nogate"]["mean_delay_ms"])
    print("\nGate benefit vs. twin-no-gate (same twin, gate off):")
    print(f"  {'thresh':>7} {'switch cut':>12} {'energy cut':>12} {'delay cost':>12}")
    for th in sorted(results.keys(), reverse=True):
        p = results[th]
        sw = np.mean(p["switches"])
        e = np.mean(p["reconfig_energy_j"])
        d = np.mean(p["mean_delay_ms"])
        print(f"  {th:>7.2f} {(1-sw/nogate_sw)*100:>10.1f}% "
              f"{(1-e/nogate_e)*100:>10.1f}% {(d/nogate_d-1)*100:>10.1f}%")


def snapshot_config():
    """
    Grab every value defined in config.py, automatically.

    Returns a plain dict {parameter_name: value} of every UPPER_CASE
    setting in config -- constants, sourced values, swept values, all of
    them. We don't hand-pick: we take everything, so the record is
    complete and stays correct even when new parameters are added later.

    Only plain numbers, strings, and simple containers are kept (so the
    result is safe to write to JSON). Things like numpy arrays or the
    ground-station dict are converted to a JSON-friendly form.
    """
    import config as _cfg

    def jsonable(v):
        # numbers and strings pass straight through
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        # numpy scalars -> python numbers
        try:
            import numpy as _np
            if isinstance(v, _np.floating):
                return float(v)
            if isinstance(v, _np.integer):
                return int(v)
            if isinstance(v, _np.ndarray):
                return v.tolist()
        except Exception:
            pass
        # dicts / lists / tuples -> recurse
        if isinstance(v, dict):
            return {str(k): jsonable(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [jsonable(x) for x in v]
        # anything else -> its text form, so we still record SOMETHING
        return str(v)

    snap = {}
    for name in dir(_cfg):
        if name.startswith("_"):
            continue                      # skip private / dunder names
        if not name.isupper():
            continue                      # config settings are UPPER_CASE
        snap[name] = jsonable(getattr(_cfg, name))
    return snap


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 900

    ref, results = sweep(seeds, duration)
    summarize(ref, results)

    # save raw for later plotting AND the full config that produced it,
    # so every result file is self-documenting. This is what lets the
    # adaptive-threshold work later match "conditions -> best threshold".
    out = {
        "config": snapshot_config(),          # <-- every parameter + value
        "ref": ref,
        "results": {str(k): v for k, v in results.items()},
        "seeds": seeds,
        "duration_s": duration,
    }
    with open("sweep_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved sweep_results.json  (now includes full config snapshot)")
