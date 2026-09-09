"""
build_table.py -- generate the training table for the adaptive system.

WHAT IT DOES
  Runs the simulation across a grid of the TWO parameters that actually
  move the best threshold -- telemetry delay (how stale the twin is) and
  load sigma (how uneven the links are) -- and records the best gate
  threshold for each combination.

  The output is a simple CSV, one row per situation:

      telemetry_delay_s, load_sigma, best_threshold, switch_cut_pct,
      energy_cut_pct, delay_cost_pct, traffic_model

  This table IS the training data for the kNN. Hand the CSV to whoever
  builds the kNN; it needs nothing else.

"BEST THRESHOLD" DEFINITION (must be identical for every row):
  the STRICTEST threshold whose delay cost stays under DELAY_BUDGET_PCT.
  i.e. cut as many switches / as much energy as possible while keeping
  delay near-optimal. This matches the paper's own claim.

USAGE
  python build_table.py                  # defaults: 5 seeds, 600 ticks
  python build_table.py 5 800            # seeds, ticks
  Set TRAFFIC_MODEL in config first (smooth or bursty), OR run once each
  and the CSV records which model produced each row.
"""

import sys
import csv
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import config


# ---- the grid: the two parameters that matter, at sensible steps ----
TELEMETRY_DELAYS = [1.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0]
LOAD_SIGMAS      = [0.05, 0.15, 0.25, 0.35, 0.50, 0.70]

DELAY_BUDGET_PCT = 2.0   # "near-optimal" = delay cost stays under this


def best_threshold_from(ref, results):
    """
    The one definition of 'best', applied identically everywhere:
    strictest threshold whose delay cost <= DELAY_BUDGET_PCT.
    Returns (best_threshold, switch_cut%, energy_cut%, delay_cost%).
    """
    ng = ref["twin_nogate"]
    ng_sw = np.mean(ng["switches"])
    ng_e = np.mean(ng["reconfig_energy_j"])
    ng_d = np.mean(ng["mean_delay_ms"])

    best = None
    for th in sorted([float(k) for k in results], reverse=True):
        r = results[th]
        d = (np.mean(r["mean_delay_ms"]) / ng_d - 1) * 100
        sw = (1 - np.mean(r["switches"]) / ng_sw) * 100
        e = (1 - np.mean(r["reconfig_energy_j"]) / ng_e) * 100
        if d <= DELAY_BUDGET_PCT:
            best = (th, sw, e, d)   # keep tightening while under budget
    if best is None:               # nothing met budget: loosest is safest
        best = (1.0, 0.0, 0.0, 0.0)
    return best


def _one_cell(delay, sigma, seeds, duration, model):
    """Run one grid cell in its own process: set the two params, sweep,
    extract the best threshold. Returns a CSV-ready row."""
    import importlib
    import config as cfg
    importlib.reload(cfg)
    cfg.TELEMETRY_DELAY_S = delay
    cfg.BACKGROUND_LOAD_SIGMA = sigma
    cfg.TRAFFIC_MODEL = model

    import traffic, twin, simulate, sweep
    for m in (traffic, twin, simulate, sweep):
        importlib.reload(m)

    ref, results = sweep.sweep(seeds, duration)
    bt, sw, e, d = best_threshold_from(ref, results)
    return {
        "telemetry_delay_s": delay,
        "load_sigma": sigma,
        "best_threshold": bt,
        "switch_cut_pct": round(sw, 1),
        "energy_cut_pct": round(e, 1),
        "delay_cost_pct": round(d, 1),
        "traffic_model": model,
    }


def build(seeds, duration, model, workers=None):
    import os
    workers = workers or os.cpu_count() or 1
    cells = list(itertools.product(TELEMETRY_DELAYS, LOAD_SIGMAS))
    print(f"Building table: {len(cells)} cells, {seeds} seeds x {duration} "
          f"ticks, model='{model}', {workers} cores")

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_cell, d, s, seeds, duration, model): (d, s)
                for (d, s) in cells}
        done = 0
        for f in as_completed(futs):
            row = f.result()
            rows.append(row)
            done += 1
            print(f"  [{done}/{len(cells)}] delay={row['telemetry_delay_s']:>5} "
                  f"sigma={row['load_sigma']:>5} -> best {row['best_threshold']:.2f}")

    # sort for readability
    rows.sort(key=lambda r: (r["telemetry_delay_s"], r["load_sigma"]))

    fname = f"training_table_{model}.csv"
    with open(fname, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {fname}  ({len(rows)} rows) -- hand this to the kNN")
    return fname


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    model = config.TRAFFIC_MODEL   # whatever config is set to
    build(seeds, duration, model)
