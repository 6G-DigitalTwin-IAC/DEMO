"""
batch_runner.py -- run many sweeps automatically, without babysitting.

WHAT IT DOES
  You give it a list of experiments. Each experiment says "set parameter P
  to value V, then run the full threshold sweep." The runner works through
  the whole list on its own, saving one result file per experiment, and
  keeps going even if one crashes.

WHY IT EXISTS
  So you never again hand-edit config.py, run, wait, come back, edit,
  repeat. Define the list once, start it, walk away.

WHAT IT WILL AND WON'T TOUCH
  It ONLY varies "condition" parameters -- the things a live network could
  actually measure while running (load, spread, drift speed, telemetry
  delay, buffer behaviour). It refuses to touch:
    - physics (light speed, gravity, Earth radius): laws, not settings
    - network design (satellite count, altitude, inclination, planes):
      fixed for this project; changing them = a different constellation,
      which is future work.
  This keeps every result comparable: same constellation, only conditions
  differ. If you ask it to vary a forbidden parameter it stops and tells
  you, so nobody pollutes the dataset by accident.

PARALLELISM
  The experiments are independent, so it runs several at once -- one per
  CPU core. This is the RIGHT kind of speedup for this problem (many
  separate jobs, no coordination). It is NOT a GPU job: the simulation is
  branchy decision logic, which GPUs are bad at. More CPU cores = faster.
  A laptop gives ~8x; a school compute server can give 32-64x. Ask them
  for CPU cores, not GPUs.

USAGE
  1. Edit the EXPERIMENTS section at the bottom.
  2. Run:  python batch_runner.py
     or:    python batch_runner.py 3 400      (seeds, ticks per run)
  3. Results land in ./batch_results/ , one JSON per experiment, each
     named by what was changed (e.g. BACKGROUND_LOAD_MEAN=0.60.json).
"""

import os
import sys
import json
import time
import traceback
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed


# =====================================================================
# WHAT'S ALLOWED TO CHANGE
# =====================================================================
# Only these condition parameters may be varied. Everything else is
# either physics or fixed network design and must stay constant so all
# runs are comparable.

ALLOWED_PARAMS = {
    "BACKGROUND_LOAD_MEAN",          # how busy the network is
    "BACKGROUND_LOAD_SIGMA",         # how uneven the links are
    "BACKGROUND_CORRELATION_TIME_S", # how fast load changes
    "TELEMETRY_DELAY_S",             # how stale the twin is
    "TELEMETRY_UPDATE_INTERVAL_S",   # how often the twin updates
    "BUFFER_DELAY_MS",               # congestion severity
    "BUFFER_EXPONENT",               # congestion onset shape
}

# Belt-and-braces: things that must NEVER be swept, with a clear reason.
FORBIDDEN_REASON = {
    "C_KM_S": "physics", "MU_EARTH": "physics", "R_EARTH_KM": "physics",
    "EARTH_ROT_RATE": "physics",
    "N_SATS": "network design (future work)",
    "NUM_PLANES": "network design (future work)",
    "SATS_PER_PLANE": "network design (future work)",
    "ALTITUDE_KM": "network design (future work)",
    "INCLINATION_DEG": "network design (future work)",
}


def _run_one(param, value, seeds, duration):
    """
    Run ONE experiment in its own fresh process: set `param`=`value` in
    config, run the sweep, save a result file. Returns a short status.

    Runs in a separate process (that's how we parallelise), so it imports
    and configures its own copy of everything -- no interference between
    concurrent runs.
    """
    try:
        # import fresh inside the worker process
        import config
        importlib.reload(config)

        # safety: never allow a forbidden or unknown parameter
        if param in FORBIDDEN_REASON:
            return (param, value, "SKIPPED",
                    f"forbidden ({FORBIDDEN_REASON[param]})")
        if param not in ALLOWED_PARAMS:
            return (param, value, "SKIPPED",
                    "not in ALLOWED_PARAMS (only measurable conditions)")

        # override the one parameter
        setattr(config, param, value)

        # re-derive anything that depends on config, then reload the
        # modules that read config at import time so they pick up the change
        import traffic, twin, simulate, sweep
        importlib.reload(traffic)
        importlib.reload(twin)
        importlib.reload(simulate)
        importlib.reload(sweep)

        ref, results = sweep.sweep(seeds, duration)

        out = {
            "config": sweep.snapshot_config(),   # full snapshot (includes the change)
            "changed": {"param": param, "value": value},
            "ref": ref,
            "results": {str(k): v for k, v in results.items()},
            "seeds": seeds,
            "duration_s": duration,
        }

        os.makedirs("batch_results", exist_ok=True)
        # name the file by what changed, e.g. BACKGROUND_LOAD_MEAN=0.60.json
        safe_val = str(value).replace("/", "_")
        fname = f"batch_results/{param}={safe_val}.json"
        with open(fname, "w") as f:
            json.dump(out, f, indent=2)

        return (param, value, "OK", fname)

    except Exception as e:
        # crash in one run must NOT stop the batch. Log and move on.
        os.makedirs("batch_results", exist_ok=True)
        with open("batch_results/_errors.log", "a") as f:
            f.write(f"\n=== {param}={value} FAILED ===\n")
            f.write(traceback.format_exc())
        return (param, value, "ERROR", str(e))


def run_batch(experiments, seeds=3, duration=400, max_workers=None):
    """
    experiments: list of (param_name, value) pairs to run.
    Runs them in parallel across CPU cores, saving one file each.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    print(f"Batch: {len(experiments)} runs, {seeds} seeds x {duration} ticks each")
    print(f"Using {max_workers} CPU core(s) in parallel")
    print(f"(To go faster, run this on a machine with more CPU cores.)\n")

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, p, v, seeds, duration): (p, v)
                   for (p, v) in experiments}
        for fut in as_completed(futures):
            param, value, status, detail = fut.result()
            done += 1
            print(f"[{done}/{len(experiments)}] {status:8} {param}={value}"
                  + (f"  -> {detail}" if status in ("OK", "ERROR", "SKIPPED") else ""))

    dt = time.time() - t0
    print(f"\nDone. {done} runs in {dt/60:.1f} min. Results in ./batch_results/")
    print("Any failures are logged in batch_results/_errors.log")


def build_screening(param_values):
    """
    Turn a {param: [values]} dict into a flat list of (param, value) runs
    for one-at-a-time screening. Each parameter is varied alone; the rest
    stay at their config defaults.
    """
    exps = []
    for param, values in param_values.items():
        for v in values:
            exps.append((param, v))
    return exps


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    # =================================================================
    # EDIT THIS: which conditions to screen, and at what values.
    # One parameter changes per run; everything else stays at default.
    # This is the "which parameters actually move the threshold?" test.
    # =================================================================
    SCREENING = {
        "BACKGROUND_LOAD_MEAN":          [0.30, 0.45, 0.60, 0.75, 0.90],
        "BACKGROUND_LOAD_SIGMA":         [0.05, 0.15, 0.30, 0.50, 0.70],
        "BACKGROUND_CORRELATION_TIME_S": [10.0, 20.0, 30.0, 45.0, 60.0],
        "TELEMETRY_DELAY_S":             [1.0, 5.0, 10.0, 20.0, 40.0],
        "BUFFER_DELAY_MS":               [100.0, 200.0, 300.0, 500.0],
        "BUFFER_EXPONENT":               [1.0, 2.0, 3.0],
    }

    experiments = build_screening(SCREENING)
    run_batch(experiments, seeds=seeds, duration=duration)
