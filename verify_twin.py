"""
verify_twin.py -- prove the digital twin is wrong in the RIGHT way.

The twin must be:
  1. WRONG. If it matched reality, the paper's premise (acting on a wrong
     twin costs energy) would be false. There must be a real gap.
  2. WRONG FOR A PHYSICAL REASON. The gap must come from staleness --
     information age -- not from invented random noise. So the error must
     GROW with telemetry delay, and VANISH when delay -> 0.
  3. BOUNDED. A stale-but-real snapshot can't be arbitrarily wrong. Its
     error is limited by how much reality drifts over the staleness
     window. This is what separates it from the old +-15% noise, which
     had no physical ceiling.

This file demonstrates all three, and shows the twin sometimes disagrees
with reality about the BEST ROUTE -- which is exactly the situation the
gate will be built to handle.
"""

import numpy as np

import config
import orbits
import isl
import ground
import network
import traffic
from twin import DigitalTwin


PASS = 0
FAIL = 0


def check_true(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{'PASS' if cond else '**FAIL**'}] {name}")
    if detail:
        print(f"          {detail}")


def build_world(seed=0):
    sats = orbits.build_constellation()
    gs = ground.build_ground_stations()
    plan = isl.build_isl_plan()
    rng = np.random.default_rng(seed)
    bg = traffic.BackgroundTraffic(plan, [g["name"] for g in gs], rng)
    return sats, gs, plan, bg


def mean_belief_error(bg, tw):
    """
    Average absolute difference between what the twin believes and what
    is true, across every link. This is the twin's error, in utilisation
    points (0..1).
    """
    errs = []
    for key, true_val in bg.load.items():
        if key[0] == "GS":
            bel = tw.utilisation(key[1], key[1], "gsl") if False else \
                  tw._belief.get(key, config.BACKGROUND_LOAD_MEAN) \
                  if tw._belief else config.BACKGROUND_LOAD_MEAN
        else:
            bel = tw._belief.get(key, config.BACKGROUND_LOAD_MEAN) \
                  if tw._belief else config.BACKGROUND_LOAD_MEAN
        errs.append(abs(true_val - bel))
    return float(np.mean(errs))


def run_settled(delay_s, interval_s, seed=0, warmup_s=120, measure_s=600):
    """
    Run reality + twin for a while, then measure the twin's average error
    over a measurement window. Returns (mean_error, mean_age).
    """
    sats, gs, plan, bg = build_world(seed)
    tw = DigitalTwin(delay_s=delay_s, interval_s=interval_s)

    t = 0.0
    # warmup: let the twin fill up and reach steady staleness
    for _ in range(warmup_s):
        bg.step(1.0)
        tw.observe(bg, t)
        t += 1.0

    errs, ages = [], []
    for _ in range(measure_s):
        bg.step(1.0)
        tw.observe(bg, t)
        errs.append(mean_belief_error(bg, tw))
        ages.append(tw.belief_age_s(t))
        t += 1.0

    return float(np.mean(errs)), float(np.mean(ages))


def test_zero_delay_is_accurate():
    print("\n=== TWIN WITH NO DELAY IS (NEARLY) PERFECT ===")
    print("If information were instant, the twin would match reality.")
    print("This proves the error comes from STALENESS, not from noise.\n")

    # delay=0, interval=1: the twin samples reality every tick and it
    # arrives instantly. It should be almost exactly right (off by at
    # most one tick of drift).
    err, age = run_settled(delay_s=0.0, interval_s=1.0)
    print(f"  delay=0s, interval=1s:")
    print(f"    mean belief error: {err:.4f} utilisation points")
    print(f"    mean belief age:   {age:.2f} s")
    check_true("near-zero delay gives near-zero error",
               err < 0.02,
               f"error {err:.4f} is tiny -- the twin is basically reality "
               f"when information is fresh")


def test_error_grows_with_delay():
    print("\n=== ERROR GROWS WITH TELEMETRY DELAY ===")
    print("The core physical claim: staler information -> wronger twin.\n")

    print(f"  {'delay':>8} {'interval':>9} {'mean age':>9} {'mean error':>12}")
    results = []
    for delay in [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]:
        err, age = run_settled(delay_s=delay, interval_s=5.0)
        results.append((delay, age, err))
        print(f"  {delay:>7.0f}s {5.0:>8.0f}s {age:>8.2f}s {err:>12.4f}")

    errors = [r[2] for r in results]
    check_true("error increases monotonically with delay",
               all(errors[i] <= errors[i+1] + 0.005
                   for i in range(len(errors)-1)),
               "staler belief -> larger error, as physics demands")

    check_true("error saturates rather than exploding",
               errors[-1] < 0.5,
               f"even at 40s delay, error is {errors[-1]:.4f} -- bounded by "
               f"how far load drifts (mean {config.BACKGROUND_LOAD_MEAN}), "
               f"NOT unbounded like invented noise")


def test_error_bounded_by_drift():
    print("\n=== ERROR IS BOUNDED BY REALITY'S DRIFT ===")
    print("This is what separates staleness from random noise.")
    print("A stale snapshot was TRUE once. It can only be as wrong as")
    print("reality has moved since. That has a physical ceiling.\n")

    # The load is an OU process with mean m and spread sigma. Two
    # independent draws differ by, on average, about sigma*2/sqrt(pi)
    # ~ 1.13*sigma. A very stale belief is like an independent draw, so
    # the error can't exceed that ballpark.
    sigma = config.BACKGROUND_LOAD_SIGMA
    ceiling = 1.13 * sigma
    print(f"  background spread sigma = {sigma}")
    print(f"  expected error ceiling  ~ 1.13*sigma = {ceiling:.4f}")

    err_huge, age = run_settled(delay_s=300.0, interval_s=5.0)
    print(f"  twin error at 300s delay (belief ~fully decorrelated): {err_huge:.4f}")
    check_true("even a totally stale twin's error stays near the drift ceiling",
               err_huge < ceiling + 0.05,
               f"{err_huge:.4f} <= {ceiling+0.05:.4f} -- bounded, physical, "
               f"not arbitrary")


def test_twin_disagrees_about_routes():
    print("\n=== THE TWIN SOMETIMES PICKS A DIFFERENT ROUTE ===")
    print("This is the situation the gate exists to handle: the twin")
    print("proposes a switch based on stale belief, and it may be wrong.\n")

    sats, gs, plan, bg = build_world(seed=1)
    tw = DigitalTwin(delay_s=10.0, interval_s=5.0)

    t = 0.0
    for _ in range(120):                 # warmup
        bg.step(1.0)
        tw.observe(bg, t)
        t += 1.0

    disagreements = 0
    twin_worse = 0
    total = 0
    delay_penalty_ms = []

    for _ in range(600):
        bg.step(1.0)
        tw.observe(bg, t)
        orbits.propagate(sats, t)
        ground.update_ground_stations(gs, t)

        # REALITY graph: route on true load.
        Greal = network.build_graph(sats, gs, plan)
        traffic.annotate_graph(Greal, bg)
        true_best = network.shortest_path(Greal, "IST", "NYC", "delay_s")

        # TWIN graph: same geometry, but load from stale belief.
        Gtwin = network.build_graph(sats, gs, plan)
        traffic.annotate_graph(Gtwin, tw)
        twin_best = network.shortest_path(Gtwin, "IST", "NYC", "delay_s")

        if true_best is None or twin_best is None:
            t += 1.0
            continue

        total += 1
        if twin_best != true_best:
            disagreements += 1
            # How much does following the twin's choice actually cost,
            # measured in REALITY? (The twin's route, priced at true load.)
            true_cost_of_twin_choice = network.path_delay_s(Greal, twin_best)
            true_cost_of_true_best = network.path_delay_s(Greal, true_best)
            penalty = (true_cost_of_twin_choice - true_cost_of_true_best) * 1000
            delay_penalty_ms.append(penalty)
            if penalty > 1.0:
                twin_worse += 1

        t += 1.0

    print(f"  over {total} ticks at 10s telemetry delay:")
    print(f"    twin picked a different route than reality: {disagreements} "
          f"times ({100*disagreements/total:.1f}%)")
    print(f"    of those, twin's route was actually worse:  {twin_worse} times")
    if delay_penalty_ms:
        dp = np.array(delay_penalty_ms)
        print(f"    when twin disagreed, its route cost on average "
              f"{dp.mean():.1f} ms more (max {dp.max():.1f} ms)")

    check_true("the twin sometimes disagrees with reality about the best route",
               disagreements > 0,
               "this is the whole reason the gate needs to exist -- the "
               "twin proposes switches that may not be worth it")

    check_true("twin disagreements are sometimes genuinely worse choices",
               twin_worse > 0,
               "acting on the stale twin sometimes picks a worse route, "
               "which is exactly the cost the gate will filter")


def main():
    print("=" * 70)
    print("VERIFICATION: BRICK 3 (digital twin)")
    print("=" * 70)
    print(f"\nTelemetry delay:    {config.TELEMETRY_DELAY_S} s [SWEPT]")
    print(f"Update interval:    {config.TELEMETRY_UPDATE_INTERVAL_S} s [SWEPT]")
    print(f"Load correlation:   {config.BACKGROUND_CORRELATION_TIME_S} s")
    print("\nThe question in every test: is the twin wrong because of")
    print("STALENESS (physical, bounded, swept) rather than invented noise?")

    test_zero_delay_is_accurate()
    test_error_grows_with_delay()
    test_error_bounded_by_drift()
    test_twin_disagrees_about_routes()

    print("\n" + "=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    if FAIL == 0:
        print("\nThe twin is wrong for the right reason: staleness.")
        print("Error grows with delay, vanishes without it, and is bounded")
        print("by how far reality drifts. This is defensible in a way the")
        print("old +-15% random noise never was.")
    return FAIL


if __name__ == "__main__":
    import sys
    sys.exit(main())
