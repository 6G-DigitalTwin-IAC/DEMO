"""
isl.py -- inter-satellite links. The "+Grid" topology.

=========================================================================
THIS FILE FIXES THE OLD CODE'S BIGGEST MISTAKE.
=========================================================================

OLD CODE:
    for every pair of satellites (i, j):
        if distance(i, j) < 2600 km:  create a link

Wrong, and wrong in a way that poisons everything downstream:
  - Each satellite got a varying number of links -- sometimes 2,
    sometimes 15 -- depending on who happened to drift nearby.
  - Links appeared and vanished constantly. The topology never settled.
  - It is physically impossible. A laser terminal is a telescope on a
    gimbal. You get FOUR. You cannot conjure a fifteenth.

WHAT REAL CONSTELLATIONS DO -- the "+Grid":

Four terminals per satellite, aimed at four FIXED neighbours, wired at
design time and never re-aimed:

                 plane p-1     plane p     plane p+1
                     |             |             |
    slot s-1   ---- sat --------- sat --------- sat ----
                     |             |             |
    slot s     ---- sat --------- ME ---------- sat ----
                     |             |             |
    slot s+1   ---- sat --------- sat --------- sat ----

  ME connects to:
    UP    (slot s-1, same plane)  <- intra-plane, "aft"
    DOWN  (slot s+1, same plane)  <- intra-plane, "fore"
    LEFT  (plane p-1, same slot)  <- inter-plane
    RIGHT (plane p+1, same slot)  <- inter-plane

  That's a "+". Hence +Grid.

WHY PARTNERS ARE FIXED AND NEVER SWITCH:
  Acquiring a laser lock means aiming a beam at a target thousands of km
  away, both of you moving at 7.6 km/s, hitting something the size of a
  dinner plate. It takes seconds to minutes. If satellites re-shopped
  for closer partners, they'd spend their lives acquiring locks instead
  of carrying data. So: pick a partner, keep it, live with the distance
  changing.

THE TWO KINDS BEHAVE COMPLETELY DIFFERENTLY:

  INTRA-PLANE: NEVER BREAKS.
      Same ring, same speed, fixed angle apart. The distance is
      CONSTANT -- not approximately, exactly. Verified to 0.000000 km
      over a full orbit. Like two horses on a carousel.

  INTER-PLANE: VARIES.
      Neighbouring rings are tilted relative to each other, so the two
      satellites converge and diverge every orbit. Measured here:
      closest ~2316 km at 53 deg latitude, farthest ~3680 km at the
      equator. Can break via the polar cutoff or the range limit --
      though at 53 deg inclination, neither ever fires. See config.py.
"""

import numpy as np

from config import (
    NUM_PLANES, SATS_PER_PLANE, N_SATS,
    ISL_MAX_RANGE_KM, ISL_POLAR_CUTOFF_DEG, ISL_DISABLE_SEAM,
    C_KM_S,
)


def sat_id(plane, slot):
    """Satellite id from (plane, slot). Must match orbits.build_constellation()."""
    return plane * SATS_PER_PLANE + slot


def build_isl_plan():
    """
    Decide the FIXED wiring: who aims a laser at whom.

    Called ONCE at startup. Never changes.

    Returns: list of (sat_a, sat_b, kind), kind in {"intra", "inter"}.
             Each undirected link appears once, with a < b.

    THE SEAM -- AND WHY OUR CONSTELLATION DOESN'T HAVE ONE:

      A "seam" is a boundary where two neighbouring orbital planes run
      in OPPOSITE directions (one ascending, one descending). Relative
      velocity there is about twice orbital speed, far too fast for a
      laser terminal to track, so no ISL can be maintained across it.

      CRITICAL: seams are a Walker STAR feature, NOT Walker Delta.
        Walker STAR  spreads ascending nodes over 180 deg, so wrapping
                     around puts you beside counter-rotating planes.
                     Iridium and OneWeb are Star -- they have seams.
        Walker DELTA spreads ascending nodes over the full 360 deg, so
                     every plane runs the SAME direction. No counter-
                     rotation anywhere. No seam.

      OURS IS WALKER DELTA, so ISL_DISABLE_SEAM is False and all 264
      satellites get their full 4 links.

      (An earlier version of this file wrongly claimed Walker Delta
      constellations leave a seam unlinked, and artificially cut 22
      links. That was incorrect and has been fixed. The flag remains so
      a Walker Star variant can be simulated later.)

      Sources: IETF draft-piraux-space-constellation-code-00;
      MATLAB walkerDelta documentation; arXiv:2209.05984.
    """
    links = []
    seen = set()

    for p in range(NUM_PLANES):
        for s in range(SATS_PER_PLANE):
            me = sat_id(p, s)

            # --- INTRA-PLANE: link to the next sat in my own ring ---
            # Only the "fore" link (s -> s+1). The "aft" link is the same
            # undirected edge as sat s-1's fore link -- adding both would
            # duplicate it. Every satellite still ends up with 2.
            # The modulo wraps the last slot to slot 0, closing the ring.
            fore = sat_id(p, (s + 1) % SATS_PER_PLANE)
            key = (min(me, fore), max(me, fore))
            if key not in seen:
                seen.add(key)
                links.append((key[0], key[1], "intra"))

            # --- INTER-PLANE: link to the same slot in the next ring ---
            # Again only "right"; "left" is someone else's right link.
            is_seam = (p == NUM_PLANES - 1)      # wrapping 11 -> 0

            if is_seam and ISL_DISABLE_SEAM:
                continue

            right = sat_id((p + 1) % NUM_PLANES, s)
            key = (min(me, right), max(me, right))
            if key not in seen:
                seen.add(key)
                links.append((key[0], key[1], "inter"))

    return links


def active_links(sats, isl_plan):
    """
    Given where the satellites ARE right now, which planned links are up?

    intra -> always up. Nothing can break it.
    inter -> up unless too polar, or out of range.

    Returns: list of (a, b, kind, dist_km, prop_s)
             prop_s = one-way propagation delay = distance / c
    """
    out = []

    for (a, b, kind) in isl_plan:
        sa, sb = sats[a], sats[b]
        d_km = float(np.linalg.norm(sa["pos_eci"] - sb["pos_eci"]))

        if kind == "intra":
            # Same ring, constant spacing. Always up, no checks.
            out.append((a, b, kind, d_km, d_km / C_KM_S))
            continue

        # ---------- inter-plane checks ----------

        # POLAR CUTOFF. Near the poles the rings converge and the
        # pointing geometry changes too fast to hold a lock.
        # NOTE: at 53 deg inclination this NEVER fires -- satellites
        # never exceed 53 deg latitude. Correct and ready for a polar
        # shell. Do not lower it to force link breaks.
        if (abs(sa["lat_deg"]) > ISL_POLAR_CUTOFF_DEG
                or abs(sb["lat_deg"]) > ISL_POLAR_CUTOFF_DEG):
            continue

        # RANGE CUTOFF. Finite terminal reach.
        if d_km > ISL_MAX_RANGE_KM:
            continue

        out.append((a, b, kind, d_km, d_km / C_KM_S))

    return out


def plan_summary(isl_plan):
    """Static facts about the wiring. For verify.py and the paper."""
    intra = sum(1 for l in isl_plan if l[2] == "intra")
    inter = sum(1 for l in isl_plan if l[2] == "inter")
    return {
        "total_planned_links": len(isl_plan),
        "intra_plane_links": intra,
        "inter_plane_links": inter,
        "mean_terminals_per_sat": 2.0 * len(isl_plan) / N_SATS,
    }
