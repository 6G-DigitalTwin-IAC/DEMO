"""
orbits.py -- where the satellites are, at any moment in time.

JOB: given a time t in seconds, work out the 3D position of every
satellite. Nothing else.

TWO FRAMES OF REFERENCE (this trips everyone up, read it once):

  ECI = Earth-Centered Inertial.
        Origin at Earth's centre, axes pointed at fixed stars.
        DOES NOT ROTATE. Satellites live here, because orbits are
        simple in a non-rotating frame.

  ECEF = Earth-Centered Earth-Fixed.
        Rotates with the Earth. Ground stations live here
        (Istanbul is always at 29 deg E).

We do the physics in ECI and convert ground stations INTO it. The
alternative -- converting satellites into ECEF -- means adding
fictitious forces. Don't.

MODELLED:
  - circular orbits (eccentricity = 0)
  - real Keplerian motion: correct period, correct speed
  - a rotating Earth underneath

NOT MODELLED (state as limitations in the paper):
  - J2 perturbation. Earth's equatorial bulge drags RAAN ~5 deg/day at
    550 km. Over a 1-hour run that's ~0.2 deg, and it affects all planes
    almost equally -- so RELATIVE topology, which is all routing cares
    about, barely moves.
  - atmospheric drag, solar radiation pressure, station-keeping burns
  - eccentricity (real orbits are slightly elliptical)
"""

import numpy as np

from config import (
    NUM_PLANES, SATS_PER_PLANE, N_SATS, PHASING_F,
    INCLINATION_RAD, INCLINATION_DEG, ORBIT_R_KM,
    MEAN_MOTION_RAD_S, ORBITAL_PERIOD_S, ORBITAL_SPEED_KM_S,
    EARTH_ROT_RATE, R_EARTH_KM,
)


def build_constellation():
    """
    Create the satellites and assign each one its orbit.

    WALKER DELTA CONSTRUCTION,  i : T/P/F

    Plane p of P gets:
        RAAN_p = p * 360/P

        RAAN = "Right Ascension of the Ascending Node". Plain English:
        which way the orbital ring is rotated around Earth's axis.
        Spreading planes evenly around 360 deg spreads coverage evenly.

    Satellite s of S in plane p gets starting angle along its ring:
        M0 = s * 360/S  +  p * F * 360/T
             ^^^^^^^^^^     ^^^^^^^^^^^^^
             even spacing    phasing offset between planes

    The phasing term is the entire point of F. With F=0 every plane is
    in lockstep and satellites line up in rows, leaving gaps. F=1 nudges
    each plane relative to its neighbour, staggering them. That is what
    makes it a *Delta* constellation.

    Returns: list of dicts. Positions are zero until propagate() runs.
    """
    sats = []

    for p in range(NUM_PLANES):
        raan = 2.0 * np.pi * p / NUM_PLANES

        for s in range(SATS_PER_PLANE):
            M0 = (2.0 * np.pi * s / SATS_PER_PLANE
                  + 2.0 * np.pi * PHASING_F * p / N_SATS)

            sats.append({
                "id": p * SATS_PER_PLANE + s,
                "plane": p,
                "slot": s,
                "raan": raan,
                "M0": M0,
                "pos_eci": np.zeros(3),
                "lat_deg": 0.0,
                "lon_deg": 0.0,
            })

    return sats


def propagate(sats, t_s):
    """
    Move every satellite to where it actually is at time t_s.

    THE MATH, for one satellite:

    1. How far around its ring is it now?
           M = M0 + n*t          (angle = start + rate * time)
       For a circular orbit this is exact: with e=0 the mean anomaly
       IS the true anomaly, so no Kepler equation to solve.

    2. Position in a flat 2D ring:
           x = r*cos(M),  y = r*sin(M),  z = 0

    3. Tilt the ring up by the inclination.
       Rotate about the X axis. X unchanged, Y and Z mix.

    4. Spin the tilted ring around Earth's axis by RAAN.
       Rotate about the Z axis. Z unchanged, X and Y mix.

    Modifies sats in place.
    """
    n = MEAN_MOTION_RAD_S
    cos_i = np.cos(INCLINATION_RAD)
    sin_i = np.sin(INCLINATION_RAD)

    for sat in sats:
        # 1. angle along the ring, right now
        M = sat["M0"] + n * t_s

        # 2. flat ring
        x_orb = ORBIT_R_KM * np.cos(M)
        y_orb = ORBIT_R_KM * np.sin(M)

        # 3. tilt by inclination (rotate about X)
        x_inc = x_orb
        y_inc = y_orb * cos_i
        z_inc = y_orb * sin_i

        # 4. rotate by RAAN (rotate about Z)
        cos_r = np.cos(sat["raan"])
        sin_r = np.sin(sat["raan"])
        x = x_inc * cos_r - y_inc * sin_r
        y = x_inc * sin_r + y_inc * cos_r
        z = z_inc

        sat["pos_eci"] = np.array([x, y, z])

        # --- sub-satellite point (lat/lon on Earth directly below) ---
        # Needed for the polar ISL cutoff and for drawing maps.

        # Latitude is frame-independent: Z is the same axis in ECI and ECEF.
        sat["lat_deg"] = float(np.degrees(np.arcsin(z / ORBIT_R_KM)))

        # Longitude is NOT. Correct for how far Earth has turned since t=0.
        lon_eci = np.arctan2(y, x)
        lon_ecef = lon_eci - EARTH_ROT_RATE * t_s
        sat["lon_deg"] = float(np.degrees(
            np.arctan2(np.sin(lon_ecef), np.cos(lon_ecef))
        ))


def walker_notation():
    """One-line description of this constellation, for the paper."""
    return (f"Walker Delta {INCLINATION_DEG:.1f}: "
            f"{N_SATS}/{NUM_PLANES}/{PHASING_F}")


def describe():
    """Human-readable summary. Print this at the top of any run."""
    return {
        "notation": walker_notation(),
        "n_satellites": N_SATS,
        "n_planes": NUM_PLANES,
        "sats_per_plane": SATS_PER_PLANE,
        "altitude_km": ORBIT_R_KM - R_EARTH_KM,
        "period_s": ORBITAL_PERIOD_S,
        "period_min": ORBITAL_PERIOD_S / 60.0,
        "speed_km_s": ORBITAL_SPEED_KM_S,
        "max_latitude_deg": INCLINATION_DEG,
    }
