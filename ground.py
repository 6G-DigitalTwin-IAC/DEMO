"""
ground.py -- ground stations on a ROTATING Earth.

=========================================================================
WHAT THE OLD CODE GOT WRONG
=========================================================================
It did this once, at startup:

    A_pos = latlon_to_xyz(41.0, 29.0)     # Istanbul
    B_pos = latlon_to_xyz(40.7, -74.0)    # New York

and never touched them again. Istanbul was nailed to a fixed point in
space forever.

But the Earth spins. In the inertial frame -- the one the satellites
live in -- Istanbul moves EAST at 350.6 m/s. Over an hour that's
1262 km. Ignore it and your ground station connects to satellites it
physically cannot see.

Why 350 and not 465? The equator moves at 465 m/s. Istanbul is at
41 deg N, riding a smaller circle: cos(41) * 465 = 350.6.

=========================================================================
ELEVATION ANGLE -- what decides visibility
=========================================================================

              satellite
                 /|
                / |
               /  |  elevation angle
              /   |  (from the local horizontal)
             /____|______________
          station    horizon

    90 deg = straight overhead
    25 deg = our cutoff
     0 deg = on the horizon
    < 0    = below the horizon, physically invisible

WHY 25 AND NOT 0:
  - Air mass. At 90 deg you punch straight up through the atmosphere.
    At 5 deg you cut through ~10x as much. Rain fade and scintillation
    get much worse.
  - Slant range. A satellite on the horizon is ~2600 km away, not 550.
  - Obstacles. Buildings, hills, trees live near the horizon.
  - Terminals physically cannot point at the ground.
"""

import numpy as np

from config import (
    R_EARTH_KM, EARTH_ROT_RATE, C_KM_S,
    MIN_ELEVATION_DEG, GSL_MAX_LINKS, GROUND_STATIONS,
)


def build_ground_stations():
    """Create ground stations from config. Positions filled by update()."""
    return [
        {
            "name": name,
            "lat_deg": lat,
            "lon_deg": lon,          # fixed longitude ON the Earth
            "pos_eci": np.zeros(3),
        }
        for name, (lat, lon) in GROUND_STATIONS.items()
    ]


def gs_position_eci(lat_deg, lon_deg, t_s):
    """
    Where is this ground station in the INERTIAL frame at time t?

    Latitude never changes -- it isn't going north or south. But its
    longitude in the inertial frame grows as the Earth turns:

        lon_eci(t) = lon_earth + rotation_rate * t

    At t=0 we define the frames to line up.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg) + EARTH_ROT_RATE * t_s

    return np.array([
        R_EARTH_KM * np.cos(lat) * np.cos(lon),
        R_EARTH_KM * np.cos(lat) * np.sin(lon),
        R_EARTH_KM * np.sin(lat),
    ])


def update_ground_stations(gs_list, t_s):
    """Move every ground station to where it is at time t_s. In place."""
    for gs in gs_list:
        gs["pos_eci"] = gs_position_eci(gs["lat_deg"], gs["lon_deg"], t_s)


def elevation_deg(gs_pos, sat_pos):
    """
    How high above the station's local horizon is this satellite?

    THE MATH:
      "Local up" is the direction from Earth's centre to the station,
      normalised. (True for a spherical Earth, which we assume.)

      Take the vector station -> satellite, normalise, dot with up.
      That dot product is the SINE of the elevation angle: it's the
      component of the direction-to-satellite that points straight up.
      arcsin gives the angle.
    """
    gx, gy, gz = gs_pos[0], gs_pos[1], gs_pos[2]
    gn = (gx*gx + gy*gy + gz*gz) ** 0.5
    tx, ty, tz = sat_pos[0]-gx, sat_pos[1]-gy, sat_pos[2]-gz
    rng = (tx*tx + ty*ty + tz*tz) ** 0.5
    if rng < 1e-9:
        return 90.0
    sin_el = (tx*gx + ty*gy + tz*gz) / (rng * gn)
    if sin_el > 1.0: sin_el = 1.0
    elif sin_el < -1.0: sin_el = -1.0
    import math
    return math.degrees(math.asin(sin_el))


def visible_satellites(gs, sats):
    """
    Which satellites can this station actually use right now?

    Returns the best GSL_MAX_LINKS, highest elevation first.

    WHY NOT ALL: a real gateway has a fixed number of dishes. It tracks
    a couple of satellites, not the whole sky. Highest elevation is what
    real terminals pick -- best signal, and they stay up longest.

    Returns: list of (sat_id, dist_km, elevation_deg)
    """
    vis = []
    for sat in sats:
        el = elevation_deg(gs["pos_eci"], sat["pos_eci"])
        if el >= MIN_ELEVATION_DEG:
            d_km = float(np.linalg.norm(sat["pos_eci"] - gs["pos_eci"]))
            vis.append((sat["id"], d_km, el))

    vis.sort(key=lambda x: -x[2])
    return vis[:GSL_MAX_LINKS]


def active_gsls(gs_list, sats):
    """
    All ground-satellite links up right now.
    Returns: (gs_name, sat_id, dist_km, prop_s, elevation_deg)
    """
    out = []
    for gs in gs_list:
        for (sid, d_km, el) in visible_satellites(gs, sats):
            out.append((gs["name"], sid, d_km, d_km / C_KM_S, el))
    return out
