"""
config.py -- every number the simulation uses, in one place.

RULES FOR THIS FILE:
  1. Everything is in REAL PHYSICAL UNITS.
       distances -> kilometres (km)
       times     -> seconds (s)
       data      -> bits, bits/second
       angles    -> degrees if the name says _DEG, radians otherwise
  2. Every number says WHERE IT CAME FROM. One of:
       [PHYSICS]  a constant of nature. Do not tune.
       [SOURCED]  a real published figure. Reference given.
       [DERIVED]  computed from other numbers here.
       [SWEPT]    we don't know it, so we test a range and report the
                  whole curve. Never a single guessed value.
       [ASSUMED]  a modelling choice we cannot source. Must be
                  justified in the paper's limitations section.
  3. If you change a number, update any docstring that quotes it.

SCOPE: Brick 1 (real network) + Brick 2 (traffic and congestion).
No digital twin, no gate yet.
"""

import numpy as np

# ===========================================================================
# PHYSICAL CONSTANTS  [PHYSICS]
# ===========================================================================

R_EARTH_KM = 6371.0
# Mean Earth radius (IUGG mean radius).

MU_EARTH = 398600.4418
# Earth's gravitational parameter, km^3/s^2. This is G*M_earth.
# It's the number that makes Kepler's laws produce real answers.
# Source: WGS-84 / EGM-96.

C_KM_S = 299792.458
# Speed of light in vacuum, km/s. Exact by definition.
# Correct for ISLs (they're in vacuum). For ground links the lower
# atmosphere slows light by ~0.03%, far below anything we resolve.

EARTH_ROT_RATE = 7.2921159e-5
# Earth's rotation, rad/s = 2*pi / 86164.1 s.
# That's one SIDEREAL day (rotation relative to the stars), not a 24h
# solar day. Sidereal is correct here because the satellites live in a
# star-fixed frame.


# ===========================================================================
# CONSTELLATION GEOMETRY -- Walker Delta  [SOURCED]
# ===========================================================================
# Walker Delta notation:  i : T / P / F
#     i = inclination (deg), T = total sats, P = planes, F = phasing
#
# OURS:  Walker Delta 53.0 : 264/12/1
#
# Starlink shell 1 is Walker Delta 53.0 : 1584/72/39. Same inclination,
# same altitude, same family. Ours is smaller so it runs fast.
#
# WHY NOT SMALLER STILL? We tried 40 sats (5 planes x 8). It is
# GEOMETRICALLY IMPOSSIBLE: 8 sats/plane sit 5297 km apart and 5 planes
# leave an 8136 km equatorial gap. Both exceed any laser terminal's
# reach. A 40-sat shell at 550 km cannot form a +Grid at all.
# 22 sats/plane (1970 km) and 12 planes (2315-3680 km) both close.
# That is why Starlink uses 22 per plane.
# ===========================================================================

INCLINATION_DEG = 53.0
# [SOURCED] Starlink shell 1 inclination.
# Concentrates coverage on mid-latitudes where people live.
# CONSEQUENCE: satellites never exceed 53 deg latitude. Ever.

NUM_PLANES = 12
# [ASSUMED] Scaled down from Starlink's 72. Chosen as the smallest
# value where inter-plane links stay within terminal range.

SATS_PER_PLANE = 22
# [SOURCED] Starlink shell 1 uses 22 satellites per plane.
# Gives 1970 km intra-plane spacing, comfortably closeable.

N_SATS = NUM_PLANES * SATS_PER_PLANE      # [DERIVED] = 264

PHASING_F = 1
# [ASSUMED] Staggers neighbouring planes so satellites don't line up in
# rows. F=0 would put them in lockstep. Must be an integer in 0..P-1.

ALTITUDE_KM = 550.0
# [SOURCED] Starlink shell 1 altitude.

ORBIT_R_KM = R_EARTH_KM + ALTITUDE_KM
# [DERIVED] = 6921.0 km. Measured from Earth's CENTRE, not the surface.
# This is what goes into the orbital mechanics. Forgetting to add
# R_EARTH here is the classic beginner bug.


# ===========================================================================
# INTER-SATELLITE LINKS (ISL) -- "+Grid" topology
# ===========================================================================
# Each satellite carries FOUR laser terminals aimed at four FIXED
# neighbours, wired at design time and never changed:
#
#     2 INTRA-plane : the sats ahead and behind in its own ring.
#                     NEVER break. Same ring, same speed, fixed angle
#                     -> the distance is literally constant. Verified
#                     to 0.000000 km variation over a full orbit.
#
#     2 INTER-plane : the nearest sat in the ring left and right.
#                     Distance varies (2315-3680 km here) as the tilted
#                     rings converge and diverge each orbit.
#
# A laser terminal is a physical telescope on a gimbal. You get four.
# You cannot conjure a fifth because a satellite drifted into range,
# and you do not re-aim at whoever is closest -- acquiring a lock takes
# seconds to minutes, so partners are fixed for the mission.
# ===========================================================================

ISL_TERMINALS_PER_SAT = 4
# [SOURCED, with caveat] Chaudhry & Yanikomeroglu (arXiv:2103.00056)
# state Starlink satellites are expected to carry four laser terminals
# for four simultaneous links. Note: current V2 Mini hardware reportedly
# carries THREE. The 4-terminal +Grid is the standard modelling
# assumption in the LEO routing literature. Worth a footnote.

ISL_MAX_RANGE_KM = 5400.0
# [SOURCED] Starlink's routing is reported to account for a maximum link
# distance of 5,400 km.
# In our constellation the longest planned link is ~3680 km, so range is
# never the binding constraint. The check stays because it must hold for
# any reconfiguration.

ISL_POLAR_CUTOFF_DEG = 70.0
# [ASSUMED] Above this |latitude|, inter-plane links switch off.
#
# WHY: near the poles the orbital planes converge and cross. Satellites
# in neighbouring planes swing from far apart to nearly touching, fast.
# The terminal would have to slew at an impossible rate. Real systems
# switch the link off.
#
# CRITICAL NOTE: at 53 deg inclination, satellites NEVER exceed 53 deg
# latitude, so this cutoff NEVER FIRES in our runs. That is honest and
# deliberate. The check is correct and ready for a polar shell.
# DO NOT lower it to make links break -- that would be fabricating a
# physical effect that does not exist at this inclination.

ISL_DISABLE_SEAM = True
# [SOURCED-ish] The "seam" is the boundary between the last plane and
# the first. There, neighbouring satellites travel in OPPOSITE
# directions (one side ascending, the other descending). Relative
# velocity ~15 km/s -- twice orbital speed. No laser tracks that.
# Real Walker Delta constellations leave the seam unlinked. It is a
# well-documented routing headache.
# CONSEQUENCE: 44 satellites (2 planes x 22) have 3 links, not 4.


# ===========================================================================
# GROUND-SATELLITE LINKS (GSL)
# ===========================================================================

MIN_ELEVATION_DEG = 25.0
# [SOURCED] Commonly cited Starlink user terminal minimum elevation.
# WHY it exists: at low elevation the signal crosses ~10x more
# atmosphere (rain fade, scintillation), slant range grows from 550 km
# to ~2600 km, and buildings/hills sit near the horizon.

GSL_MAX_LINKS = 2
# [ASSUMED] How many satellites one ground station tracks at once.
# Real gateways have multiple dishes so they can hand over without
# dropping. 2 is conservative.

GROUND_STATIONS = {
    "IST": (41.0082, 28.9784),    # Istanbul, Turkiye
    "NYC": (40.7128, -74.0060),   # New York, USA
}
# (lat_deg, lon_deg) on the ROTATING Earth.
# ~7500 km apart, both near 41 deg N -- inside the 53 deg coverage band,
# far enough apart that a route needs several ISL hops, so there are
# real alternatives to choose between.


# ===========================================================================
# LINK CAPACITY
# ===========================================================================

ISL_CAPACITY_BPS = 100e9
# [SOURCED] Starlink states data transfer up to 100 Gbps on each laser
# link, with >99% link uptime reported on their 100G fleet in LEO.

GSL_CAPACITY_BPS = 4e9
# [ASSUMED] Radio to the ground, much slower than a laser in vacuum.
# Public per-gateway figures are not clean. Set well below the ISL rate
# so the ground link is the bottleneck -- which is true in real systems.
# CONSEQUENCE: the first and last hop dominate. Justify in the paper.

PACKET_SIZE_BITS = 1500 * 8
# [SOURCED] Standard Ethernet MTU, 1500 bytes.


# ===========================================================================
# CONGESTION MODEL
# ===========================================================================
# ==========================================================================
# WE TRIED M/M/1 FIRST AND IT COLLAPSES. THIS BELONGS IN THE PAPER.
# ==========================================================================
# The textbook queueing model gives  Wq = (1/mu) * rho/(1-rho).
#
# For a 100 Gbps ISL with 1500-byte packets, mu = 8,333,333 packets/s,
# so the link swallows a packet in 0.12 MICROseconds. Meanwhile light
# needs 6.57 ms to cross a 1970 km intra-plane link.
#
# Measured M/M/1 queueing delay at 100 Gbps:
#     rho = 0.50  ->  0.0001 ms   (0.00% of one hop's propagation)
#     rho = 0.90  ->  0.0011 ms   (0.02%)
#     rho = 0.99  ->  0.0119 ms   (0.18%)   <- link effectively DEAD
#
# So M/M/1 says a dying link adds one fifth of one percent to a hop.
# Congestion would be invisible, routes would never change, and there
# would be no paper. This is not an arithmetic slip -- the queue is
# ~55,000x faster than the speed of light over these distances. It
# simply cannot compete.
#
# ==========================================================================
# WHAT REALLY HAPPENS: BUFFERBLOAT (measured, not assumed)
# ==========================================================================
# M/M/1 assumes Poisson arrivals -- smooth, memoryless traffic. Real
# traffic is BURSTY and real routers carry DEEP BUFFERS to absorb the
# bursts. Under sustained load the buffer fills and stays full, and
# every packet behind it waits for it to drain.
#
# This is measured reality for Starlink. A measurement study
# (arXiv:2310.09242) found RTT inflates ~2-4x under load, reaching
# almost 400-500 ms, against a baseline SpaceX reports as ~33 ms
# median. That is hundreds of ms of queueing -- four orders of
# magnitude above M/M/1's prediction.
#
# NOTE FOR THE PAPER: SpaceX has since right-sized buffers and improved
# queueing, cutting median latency 48.5 -> 33 ms and p99 from >150 ms to
# <65 ms. The worst bufferbloat is being actively engineered away. Our
# model targets the measured inflation regime and should say so.
# ==========================================================================

BUFFER_DELAY_MS = 200.0
# [SWEPT, centred on measurement] Queueing delay at full saturation.
# Anchored to the measured 400-500 ms loaded RTT vs ~33 ms baseline,
# i.e. a few hundred ms of one-way queueing spread across a path.
# This is the single most important number in Brick 2, so it is SWEPT,
# not fixed. See sweep_congestion.py.

BUFFER_EXPONENT = 2.0
# [ASSUMED] Shape of the fill curve: occupancy = rho ** exponent.
# 1.0 = linear, 2.0 = delay stays low while quiet then climbs sharply.
# The exponent controls WHERE congestion starts to bite. Also swept.

MAX_UTILISATION = 0.99
# [ASSUMED] Clamp. Beyond this a real link drops packets rather than
# queueing forever. Prevents infinities poisoning path computations.


# ===========================================================================
# BACKGROUND TRAFFIC
# ===========================================================================
# The load everyone else puts on the network. Without it, every link
# except our own path sits at rho = 0, every alternative route looks
# perfect, and there is nothing to route around. Real constellations
# carry traffic from everywhere.

BACKGROUND_LOAD_MEAN = 0.45
# [SWEPT] Average link utilisation before our flow.
# This decides whether the network is quiet or busy, which decides
# whether routing is interesting at all. NEVER guess this -- sweep it.

BACKGROUND_LOAD_SIGMA = 0.15
# [SWEPT] How much links differ from each other. Zero would make every
# link identical and every route equivalent.

BACKGROUND_CORRELATION_TIME_S = 30.0
# [ASSUMED] How long a link "remembers" its load, in seconds.
#
# THIS IS THE MOST IMPORTANT PARAMETER IN THE WHOLE PROJECT and it is
# worth understanding why.
#
# If load jumped to a fresh random value every tick, it would be white
# noise -- unpredictable IN PRINCIPLE. No twin could ever forecast it,
# however perfect. Our paper's question ("what happens when the twin is
# wrong?") would be rigged: the twin would be wrong always, by
# construction, and the finding would be trivial and worthless.
#
# Real load is autocorrelated. A link busy now is probably still busy
# in 5 seconds and probably different in 5 minutes. THAT is what makes
# prediction meaningful but imperfect -- the regime we actually study.
#
# 30 s is chosen to be the same order as the telemetry delay we'll add
# in Brick 3, so staleness matters but isn't hopeless.


# ===========================================================================
# SIMULATION TIMING
# ===========================================================================

TICK_S = 1.0
# One simulation step = one second of real time.
#
# THIS FIXES THE OLD CODE'S WORST BUG. The old sim had
# "OMEGA = 0.015 rad per frame" -- a frame was not a unit of anything --
# yet it reported delays in seconds. Every number was meaningless.
# Now everything is in real seconds and checkable against physics.

SIM_DURATION_S = 3600.0
# Default run length. 1 hour = ~0.63 orbits at this altitude.


# ===========================================================================
# REPRODUCIBILITY
# ===========================================================================

RANDOM_SEED = 0
# Brick 1 is fully deterministic. Brick 2's background traffic is not,
# so results must always be reported over multiple seeds.


# ===========================================================================
# DERIVED VALUES -- computed from the above. Never set by hand.
# ===========================================================================

INCLINATION_RAD = np.radians(INCLINATION_DEG)

ORBITAL_PERIOD_S = 2.0 * np.pi * np.sqrt(ORBIT_R_KM ** 3 / MU_EARTH)
# Kepler's third law. ~5730 s = 95.5 min.

MEAN_MOTION_RAD_S = 2.0 * np.pi / ORBITAL_PERIOD_S

ORBITAL_SPEED_KM_S = np.sqrt(MU_EARTH / ORBIT_R_KM)
# vis-viva for a circular orbit. ~7.589 km/s.


# ===========================================================================
# DIGITAL TWIN  [BRICK 3]
# ===========================================================================
# The twin is a model of the network that lives on the ground. It does NOT
# see reality directly. It sees telemetry reports that arrived LATE, and
# between reports it has to guess.
#
# This is the heart of the paper. The twin is wrong for a PHYSICAL reason
# -- information takes time to travel -- not because we sprinkled random
# noise on it. Nobody can argue with the speed of light.

TELEMETRY_DELAY_S = 5.0
# [SWEPT] How old the twin's information is, in seconds.
#
# WHERE THIS COMES FROM PHYSICALLY:
#   - light up+down to a 550 km satellite: ~3.7 ms round trip
#   - the report waits its turn in the telemetry pipeline
#   - the ground processes it and updates the twin
#   - decisions travel back up
# A few seconds end-to-end is a reasonable operational figure. But we do
# not trust one value: this is THE swept parameter. The headline figure of
# the paper is "gate benefit vs. telemetry delay".
#
# WHY IT MATTERS: our background load has a correlation time of 30 s. If
# telemetry delay is tiny next to 30 s, the twin is nearly right and the
# gate has little to do. If it's comparable to or bigger than 30 s, the
# twin is guessing and the gate earns its keep. The interesting regime is
# where these two numbers are the same order -- which is why the default
# sits a few seconds below the 30 s correlation time.

TELEMETRY_UPDATE_INTERVAL_S = 5.0
# [SWEPT] How often a fresh report arrives, in seconds.
#
# Real telemetry is not continuous. Reports arrive in batches. Between
# batches the twin is stuck with its last picture, getting staler by the
# second. With interval = delay = 5 s, at the moment a new report lands
# it is already 5 s old, and it ages to 10 s before the next one arrives.
#
# Setting interval = delay is the simplest honest choice. Both are swept.

TWIN_HOLDS_LAST = True
# [ASSUMED] Between reports, what does the twin believe?
#   True  -> it holds the last report it got (a stale but real snapshot).
#   False -> it could extrapolate (predict forward). We do NOT do that in
#            Brick 3, deliberately: "hold last value" is the honest floor.
#            Extrapolation is a smarter twin, and comparing against it is
#            future work. Starting simple keeps the finding clean.


# ===========================================================================
# ENERGY MODEL  [BRICK 4]
# ===========================================================================
# THE OLD CODE'S WORST CHEAT LIVED HERE. It had ENERGY_PER_SWITCH = 5e-3,
# a pure invented constant, and the RATIO between per-km cost and per-switch
# cost WAS the entire finding. Pick a big switch cost -> switches look
# expensive -> gate looks great. That is unfalsifiable and indefensible.
#
# Brick 4 fixes this by grounding every joule in a SOURCED physical rate
# and SWEEPING the one soft number, so the result cannot depend on a magic
# constant.
#
# THE CLAIM WE ARE SUPPORTING: "unnecessary reroutes waste energy."
# That is true at the JOULE level, which is all we model here. The deeper
# chain (joules -> battery discharge -> cycle aging -> satellite lifetime)
# is how the field ultimately frames energy [Yan et al., JSAC 2016; optical
# SGION work, JOCN 2025], but it needs a battery + eclipse model we do not
# build. We NAME it as future work and model the direct energy only.

ENERGY_PER_FLOP_J = 5e-9
# [SOURCED] Energy per floating-point operation on a space-qualified
# processor. Modern space processors sit in the 4-7 x 10^-9 J/FLOP range,
# with 5e-9 as the representative mid value.
#
# SOURCE TO CITE: Veeravalli, "Resource-Aware Task Allocator Design..."
# (arXiv:2601.06706, 2026), which adopts 5e-9 J/FLOP and attributes the
# 4-7e-9 range to its refs [23,24]. IMPORTANT: that paper post-dates this
# work's preparation -- READ IT AND ITS refs [23,24] YOURSELF before
# citing, and cite the primary sources [23,24] for the number, not just
# the secondary paper. Cross-check figure: the same paper reports modern
# LEO satellites carry 10-50 GFLOPS of compute, which makes a
# millisecond-scale table recompute cost tens of millions of FLOPs --
# squarely inside our swept range below. Good, our sweep brackets reality.
#
# The 4-7 range is SWEPT in sweep_energy.py so no result hangs on 5e-9.

# INDEPENDENT EVIDENCE THAT RECONFIGURATION ISN'T FREE (different unit):
# The same literature reports that dynamic network reconfiguration schemes
# "typically incur 200-500 ms reconfiguration overhead" (arXiv:2601.06706,
# citing Bertaux et al. 2015 for SDN control, which adds 5-10% control
# traffic). That's reconfiguration cost measured in TIME and SIGNALLING,
# not joules -- but it's citable support for our core premise that route
# changes carry a real, non-negligible cost. Worth citing alongside the
# energy argument.

RECONFIG_FLOPS_PER_NODE = 1e6
# [SWEPT] FLOPs a single satellite spends to update its forwarding state
# when a route change touches it: recomputing next-hop, rewriting the
# forwarding table, preparing signalling to neighbours.
#
# We cannot source this precisely -- it depends on table size and
# implementation -- so it is the PRIMARY swept parameter of Brick 4.
# 1e6 (a million FLOPs per node reconfiguration) is a plausible centre.
# The paper reports how the result varies across a wide range of it. If
# the gate helps across the whole range, the finding is independent of
# this number, which is the entire point.

TRANSMISSION_POWER_W = 5.0
# [ASSUMED] Optical ISL transmit power, watts. Public figures for laser
# terminal draw are scarce; single-digit watts is the commonly quoted
# scale for a LEO optical terminal. Used only for transmission energy,
# which is nearly CONSTANT across route choices (you send the same data
# either way), so it barely affects the gate comparison. Included for
# completeness, not because it drives the result. Also swept.

# NOTE ON WHAT COSTS WHAT, so nobody gets confused later:
#   TRANSMISSION energy ~ how many hops x how long x power. Nearly the
#       same whichever route you pick, because it's the same data. A
#       route change barely moves it.
#   RECONFIGURATION energy ~ how many NODES changed their forwarding
#       state x FLOPs x J/FLOP. This is ZERO when you don't switch and
#       positive when you do. THIS is the cost of an unnecessary reroute,
#       and THIS is what the gate saves.


# ===========================================================================
# CONTROL GATE  [BRICK 5] -- the contribution
# ===========================================================================
# The twin proposes route switches from stale belief. Brick 3 measured that
# ~46% of those proposals (at 10s delay) would move us to a WORSE route.
# The gate is a filter: it only allows a proposed switch if the predicted
# improvement is big enough to be worth the reconfiguration cost and the
# risk that the twin is wrong.
#
# THE RULE (deliberately simple and single-valued -- the old code had two
# contradictory versions):
#
#     switch  only if   predicted_new_delay  <  GATE_THRESHOLD * current_delay
#
# GATE_THRESHOLD is a fraction < 1. Read it as "the new route must be at
# least this much better before we bother".
#
#     threshold = 1.00  -> allow any improvement, however tiny (loosest)
#     threshold = 0.90  -> only switch if new route is >=10% better
#     threshold = 0.70  -> only switch if new route is >=30% better (tight)
#     threshold = 0.00  -> never switch (frozen)
#
# BOTH delays in the rule are the TWIN'S predictions -- because in reality
# the gate only has the twin's view to decide on. It cannot peek at truth.
# That's the honest constraint: the gate is as blind as the twin.

GATE_THRESHOLD = 0.90
# [SWEPT] The strictness knob. This is THE parameter of the paper. Its
# whole response curve -- delay, switches, energy vs. threshold -- is the
# result. Never reported as a single tuned value; the curve is the finding.

GATE_SWEEP_VALUES = [1.00, 0.98, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
# [SWEPT] The threshold values the sweep walks through. From "allow almost
# anything" down to "only switch for a huge win". Spaced tighter near 1.0
# because that's where behaviour changes fastest.
