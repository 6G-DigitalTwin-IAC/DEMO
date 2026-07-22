"""
traffic.py -- Brick 2: who else is using the network, and what it costs.

=========================================================================
WHY THIS FILE EXISTS
=========================================================================
Brick 1 gave us propagation delay: distance / speed of light. That's the
physical floor, and it's also BORING. Satellites move slowly, so the
shortest-light-path stays the shortest-light-path for minutes. Nothing
would ever need to reroute. No digital twin needed. No paper.

Real links carry data, and data QUEUES. Queueing delay changes fast and
depends on traffic. THAT is what makes routing a live problem -- and
what the digital twin will have to predict.

=========================================================================
THE THREE DELAYS ON A LINK
=========================================================================

  1. PROPAGATION -- distance / c.
     Brick 1. ~6.6 ms across an intra-plane ISL. Fixed by geometry.
     Cannot be beaten, only avoided by taking shorter paths.

  2. TRANSMISSION -- time to push the packet onto the link.
     packet_size / capacity. 0.12 MICROseconds on a 100 Gbps link.
     Utterly negligible here. Included because omitting it would be
     wrong, not because it changes any result.

  3. QUEUEING -- time waiting behind other packets.
     Near zero when quiet. Hundreds of ms when congested.
     THIS is the part that varies, and the part the twin predicts.

Total link delay = 1 + 2 + 3.

=========================================================================
WHAT IS "UTILISATION"?
=========================================================================
The fraction of a link's capacity in use. Written rho.

    rho = 0.0  -> completely idle
    rho = 0.5  -> half full
    rho = 0.99 -> effectively saturated

One number, and it decides how long your packet waits.
"""

import numpy as np

from config import (
    ISL_CAPACITY_BPS, GSL_CAPACITY_BPS, PACKET_SIZE_BITS,
    BUFFER_DELAY_MS, BUFFER_EXPONENT, MAX_UTILISATION,
    BACKGROUND_LOAD_MEAN, BACKGROUND_LOAD_SIGMA,
    BACKGROUND_CORRELATION_TIME_S, TICK_S,
)


# =====================================================================
# LINK CAPACITY
# =====================================================================

def link_capacity_bps(kind):
    """
    How much can this link carry per second?

    ISL -> laser in vacuum. 100 Gbps. [SOURCED: Starlink]
    GSL -> radio through atmosphere. 4 Gbps. Much slower.

    CONSEQUENCE: the ground link is the bottleneck on every path. That's
    also true in real systems -- the first and last hop matter more than
    any middle hop.
    """
    return GSL_CAPACITY_BPS if kind == "gsl" else ISL_CAPACITY_BPS


def service_rate_pps(capacity_bps):
    """Packets per second this link can serve = capacity / packet size."""
    return capacity_bps / PACKET_SIZE_BITS


# =====================================================================
# THE THREE DELAY COMPONENTS
# =====================================================================

def transmission_delay_s(capacity_bps):
    """
    Time to push one packet onto the wire = packet_size / capacity.

    Time spent SENDING, as opposed to queueing which is time spent
    WAITING. Both are real, both accumulate along a path.

    On a 100 Gbps link with a 1500-byte packet: 0.12 microseconds.
    About 55,000x smaller than propagation. Never matters here.
    """
    return float(PACKET_SIZE_BITS / capacity_bps)


def mm1_queueing_delay_s(rho, capacity_bps):
    """
    The TEXTBOOK M/M/1 queueing delay.  Wq = (1/mu) * rho/(1-rho)

    ===================================================================
    WE USE THIS ONLY TO DEMONSTRATE THAT IT DOES NOT WORK HERE.
    This function exists so verify_traffic.py can prove the collapse.
    Do not route on it.
    ===================================================================

    At 100 Gbps with 1500-byte packets, mu = 8,333,333 packets/s, so the
    link swallows a packet in 0.12 microseconds. Light needs 6.57 ms to
    cross a 1970 km intra-plane link.

    Measured output of this function:
        rho = 0.50 -> 0.0001 ms   (0.00% of one hop's propagation)
        rho = 0.90 -> 0.0011 ms   (0.02%)
        rho = 0.99 -> 0.0119 ms   (0.18%)   <- link effectively DEAD

    So a DYING link adds 0.18% to a hop. Congestion would be invisible,
    routes would never change, and there would be no paper.

    This is not an arithmetic slip. The queue is ~55,000x faster than
    the speed of light over these distances. It cannot compete. M/M/1 is
    the right model for a slow link and the wrong one for a 100 Gbps
    laser spanning 2000 km.
    """
    mu = service_rate_pps(capacity_bps)
    if rho < 0.0: rho = 0.0
    elif rho > MAX_UTILISATION: rho = MAX_UTILISATION
    if rho <= 0.0:
        return 0.0
    return float((1.0 / mu) * (rho / (1.0 - rho)))


def queueing_delay_s(rho):
    """
    How long does a packet actually wait on a link at utilisation rho?

    ===================================================================
    BUFFERBLOAT -- measured reality, not theory.
    ===================================================================

    M/M/1 assumes Poisson arrivals: perfectly smooth, memoryless
    traffic. Real traffic arrives in BURSTS, and real routers carry
    DEEP BUFFERS to absorb them. Under sustained load the buffer fills
    and STAYS full, and everything behind it waits for it to drain.

    Delay is then set by HOW FULL THE BUFFER IS, not by M/M/1.

    THE EVIDENCE (arXiv:2310.09242, a Starlink measurement study):
    RTT inflates ~2-4x under load, reaching almost 400-500 ms, against a
    baseline SpaceX reports as ~33 ms median. Hundreds of milliseconds
    of queueing -- four orders of magnitude above M/M/1's prediction.

    OUR MODEL:
        occupancy = rho ** BUFFER_EXPONENT
        delay     = BUFFER_DELAY_MS * occupancy

    With BUFFER_DELAY_MS = 200, BUFFER_EXPONENT = 2:
        rho = 0.30 ->   18.0 ms   (quiet: comparable to 6.6 ms prop)
        rho = 0.50 ->   50.0 ms
        rho = 0.70 ->   98.0 ms
        rho = 0.90 ->  162.0 ms   (busy: dwarfs propagation)
        rho = 0.99 ->  196.0 ms

    Now congestion is on the same SCALE as geometry, so detouring around
    a busy link can be worth it. That is what makes adaptive routing a
    real problem -- and what the twin is trying to predict.

    LIMITATIONS -- SAY THESE PLAINLY IN THE PAPER:
      - Phenomenological. It reproduces the ORDER OF MAGNITUDE of
        published Starlink load behaviour. It is not fitted to a dataset
        and not derived from queueing theory.
      - BUFFER_DELAY_MS and BUFFER_EXPONENT are SWEPT, not fitted.
      - SpaceX has since right-sized buffers (median 48.5 -> 33 ms,
        p99 >150 -> <65 ms). We model the measured inflation regime;
        operators are actively engineering it away.
    """
    rho = float(np.clip(rho, 0.0, MAX_UTILISATION))
    occupancy = rho ** BUFFER_EXPONENT
    return float(BUFFER_DELAY_MS * occupancy / 1000.0)


def total_link_delay_s(prop_s, rho, kind):
    """
    Full cost of crossing one link, in seconds.
    propagation + transmission + queueing.

    This is what routing minimises.
    """
    cap = link_capacity_bps(kind)
    return prop_s + transmission_delay_s(cap) + queueing_delay_s(rho)


# =====================================================================
# BACKGROUND TRAFFIC
# =====================================================================

class BackgroundTraffic:
    """
    The load everyone else puts on the network.

    Holds one drifting utilisation value per link, updated every tick.

    THIS IS GROUND TRUTH -- what is REALLY happening up there. In
    Brick 3 the digital twin will get a DELAYED, IMPERFECT view of this
    object, and the gap between this and what the twin believes is the
    entire subject of the paper.

    ===================================================================
    WHY THE LOAD DRIFTS INSTEAD OF JUMPING
    ===================================================================
    This is the most important design decision in Brick 2.

    If load jumped to a fresh random value every tick, it would be white
    noise -- unpredictable IN PRINCIPLE. No twin could forecast it,
    however perfect. The paper's question ("what happens when the twin
    is wrong?") would be RIGGED: the twin would be wrong always, by
    construction, and the finding would be trivial and worthless.

    Real load is autocorrelated. A link busy now is probably still busy
    in 5 seconds, and probably different in 5 minutes. THAT is what
    makes prediction meaningful but imperfect -- the regime we study.

    We use an Ornstein-Uhlenbeck process, the standard way to make a
    number wander smoothly around a mean:

        dx = theta*(mean - x)*dt  +  sigma*sqrt(2*theta*dt)*N(0,1)
             [pull back to mean]     [random kick]

    theta = 1/correlation_time controls how fast it forgets where it was.
    """

    def __init__(self, isl_plan, gs_names, rng):
        """
        One load value per PLANNED link.

        We key on the PLANNED wiring, not currently-active links, so a
        link that drops and returns doesn't reset its history. The
        traffic behind it didn't vanish -- it just couldn't get through.
        """
        self.rng = rng
        self.theta = 1.0 / BACKGROUND_CORRELATION_TIME_S
        self.load = {}

        for (a, b, kind) in isl_plan:
            self.load[(min(a, b), max(a, b))] = self._sample_initial()

        # Ground links keyed by STATION, not satellite: the station's
        # users don't disappear when it hands over from one satellite to
        # the next. Load follows the station.
        for name in gs_names:
            self.load[("GS", name)] = self._sample_initial()

    def _sample_initial(self):
        x = self.rng.normal(BACKGROUND_LOAD_MEAN, BACKGROUND_LOAD_SIGMA)
        return float(np.clip(x, 0.0, MAX_UTILISATION))

    def step(self, dt_s=TICK_S):
        """
        Advance every link's load by one tick (Ornstein-Uhlenbeck).

        The sqrt(2*theta) factor normalises the noise so the long-run
        spread of x actually equals BACKGROUND_LOAD_SIGMA. Without it,
        changing the correlation time would silently change the spread
        too, tangling two parameters that should be independent.
        """
        theta = self.theta
        mu = BACKGROUND_LOAD_MEAN
        kick = BACKGROUND_LOAD_SIGMA * np.sqrt(2.0 * theta * dt_s)

        for k in self.load:
            x = self.load[k]
            dx = theta * (mu - x) * dt_s + kick * self.rng.normal()
            self.load[k] = float(np.clip(x + dx, 0.0, MAX_UTILISATION))

    def utilisation(self, u, v, kind):
        """Background utilisation on the link between nodes u and v."""
        if kind == "gsl":
            name = u if isinstance(u, str) else v
            return self.load.get(("GS", name), BACKGROUND_LOAD_MEAN)
        return self.load.get((min(u, v), max(u, v)), BACKGROUND_LOAD_MEAN)

    def snapshot(self):
        """
        Full copy of current load.
        Brick 3 will use this as the telemetry report the twin
        receives -- late.
        """
        return dict(self.load)


# =====================================================================
# APPLYING TRAFFIC TO THE GRAPH
# =====================================================================

def annotate_graph(G, background):
    """
    Write the TRUE delay onto every edge of the real network graph.

    Adds to each edge:
        utilisation -- true load, 0..1
        queue_s     -- queueing delay from that load
        delay_s     -- TOTAL true delay (prop + transmission + queue)

    'delay_s' is the honest cost of crossing that link right now.
    Routing on it gives the genuinely best route.

    NOTE: this is REALITY. The twin does not get to see it. In Brick 3
    the twin builds its own graph from stale telemetry and routes on
    that instead.
    """
    for u, v, d in G.edges(data=True):
        rho = background.utilisation(u, v, d["kind"])
        d["utilisation"] = rho
        d["queue_s"] = queueing_delay_s(rho)
        d["delay_s"] = total_link_delay_s(d["prop_s"], rho, d["kind"])
    return G
