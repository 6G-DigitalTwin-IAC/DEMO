"""
twin.py -- Brick 3: the digital twin.

=========================================================================
THE ONE IDEA IN THIS FILE
=========================================================================
There are two things now, and they are NOT the same:

  REALITY  -- the BackgroundTraffic object in traffic.py. What is truly
              happening on every link, right now, up in orbit.

  THE TWIN -- this file. A model on the ground. It does not see reality.
              It sees telemetry reports that ARRIVED LATE, and between
              reports it holds its last picture.

The twin is always looking at a slightly old photo of the network.

That is the whole paper. Not "we added +-15% random noise" (the old code,
indefensible because you chose the noise). The twin is wrong because
INFORMATION TAKES TIME TO TRAVEL. Light is finite, telemetry queues, the
ground takes time to process. By the time the twin learns a link got busy,
the satellite has moved and the traffic has shifted.

Nobody can argue with the speed of light.

=========================================================================
HOW IT WORKS
=========================================================================
Reality ticks every second. Every TELEMETRY_UPDATE_INTERVAL_S a report is
TAKEN from reality, but it does not reach the twin until TELEMETRY_DELAY_S
later. So the twin always holds a snapshot at least TELEMETRY_DELAY_S old,
ageing to (delay + interval) just before the next report lands.

The twin exposes the SAME interface as BackgroundTraffic -- a
.utilisation(u, v, kind) method -- so it drops straight into
network.annotate_graph() unchanged. The routing code cannot tell whether
it's routing on truth or on belief, exactly as a real router cannot tell
its view is stale.
"""

from collections import deque

from config import (
    TELEMETRY_DELAY_S, TELEMETRY_UPDATE_INTERVAL_S,
    BACKGROUND_LOAD_MEAN,
)


class DigitalTwin:
    """
    A stale, ground-based belief about link loads.

    Read interface matches BackgroundTraffic.utilisation(u, v, kind), so
    annotate_graph() works on it with no changes.

    Internally:
      _in_flight  -- reports sent by reality but not yet arrived, each a
                     (arrival_time, sample_time, snapshot) triple.
      _belief     -- the snapshot of the most recently ARRIVED report.
      _belief_sample_time -- when that snapshot was taken (for exact age).
    """

    def __init__(self, delay_s=None, interval_s=None):
        self.delay_s = TELEMETRY_DELAY_S if delay_s is None else delay_s
        self.interval_s = (TELEMETRY_UPDATE_INTERVAL_S
                           if interval_s is None else interval_s)

        self._in_flight = deque()
        self._belief = None
        self._belief_sample_time = None
        self._last_sample_time = None

    def observe(self, background, t_s):
        """
        Advance the twin one tick. Call once per tick, BEFORE building the
        twin's graph.

        1. SEND: if a new report is due, snapshot reality now and stamp it
           to arrive at t + delay. (Telemetry leaving the satellite.)
        2. RECEIVE: any in-flight report whose arrival time has passed
           becomes the current belief. (Report reaching the ground.)
        """
        # 1. SEND if due.
        if (self._last_sample_time is None
                or t_s - self._last_sample_time >= self.interval_s - 1e-9):
            self._in_flight.append((t_s + self.delay_s, t_s,
                                    background.snapshot()))
            self._last_sample_time = t_s

        # 2. RECEIVE all reports that have arrived. Reports are in send
        #    order == arrival order, so the last to pop is the freshest.
        while self._in_flight and self._in_flight[0][0] <= t_s + 1e-9:
            _, sample_time, snapshot = self._in_flight.popleft()
            self._belief = snapshot
            self._belief_sample_time = sample_time

    def utilisation(self, u, v, kind):
        """
        What the twin BELIEVES the load is on link (u, v).

        Drop-in replacement for BackgroundTraffic.utilisation.
        Before the first report arrives, the twin has no information and
        returns the mean -- the least-assuming guess.
        """
        if self._belief is None:
            return BACKGROUND_LOAD_MEAN

        if kind == "gsl":
            name = u if isinstance(u, str) else v
            return self._belief.get(("GS", name), BACKGROUND_LOAD_MEAN)
        return self._belief.get((min(u, v), max(u, v)), BACKGROUND_LOAD_MEAN)

    def belief_age_s(self, t_s):
        """
        How stale is the current belief, in seconds? = now - sample_time.

        This is THE number the paper cares about. It's at least delay_s
        (the report was already delay_s old on arrival) and grows to
        delay_s + interval_s just before the next report lands.
        Infinite before the first report arrives.
        """
        if self._belief_sample_time is None:
            return float("inf")
        return t_s - self._belief_sample_time

    def has_belief(self):
        """True once the first report has arrived."""
        return self._belief is not None
