"""
gate.py -- Brick 5: the control gate. The contribution.

=========================================================================
THE IDEA
=========================================================================
The twin, working from stale belief, keeps proposing route switches.
Brick 3 showed ~46% of them (at 10s delay) lead to a WORSE route. Obeying
every proposal makes the network thrash, and every switch burns
reconfiguration energy (Brick 4).

The gate sits between the twin's proposal and the actual routing decision
and asks one question:

    "Is this switch worth it?"

    Allow the switch only if the twin PREDICTS the new route is enough
    better than the current one:

        new_pred_delay  <  threshold * current_pred_delay

Both numbers are the TWIN'S predictions. The gate is as blind as the twin
-- it cannot see the true delay. That's the honest constraint, and it's
what makes this realistic: a real controller only has the twin to go on.

=========================================================================
THREE POLICIES, RUN SIDE BY SIDE
=========================================================================
To show what the gate does, we run three controllers on the SAME reality:

  1. DIRECT      -- always takes the true best route. An ORACLE. It sees
                    real congestion with no delay. Physically impossible
                    (no controller has instant perfect global knowledge),
                    so it's a CEILING, not a baseline: the best any method
                    could do. Its switch count is the "natural" churn.

  2. TWIN_NOGATE -- obeys the twin every tick. Switches whenever the twin's
                    best route differs from the current one. This is what
                    prior digital-twin routing does: trust the twin. It
                    thrashes, because the twin is often wrong.

  3. TWIN_GATE   -- the twin proposes, the gate filters. OUR METHOD.
                    Should get close to DIRECT's delay while making far
                    fewer switches than TWIN_NOGATE.

Each policy holds its OWN current route and only changes it according to
its own rule. They're compared on the SAME congestion and geometry every
tick, so differences are the policy's doing, not luck.
"""

import network
from config import GATE_THRESHOLD


class Policy:
    """
    One routing controller. Holds the route it's currently using and
    decides, each tick, whether to change it.

    Tracks everything the paper needs: switch count, per-tick true delay,
    and the sequence of routes (so energy.py can price the switches).
    """

    def __init__(self, name):
        self.name = name
        self.current_path = None
        self.prev_path = None          # route before the last decision
        self.switches = 0
        self.switched_this_tick = False

    def _adopt(self, new_path):
        """Move to new_path, recording whether it was a real change."""
        self.prev_path = self.current_path
        if self.current_path is not None and new_path != self.current_path:
            self.switches += 1
            self.switched_this_tick = True
        else:
            self.switched_this_tick = False
        self.current_path = new_path


class DirectPolicy(Policy):
    """
    ORACLE. Always routes on the true graph. The ceiling.
    """
    def __init__(self):
        super().__init__("direct")

    def decide(self, G_real, G_twin, src, dst):
        best = network.shortest_path(G_real, src, dst, "delay_s")
        if best is not None:
            self._adopt(best)
        else:
            self.switched_this_tick = False
        return self.current_path


class TwinNoGatePolicy(Policy):
    """
    Obeys the twin every tick. What prior work does: trust the twin.
    """
    def __init__(self):
        super().__init__("twin_nogate")

    def decide(self, G_real, G_twin, src, dst):
        best = network.shortest_path(G_twin, src, dst, "delay_s")
        if best is not None:
            self._adopt(best)
        else:
            self.switched_this_tick = False
        return self.current_path


class TwinGatePolicy(Policy):
    """
    OUR METHOD. The twin proposes; the gate filters.

    Allow the switch only if the twin predicts the new route is enough
    better than the current one:

        twin_pred(new)  <  threshold * twin_pred(current)

    Everything is evaluated on the TWIN'S graph, because the gate only
    has the twin's view -- it cannot see the truth.
    """
    def __init__(self, threshold=None):
        super().__init__("twin_gate")
        self.threshold = GATE_THRESHOLD if threshold is None else threshold

    def decide(self, G_real, G_twin, src, dst):
        proposal = network.shortest_path(G_twin, src, dst, "delay_s")

        # No current route yet, or lost connectivity: take what we can.
        if self.current_path is None:
            if proposal is not None:
                self._adopt(proposal)
            else:
                self.switched_this_tick = False
            return self.current_path

        if proposal is None:
            self.switched_this_tick = False
            return self.current_path

        # If the twin's proposal IS the current route, nothing to do.
        if proposal == self.current_path:
            self.switched_this_tick = False
            self.prev_path = self.current_path
            return self.current_path

        # THE GATE. Compare twin-predicted delays.
        # Is the current route still valid on the twin graph? If a link on
        # it has gone, its cost is inf and any proposal beats it.
        current_pred = network.path_delay_s(G_twin, self.current_path)
        proposal_pred = network.path_delay_s(G_twin, proposal)

        if proposal_pred < self.threshold * current_pred:
            self._adopt(proposal)          # worth it: switch
        else:
            self.switched_this_tick = False  # not worth it: hold
            self.prev_path = self.current_path

        return self.current_path
