"""
energy.py -- Brick 4: what a route costs, and what CHANGING it costs.

=========================================================================
THE CLAIM THIS FILE SUPPORTS
=========================================================================
"Unnecessary reroutes waste energy."

For that to mean anything, CHANGING a route has to cost something. This
file makes that cost physical and traceable, instead of the old code's
invented ENERGY_PER_SWITCH = 5e-3.

=========================================================================
TWO KINDS OF ENERGY
=========================================================================

1. TRANSMISSION ENERGY -- the cost of carrying the data along a path.
       ~ power x time, summed over hops.
   This is NEARLY CONSTANT no matter which route you pick, because it's
   the same data either way. A route change barely moves it. We include
   it for completeness; it does not drive the result.

2. RECONFIGURATION ENERGY -- the cost of CHANGING the route.
       ~ (number of nodes whose forwarding state changed)
         x (FLOPs to recompute per node)
         x (joules per FLOP, SOURCED: 5e-9)
   This is ZERO if you don't switch and positive if you do. It is the
   cost of an unnecessary reroute, and it is exactly what the gate saves.

=========================================================================
WHY WE COUNT NODES CHANGED, NOT JUST "DID THE PATH CHANGE"
=========================================================================
The old code compared whole paths: any difference counted as one switch,
and changing one hop cost the same as changing the entire route. That's
crude. Reconfiguration cost is per-SATELLITE: only the satellites whose
forwarding entry actually changed have to do work. So we count the
symmetric difference between the old node set and the new one. Swapping
one hop costs a little; rerouting the whole path costs a lot. That's more
honest and it makes the gate's job better-defined.
"""

from config import (
    ENERGY_PER_FLOP_J, RECONFIG_FLOPS_PER_NODE, TRANSMISSION_POWER_W,
)


def reconfig_nodes(old_path, new_path):
    """
    Which nodes have to update their forwarding state when the route
    changes from old_path to new_path?

    A satellite must reconfigure if its role in the path changed -- i.e.
    if it's in one path but not the other, OR its next hop changed. The
    simplest honest proxy is the set of nodes that appear in one path but
    not the other (the symmetric difference), plus nodes whose successor
    changed.

    We include BOTH: nodes added/removed, and nodes whose next-hop moved.
    Ground stations are endpoints, not routers, so we skip string nodes.

    Returns: the set of satellite ids that must reconfigure.
    """
    if old_path is None:
        # First route ever: everyone on the new path sets up. But for a
        # fair steady-state comparison we treat the very first placement
        # as setup, not as a "switch". Callers handle the first tick.
        new_sats = {n for n in (new_path or []) if isinstance(n, int)}
        return new_sats

    if new_path is None:
        return set()

    # next-hop map for each path (who does this node forward to?)
    def next_hops(path):
        return {path[i]: path[i + 1] for i in range(len(path) - 1)}

    old_nh = next_hops(old_path)
    new_nh = next_hops(new_path)

    changed = set()

    # nodes present in one path but not the other
    old_nodes = set(old_path)
    new_nodes = set(new_path)
    changed |= old_nodes.symmetric_difference(new_nodes)

    # nodes present in both but whose next hop changed
    for node in old_nodes & new_nodes:
        if old_nh.get(node) != new_nh.get(node):
            changed.add(node)

    # only satellites reconfigure; ground stations are endpoints
    return {n for n in changed if isinstance(n, int)}


def reconfig_energy_j(old_path, new_path,
                      flops_per_node=None, j_per_flop=None):
    """
    Energy to change the route from old_path to new_path, in joules.

        energy = (nodes that reconfigured) x flops_per_node x j_per_flop

    Zero if the path didn't change. This is the number the gate is trying
    to avoid spending unnecessarily.
    """
    flops_per_node = (RECONFIG_FLOPS_PER_NODE
                      if flops_per_node is None else flops_per_node)
    j_per_flop = ENERGY_PER_FLOP_J if j_per_flop is None else j_per_flop

    n_nodes = len(reconfig_nodes(old_path, new_path))
    return n_nodes * flops_per_node * j_per_flop


def transmission_energy_j(G, path, duration_s, power_w=None):
    """
    Energy to carry the flow along `path` for `duration_s` seconds.

        energy = power x time x number_of_ISL_hops

    Nearly constant across route choices (same data either way), so it
    barely affects the gate comparison. Included for completeness.

    We count ISL hops (satellite-to-satellite); the ground links are the
    endpoints of every path and wash out of any comparison.
    """
    power_w = TRANSMISSION_POWER_W if power_w is None else power_w
    if not path or len(path) < 2:
        return 0.0

    isl_hops = 0
    for a, b in zip(path[:-1], path[1:]):
        if isinstance(a, int) and isinstance(b, int):
            isl_hops += 1

    return power_w * duration_s * isl_hops


def total_energy_j(G, old_path, new_path, duration_s,
                   flops_per_node=None, j_per_flop=None, power_w=None):
    """
    Full energy over one tick: transmission along the (new) path plus any
    reconfiguration cost from having changed to it.

    Returns a dict so callers can see the breakdown -- which matters,
    because the whole point is that reconfiguration is the part that
    varies with switching and transmission is the part that doesn't.
    """
    tx = transmission_energy_j(G, new_path, duration_s, power_w)
    rc = reconfig_energy_j(old_path, new_path, flops_per_node, j_per_flop)
    return {
        "transmission_j": tx,
        "reconfig_j": rc,
        "total_j": tx + rc,
        "reconfig_nodes": len(reconfig_nodes(old_path, new_path)),
    }
