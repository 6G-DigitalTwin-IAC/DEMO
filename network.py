"""
network.py -- turn physical state + traffic into a routable graph.

This is THE REAL NETWORK. Ground truth. What is actually happening.

Later, the digital twin will hold its OWN separate graph -- built from
telemetry that arrived late -- and the two will disagree. That
disagreement is the whole point of the paper. But that's Brick 3.
For now there is only reality.

EDGE ATTRIBUTES:
    kind        -- "intra" | "inter" | "gsl"
    dist_km     -- physical distance
    prop_s      -- propagation delay = dist / c        [Brick 1]
    utilisation -- true load, 0..1                     [Brick 2]
    queue_s     -- queueing delay from that load       [Brick 2]
    delay_s     -- TOTAL: prop + transmission + queue  [Brick 2]

Route on 'delay_s' for the true best path.
Route on 'prop_s' for the geometry-only path (Brick 1 behaviour).
"""

import networkx as nx

import isl as isl_mod
import ground as ground_mod


def build_graph(sats, gs_list, isl_plan):
    """
    Build the current network graph from live physical state.

    NODES:
        satellites      -> integer ids (0 .. N_SATS-1)
        ground stations -> string names ("IST", "NYC")

    Mixing ints and strings is deliberate: you can never confuse a
    satellite with a ground station, and paths read clearly:
        ['IST', 47, 48, 70, 'NYC']
    """
    G = nx.Graph()

    for sat in sats:
        G.add_node(sat["id"], kind="sat", plane=sat["plane"], slot=sat["slot"])
    for gs in gs_list:
        G.add_node(gs["name"], kind="gs")

    for (a, b, kind, d_km, prop_s) in isl_mod.active_links(sats, isl_plan):
        G.add_edge(a, b, kind=kind, dist_km=d_km, prop_s=prop_s)

    for (name, sid, d_km, prop_s, el) in ground_mod.active_gsls(gs_list, sats):
        G.add_edge(name, sid, kind="gsl", dist_km=d_km, prop_s=prop_s,
                   elevation_deg=el)

    return G


def shortest_path(G, src, dst, weight="delay_s"):
    """
    Best path by the given weight.

    weight="delay_s" -> true best route (propagation + congestion)
    weight="prop_s"  -> geometry only, ignoring traffic (Brick 1)
    """
    try:
        return nx.shortest_path(G, src, dst, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def path_cost(G, path, weight="delay_s"):
    """Total cost along a path. inf if the path is broken."""
    if not path or len(path) < 2:
        return float("inf")
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        if not G.has_edge(a, b):
            return float("inf")
        total += G[a][b][weight]
    return total


def path_delay_s(G, path):
    """Total true delay (propagation + transmission + queueing)."""
    return path_cost(G, path, "delay_s")


def path_prop_s(G, path):
    """Propagation delay only -- the physical floor for this path."""
    return path_cost(G, path, "prop_s")


def path_distance_km(G, path):
    """Total physical distance along a path."""
    return path_cost(G, path, "dist_km")


def path_max_utilisation(G, path):
    """
    Busiest link on the path. A path is only as good as its worst hop.
    """
    if not path or len(path) < 2:
        return 0.0
    return max(G[a][b].get("utilisation", 0.0)
               for a, b in zip(path[:-1], path[1:]))


def graph_summary(G):
    kinds = {}
    for _, _, d in G.edges(data=True):
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "edges_by_kind": kinds,
    }
