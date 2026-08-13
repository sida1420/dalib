"""GENERIC_COMPATIBILITY adapters for the existing ``Node(-1)`` boundary."""

from collections.abc import Collection, Iterable, Mapping, Sequence

import kdtree
from node import Node

from .topology_validation import (
    TopologyValidationError,
    is_valid_sensor_id,
    validate_input_contract,
    validate_topology,
)


def assign_members_to_cluster_heads(
    nodes: Sequence[object],
    cluster_heads: Iterable[int],
    radius: float,
    dist_matrix: Sequence[Sequence[float]],
    residual_e: Sequence[float],
) -> dict[int, list[int]] | None:
    """GENERIC_COMPATIBILITY: mirror only ``evaluate.network_config`` lines 5–23.

    The helper uses the existing KDTree and caller-provided distance matrix to
    assign every live non-CH sensor to its nearest live CH.  It intentionally
    performs no CH-to-sink routing, energy accounting, or state mutation.
    """

    heads = tuple(cluster_heads)
    node_count = len(nodes)
    if (
        not heads
        or len(residual_e) != node_count
        or any(not is_valid_sensor_id(head, node_count) for head in heads)
        or len(set(heads)) != len(heads)
        or any(residual_e[head] <= 0 for head in heads)
    ):
        return None

    if len(dist_matrix) != node_count or any(len(row) != node_count for row in dist_matrix):
        return None

    tree = kdtree.KDTree(
        True,
        sorted(heads, key=lambda idx: nodes[idx].x),
        sorted(heads, key=lambda idx: nodes[idx].y),
    )
    assignments = {head: [] for head in heads}
    head_lookup = set(heads)

    for sensor_id in range(node_count):
        if sensor_id in head_lookup or residual_e[sensor_id] <= 0:
            continue
        nearest_head, _ = tree.nearest(nodes, sensor_id, None, 1e9)
        if dist_matrix[nearest_head][sensor_id] > radius:
            return None
        assignments[nearest_head].append(sensor_id)

    return assignments


def materialize_topology(
    cluster_heads: Iterable[int],
    members_by_head: Mapping[int, Sequence[int]],
    selected_routes: Mapping[int, Sequence[int]],
    live_nodes: Collection[int],
    node_count: int,
) -> Node | None:
    """GENERIC_COMPATIBILITY: convert decided source-to-sink paths to ``Node``.

    Each route is ``[cluster_head, ..., -1]``.  A route decision is never
    selected or scored here.  Invalid input, including a relay with competing
    parents, returns ``None`` instead of duplicating a physical sensor.
    """

    try:
        heads = tuple(cluster_heads)
        validate_input_contract(
            heads, members_by_head, selected_routes, live_nodes, node_count
        )
        root = Node(-1)
        topology_nodes: dict[int, Node] = {}
        parents: dict[int, int] = {}

        def topology_node(sensor_id: int) -> Node:
            if sensor_id not in topology_nodes:
                # ``isCH`` is the existing evaluator's forwarding-data flag.
                # It is required for every selected-route node, including a
                # physical cluster member that relays another CH's traffic.
                topology_nodes[sensor_id] = Node(sensor_id, isCH=True)
            return topology_nodes[sensor_id]

        for head in heads:
            route = selected_routes[head]
            for child_id, parent_id in zip(route, route[1:]):
                child = topology_node(child_id)
                parent = root if parent_id == -1 else topology_node(parent_id)
                _attach(child, parent, parents)

        for head, members in members_by_head.items():
            parent = topology_node(head)
            for member_id in members:
                if member_id in topology_nodes:
                    # A member may also be a selected route relay.  Its sole
                    # physical parent is already dictated by that route; the
                    # membership mapping remains logical metadata.
                    continue
                member = Node(member_id, isCH=False)
                _attach(member, parent, parents)

        validate_topology(
            root,
            node_count=node_count,
            live_nodes=live_nodes,
            cluster_heads=heads,
            members_by_head=members_by_head,
            selected_routes=selected_routes,
        )
        return root
    except TopologyValidationError:
        return None


def _attach(child: Node, parent: Node, parents: dict[int, int]) -> None:
    existing_parent = parents.get(child.idx)
    if existing_parent is not None:
        if existing_parent != parent.idx:
            raise TopologyValidationError(
                f"sensor {child.idx} has conflicting parents {existing_parent} and {parent.idx}"
            )
        return
    parents[child.idx] = parent.idx
    child.set_parent(parent.idx)
    parent.branches.append(child)
