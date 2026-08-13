"""GENERIC_COMPATIBILITY validation for the existing ``Node(-1)`` contract."""

from collections.abc import Collection, Iterable, Mapping, Sequence

from node import Node


class TopologyValidationError(ValueError):
    """Raised when an already-decided routing structure cannot form one tree."""


def validate_input_contract(
    heads: tuple[int, ...],
    members_by_head: Mapping[int, Sequence[int]],
    selected_routes: Mapping[int, Sequence[int]],
    live_nodes: Collection[int],
    node_count: int,
) -> None:
    """GENERIC_COMPATIBILITY: validate decisions before topology conversion."""

    live_set = set(live_nodes)
    if not heads or len(set(heads)) != len(heads):
        raise TopologyValidationError("cluster heads must be a non-empty unique collection")
    if set(members_by_head) != set(heads) or set(selected_routes) != set(heads):
        raise TopologyValidationError("members and routes must be supplied for every CH")

    members_seen: set[int] = set()
    for head in heads:
        if not is_valid_sensor_id(head, node_count) or head not in live_set:
            raise TopologyValidationError(f"invalid or dead CH: {head}")
        route = tuple(selected_routes[head])
        if len(route) < 2 or route[0] != head or route[-1] != -1:
            raise TopologyValidationError(f"route for CH {head} must start at CH and end at sink")
        route_sensors = route[:-1]
        if len(set(route_sensors)) != len(route_sensors):
            raise TopologyValidationError(f"cycle in route for CH {head}")
        if any(
            not is_valid_sensor_id(sensor_id, node_count) or sensor_id not in live_set
            for sensor_id in route_sensors
        ):
            raise TopologyValidationError(f"route for CH {head} contains invalid/dead sensor")
        for member_id in members_by_head[head]:
            if (
                not is_valid_sensor_id(member_id, node_count)
                or member_id not in live_set
                or member_id in heads
                or member_id in members_seen
            ):
                raise TopologyValidationError(f"invalid member assignment for sensor {member_id}")
            members_seen.add(member_id)


def validate_topology(
    root: Node,
    *,
    node_count: int,
    live_nodes: Collection[int],
    cluster_heads: Iterable[int],
    members_by_head: Mapping[int, Sequence[int]],
    selected_routes: Mapping[int, Sequence[int]],
) -> None:
    """GENERIC_COMPATIBILITY: enforce the existing simulator tree contract."""

    heads = tuple(cluster_heads)
    live_set = set(live_nodes)
    if root.idx != -1 or root.p_idx != -1:
        raise TopologyValidationError("topology root must be Node(idx=-1, p_idx=-1)")

    expected_members = {
        member for members in members_by_head.values() for member in members
    }
    expected_route_nodes = {
        sensor_id
        for route in selected_routes.values()
        for sensor_id in route
        if sensor_id != -1
    }
    expected_nodes = expected_members | expected_route_nodes | set(heads)
    parent_by_id: dict[int, int] = {}
    seen_ids: set[int] = set()
    seen_objects: set[int] = set()
    stack: list[tuple[Node, int | None]] = [(root, None)]

    while stack:
        node, expected_parent = stack.pop()
        if id(node) in seen_objects:
            raise TopologyValidationError("cycle or duplicate Node object detected")
        seen_objects.add(id(node))

        if expected_parent is not None:
            if not is_valid_sensor_id(node.idx, node_count):
                raise TopologyValidationError(f"invalid sensor ID: {node.idx}")
            if node.idx not in live_set:
                raise TopologyValidationError(f"dead sensor included: {node.idx}")
            if node.idx in seen_ids:
                raise TopologyValidationError(f"duplicate physical sensor: {node.idx}")
            if node.p_idx != expected_parent:
                raise TopologyValidationError(f"inconsistent p_idx for sensor {node.idx}")
            seen_ids.add(node.idx)
            parent_by_id[node.idx] = expected_parent

        for branch in node.branches:
            stack.append((branch, node.idx))

    if seen_ids != expected_nodes:
        raise TopologyValidationError("topology does not contain exactly the intended sensors")

    for head, members in members_by_head.items():
        for member_id in members:
            if member_id not in expected_route_nodes and parent_by_id.get(member_id) != head:
                raise TopologyValidationError(
                    f"member {member_id} is not attached to assigned CH {head}"
                )

    for head in heads:
        route = selected_routes[head]
        for child_id, parent_id in zip(route, route[1:]):
            if parent_by_id.get(child_id) != parent_id:
                raise TopologyValidationError(
                    f"route edge {child_id}->{parent_id} is not materialized"
                )

    _validate_node_roles(root, expected_route_nodes)


def is_valid_sensor_id(sensor_id: int, node_count: int) -> bool:
    """GENERIC_COMPATIBILITY: check an ID without deriving new geometry/state."""

    return isinstance(sensor_id, int) and 0 <= sensor_id < node_count


def _validate_node_roles(root: Node, forwarding_nodes: set[int]) -> None:
    stack = [root]
    while stack:
        node = stack.pop()
        for branch in node.branches:
            if branch.idx in forwarding_nodes and not branch.isCH:
                raise TopologyValidationError(
                    f"forwarding route node not marked isCH: {branch.idx}"
                )
            if branch.idx not in forwarding_nodes and branch.isCH:
                raise TopologyValidationError(
                    f"ordinary member marked isCH: {branch.idx}"
                )
            stack.append(branch)
