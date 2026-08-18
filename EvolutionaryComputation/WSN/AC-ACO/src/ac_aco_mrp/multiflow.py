"""Hybrid-only representation for independently selected CH routing flows."""

from collections.abc import Collection, Iterable, Mapping, Sequence
from types import MappingProxyType

from mrp.topology_validation import is_valid_sensor_id

from .types import HybridMultiFlowPlan, HybridSelectedFlow


class HybridMultiFlowError(ValueError):
    """Raised when independently selected Hybrid flows are physically invalid."""


def build_hybrid_multiflow_plan(
    cluster_heads: Iterable[int],
    members_by_head: Mapping[int, Sequence[int]],
    selected_routes: Mapping[int, Sequence[int]],
    live_nodes: Collection[int],
    node_count: int,
) -> HybridMultiFlowPlan:
    """Retain every selected CH route without imposing a cross-flow parent map."""

    heads = tuple(cluster_heads)
    live = set(live_nodes)
    if not heads or len(set(heads)) != len(heads):
        raise HybridMultiFlowError("cluster heads must be a non-empty unique collection")
    if set(members_by_head) != set(heads) or set(selected_routes) != set(heads):
        raise HybridMultiFlowError("members and routes must be supplied for every CH")
    if any(not is_valid_sensor_id(head, node_count) or head not in live for head in heads):
        raise HybridMultiFlowError("cluster heads must be live valid sensors")

    members = {head: tuple(members_by_head[head]) for head in heads}
    _validate_member_assignments(heads, members, live, node_count)
    flows = tuple(HybridSelectedFlow(head, tuple(selected_routes[head])) for head in heads)
    for flow in flows:
        _validate_flow(flow, live, node_count)
    if {flow.source_ch for flow in flows} != set(heads):
        raise HybridMultiFlowError("every winning CH requires exactly one selected flow")

    route_nodes = {sensor for flow in flows for sensor in flow.route[:-1]}
    physical = tuple(sorted(route_nodes | {member for values in members.values() for member in values}))
    owners = {head: head for head in heads}
    for flow in flows:
        for sensor in flow.route[:-1]:
            owners.setdefault(sensor, flow.source_ch)
    return HybridMultiFlowPlan(
        flows,
        MappingProxyType(members),
        physical,
        MappingProxyType(owners),
    )


def _validate_member_assignments(heads, members, live, node_count):
    assigned = [member for values in members.values() for member in values]
    if len(assigned) != len(set(assigned)):
        raise HybridMultiFlowError("member assignments must be unique")
    if any(
        not is_valid_sensor_id(member, node_count) or member not in live or member in heads
        for member in assigned
    ):
        raise HybridMultiFlowError("member assignments must contain live non-CH sensors")
    if set(assigned) != live - set(heads):
        raise HybridMultiFlowError("every live non-CH sensor requires one member assignment")


def _validate_flow(flow, live, node_count):
    route = flow.route
    if len(route) < 2 or route[0] != flow.source_ch or route[-1] != -1:
        raise HybridMultiFlowError(f"route for CH {flow.source_ch} must start at CH and end at sink")
    sensors = route[:-1]
    if len(set(sensors)) != len(sensors):
        raise HybridMultiFlowError(f"loop within route for CH {flow.source_ch}")
    if any(not is_valid_sensor_id(sensor, node_count) or sensor not in live for sensor in sensors):
        raise HybridMultiFlowError(f"route for CH {flow.source_ch} contains invalid/dead sensor")
