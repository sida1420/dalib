"""Final physical-energy ledger for Hybrid multi-flow routing plans."""

from collections import defaultdict
import math
from types import MappingProxyType

from evaluate import E_data_receiving, E_transmitting
from mrp.runner_types import PureMRPParameters

from .multiflow import HybridMultiFlowError
from .types import HybridMultiFlowEnergy, HybridMultiFlowPlan


def evaluate_hybrid_multiflow_energy(
    plan: HybridMultiFlowPlan,
    dist_matrix,
    base_dists,
    parameters: PureMRPParameters,
) -> HybridMultiFlowEnergy:
    """Charge final selected flows once, using the existing radio primitives.

    A route-node's existing one-payload contribution is assigned once to its
    own selected-CH flow, or (for a relay-only sensor) to its first Phase-III
    flow.  This is identical to the legacy tree whenever every sensor has one
    next hop, while allowing a relay to forward independent flows to different
    next hops without duplicating its physical residual-energy state.
    """

    node_count = len(dist_matrix)
    if not isinstance(plan, HybridMultiFlowPlan) or len(base_dists) != node_count:
        raise HybridMultiFlowError("multi-flow plan and physical geometry are required")
    if any(len(row) != node_count for row in dist_matrix):
        raise HybridMultiFlowError("distance matrix must be square")
    edge_bits = _route_edge_payloads(plan, parameters.bit_count)
    route_nodes = {sensor for flow in plan.selected_flows for sensor in flow.route[:-1]}
    for head, members in plan.member_assignments.items():
        for member in members:
            if member not in route_nodes:
                edge_bits[(member, head)] += parameters.bit_count

    e_m = [0.0] * node_count
    for (sender, receiver), bits in edge_bits.items():
        distance = base_dists[sender] if receiver == -1 else dist_matrix[sender][receiver]
        if not math.isfinite(distance) or distance < 0 or distance > parameters.communication_radius:
            raise HybridMultiFlowError(f"selected flow link {sender}->{receiver} is outside communication radius")
        e_m[sender] += E_transmitting(
            parameters.e_elec, parameters.free_space_coeff, parameters.multipath_coeff,
            distance, parameters.d0, bits + parameters.ctrl_bit,
        )
        if receiver != -1:
            e_m[receiver] += E_data_receiving(parameters.e_elec, bits + parameters.ctrl_bit)
    if any(not math.isfinite(value) or value < 0 for value in e_m):
        raise HybridMultiFlowError("multi-flow energy must be finite and non-negative")
    return HybridMultiFlowEnergy(tuple(e_m), sum(e_m), MappingProxyType(dict(edge_bits)))


def _route_edge_payloads(plan, bit_count):
    flows = {flow.source_ch: flow.route for flow in plan.selected_flows}
    edge_bits = defaultdict(float)
    for sensor, owner in plan.local_payload_flow_by_sensor.items():
        route = flows.get(owner)
        if route is None or sensor not in route[:-1]:
            raise HybridMultiFlowError("local route-node payload has no selected owner flow")
        start = route.index(sensor)
        for child, parent in zip(route[start:], route[start + 1:]):
            edge_bits[(child, parent)] += bit_count
    return edge_bits
