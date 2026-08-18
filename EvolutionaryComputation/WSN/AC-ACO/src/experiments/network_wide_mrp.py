"""Experimental network-wide Pure-MRP adaptation for fair workload comparison."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from ac_aco_mrp.multiflow import build_hybrid_multiflow_plan
from ac_aco_mrp.multiflow_energy import evaluate_hybrid_multiflow_energy
from ac_aco_mrp.types import HybridMultiFlowPlan
from mrp.multi_ch_routing import MRPRoutingFailure, route_cluster_heads_with_mrp
from mrp.network_wide_clustering import (
    NetworkWideClusterFormationResult,
    select_network_wide_cluster_heads,
)
from mrp.phase2.routing import discover_mrp_phase2_routes
from mrp.phase2.control_energy import validate_ant_energy_parameters
from mrp.phase3.path_selection import select_phase3_route
from mrp.runner_lifecycle import (
    apply_baseline_lifecycle, combine_data_and_control_energy,
    commit_control_energy_after_failure, validate_pure_mrp_state,
)
from mrp.runner_types import (
    MRPCHRoutingResult, PheromoneLifecycle, PureMRPParameters, PureMRPState,
)
from mrp.topology import assign_members_to_cluster_heads


@dataclass(frozen=True)
class NetworkWideMRPRoundResult:
    """One all-live-network MRP clustering/routing/energy result."""

    round_index: int
    phase1_result: NetworkWideClusterFormationResult | None
    members_by_head: Mapping[int, tuple[int, ...]] | None
    ch_routing_results: tuple[MRPCHRoutingResult, ...]
    selected_routes: Mapping[int, tuple[int, ...]] | None
    final_plan: HybridMultiFlowPlan | None
    e_m_list: tuple[float, ...] | None
    e_sum: float | None
    state_before: PureMRPState
    state_after: PureMRPState
    newly_dead: tuple[int, ...]
    success: bool
    failure_stage: str | None
    failure_reason: str | None
    failed_cluster_head: int | None = None
    routing_failure_classification: str | None = None
    final_topology: object | None = None
    failed_phase2_result: object | None = None
    routing_status_by_ch: Mapping[int, str] | None = None
    data_energy: float | None = None
    ant_control_energy: object | None = None


def run_network_wide_mrp_round(
    nodes: Sequence[object], dist_matrix, base_dists, state: PureMRPState,
    hop_counts: Mapping[int, int], parameters: PureMRPParameters, rng: object,
    pheromone_lifecycle: PheromoneLifecycle, target_cluster_head_count: int,
    round_index: int,
) -> NetworkWideMRPRoundResult:
    """Select N MRP CHs, route all N with the common engine, and commit once."""

    _require_contract(nodes, dist_matrix, state, parameters, pheromone_lifecycle)
    phase1 = None
    members = None
    routed = ()
    selected_routes = None
    try:
        phase1 = select_network_wide_cluster_heads(
            nodes, state.live_nodes, state.residual_e, dist_matrix,
            parameters.communication_radius, parameters.config,
            target_cluster_head_count,
        )
    except Exception as error:
        return _failure(round_index, state, "mrp_clustering", error)
    members_raw = assign_members_to_cluster_heads(
        nodes, phase1.selected_cluster_heads, parameters.communication_radius,
        dist_matrix, state.residual_e,
    )
    if members_raw is None:
        return _failure(
            round_index, state, "member_assignment",
            "MRP_CLUSTERING_EXHAUSTED: nearest-CH assignment did not cover all live sensors",
            phase1,
        )
    members = {head: tuple(values) for head, values in members_raw.items()}
    try:
        routing = route_cluster_heads_with_mrp(
            phase1.selected_cluster_heads, state.live_nodes, state, dist_matrix,
            base_dists, hop_counts, parameters, rng, pheromone_lifecycle,
            phase2_discovery=discover_mrp_phase2_routes,
            phase3_selector=select_phase3_route,
        )
    except MRPRoutingFailure as error:
        partial = {item.cluster_head: item.phase3_result.selected_route for item in error.partial_results}
        return _failure(
            round_index, state,
            "mrp_phase2" if error.stage == "phase2" else "mrp_routing",
            error, phase1, members, error.partial_results, partial,
            failed_cluster_head=error.cluster_head,
            routing_failure_classification=error.classification,
            failed_phase2_result=error.phase2_result,
            control_energy=error.control_energy,
        )
    routed = routing.ch_routing_results
    selected_routes = dict(routing.selected_routes)
    try:
        plan = build_hybrid_multiflow_plan(
            phase1.selected_cluster_heads, members, selected_routes,
            state.live_nodes, len(nodes),
        )
    except Exception as error:
        return _failure(
            round_index, state, "multiflow_plan", error, phase1, members,
            routed, selected_routes,
            control_energy=routing.control_energy,
        )
    try:
        energy = evaluate_hybrid_multiflow_energy(plan, dist_matrix, base_dists, parameters)
        combined_e_m, combined_sum = combine_data_and_control_energy(
            energy.e_m_list, energy.e_sum, routing.control_energy,
        )
        persisted = (
            routing.pheromone_after
            if pheromone_lifecycle is PheromoneLifecycle.PERSIST_ACROSS_ROUNDS else None
        )
        state_after, newly_dead = apply_baseline_lifecycle(
            state, combined_e_m, combined_sum, persisted,
        )
    except Exception as error:
        return _failure(
            round_index, state, "final_energy", error, phase1, members,
            routed, selected_routes, plan,
            control_energy=routing.control_energy,
        )
    return NetworkWideMRPRoundResult(
        round_index, phase1, members, routed, selected_routes, plan,
        combined_e_m, combined_sum, state, state_after, newly_dead,
        True, None, None,
        routing_status_by_ch=MappingProxyType({
            head: "SUCCESS" for head in phase1.selected_cluster_heads
        }),
        data_energy=energy.e_sum, ant_control_energy=routing.control_energy,
    )


def _failure(
    round_index, state, stage, reason, phase1=None, members=None, routed=(),
    selected_routes=None, plan=None, failed_cluster_head=None,
    routing_failure_classification=None, failed_phase2_result=None,
    control_energy=None,
):
    statuses = _routing_statuses(
        phase1, routed, failed_cluster_head, routing_failure_classification,
    )
    state_after, newly_dead = commit_control_energy_after_failure(state, control_energy)
    charged = control_energy is not None and not control_energy.is_zero
    return NetworkWideMRPRoundResult(
        round_index, phase1, members, tuple(routed), selected_routes, plan,
        control_energy.e_m_list if charged else None,
        control_energy.total_energy if charged else None,
        state, state_after, newly_dead, False, stage, str(reason),
        failed_cluster_head, routing_failure_classification,
        None, failed_phase2_result, statuses,
        None, control_energy,
    )


def _routing_statuses(phase1, routed, failed_cluster_head, failure_status):
    if phase1 is None:
        return None
    completed = {item.cluster_head for item in routed}
    if failed_cluster_head is None:
        return MappingProxyType({head: "SUCCESS" for head in completed})
    statuses = {}
    for head in phase1.selected_cluster_heads:
        if head in completed:
            statuses[head] = "SUCCESS"
        elif head == failed_cluster_head:
            statuses[head] = failure_status or "MRP_PHASE3_FAILURE"
        else:
            statuses[head] = "NOT_ATTEMPTED_AFTER_REQUIRED_FAILURE"
    return MappingProxyType(statuses)


def _require_contract(nodes, dist_matrix, state, parameters, lifecycle):
    if not isinstance(state, PureMRPState) or not isinstance(parameters, PureMRPParameters):
        raise ValueError("network-wide MRP requires complete physical state and parameters")
    if not isinstance(lifecycle, PheromoneLifecycle) or len(nodes) != len(state.residual_e):
        raise ValueError("network-wide MRP lifecycle/state dimensions are invalid")
    validate_ant_energy_parameters(
        parameters.charge_ant_energy, parameters.ant_control_packet_bits,
    )
    validate_pure_mrp_state(
        state, len(nodes), dist_matrix, parameters.communication_radius,
        parameters.config,
    )
