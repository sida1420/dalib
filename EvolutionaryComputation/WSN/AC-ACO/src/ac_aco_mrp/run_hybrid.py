"""Import-safe Hybrid runner: AC-ACO CH choice, then MRP-only final routing."""

from collections.abc import Sequence
from functools import partial

from mrp.phase2.control_energy import validate_ant_energy_parameters
from mrp.phase2.routing import discover_mrp_phase2_routes
from mrp.phase3.path_selection import select_phase3_route
from mrp.multi_ch_routing import MRPRoutingFailure, route_cluster_heads_with_mrp
from mrp.runner_lifecycle import (
    apply_baseline_lifecycle, combine_data_and_control_energy,
    commit_control_energy_after_failure,
)
from mrp.runner_types import PheromoneLifecycle
from mrp.topology import assign_members_to_cluster_heads

from .multiflow import build_hybrid_multiflow_plan
from .multiflow_energy import evaluate_hybrid_multiflow_energy

from .phase1 import select_ac_aco_cluster_heads
from .types import (
    HybridParameters,
    HybridRoundContext,
    HybridRoundResult,
    HybridRunResult,
    HybridState,
)


def run_hybrid_round(
    nodes: Sequence[object], dist_matrix: Sequence[Sequence[float]], base_dists: Sequence[float],
    state: HybridState, context: HybridRoundContext, parameters: HybridParameters,
    ac_aco_rng: object, mrp_rng: object, pheromone_lifecycle: PheromoneLifecycle,
    round_index: int,
) -> HybridRoundResult:
    """Run one atomic Hybrid round; legacy Greedy never becomes final routing."""

    _require_contract(nodes, state, context, parameters, pheromone_lifecycle)
    if ac_aco_rng is mrp_rng:
        raise ValueError("Hybrid requires separate caller-owned AC-ACO and MRP RNG streams")
    phase1 = None
    members = None
    routed = []
    selected_routes: dict[int, tuple[int, ...]] = {}
    pheromone_before = pheromone_after = None
    control_energy = None
    try:
        phase1 = select_ac_aco_cluster_heads(
            nodes, dist_matrix, base_dists, state.physical_state.residual_e,
            state.physical_state.live_nodes, state.ac_aco_state, parameters.ac_aco,
            parameters.mrp, ac_aco_rng, round_index,
        )
    except Exception as error:
        return _failure(round_index, state, "ac_aco_phase1", error)
    members = assign_members_to_cluster_heads(
        nodes, phase1.selected_cluster_heads, parameters.mrp.communication_radius,
        dist_matrix, state.physical_state.residual_e,
    )
    if members is None:
        return _failure(round_index, state, "member_assignment", "nearest-CH assignment failed", phase1)
    members = {head: tuple(member_ids) for head, member_ids in members.items()}
    try:
        phase3_selector = (
            select_phase3_route if not parameters.phase3_top_k
            else partial(select_phase3_route, top_k=parameters.phase3_top_k)
        )
        routing = route_cluster_heads_with_mrp(
            phase1.selected_cluster_heads, state.physical_state.live_nodes,
            state.physical_state, dist_matrix, base_dists, context.hop_counts,
            parameters.mrp, mrp_rng, pheromone_lifecycle,
            phase2_discovery=discover_mrp_phase2_routes,
            phase3_selector=phase3_selector,
        )
    except MRPRoutingFailure as error:
        stage = "mrp_phase2" if error.stage == "phase2" else "mrp_routing"
        return _failure(
            round_index, state, stage, error, phase1, members,
            error.partial_results, error.pheromone_before, error.pheromone_after,
            failed_cluster_head=error.cluster_head,
            routing_failure_classification=error.classification,
            failed_phase2_result=error.phase2_result,
            control_energy=error.control_energy,
        )
    routed = list(routing.ch_routing_results)
    selected_routes = dict(routing.selected_routes)
    pheromone_before, pheromone_after = routing.pheromone_before, routing.pheromone_after
    control_energy = getattr(routing, "control_energy", None)
    try:
        plan = build_hybrid_multiflow_plan(
            phase1.selected_cluster_heads, members, selected_routes,
            state.physical_state.live_nodes, len(nodes),
        )
    except Exception as error:
        return _failure(round_index, state, "multiflow_plan", error, phase1, members, routed, pheromone_before, pheromone_after, selected_routes, control_energy=control_energy)
    try:
        energy = evaluate_hybrid_multiflow_energy(
            plan, dist_matrix, base_dists, parameters.mrp,
        )
        combined_e_m, combined_sum = combine_data_and_control_energy(
            energy.e_m_list, energy.e_sum, control_energy,
        )
        persisted = pheromone_after if pheromone_lifecycle is PheromoneLifecycle.PERSIST_ACROSS_ROUNDS else None
        physical_after, newly_dead = apply_baseline_lifecycle(
            state.physical_state, combined_e_m, combined_sum, persisted,
        )
    except Exception as error:
        return _failure(round_index, state, "final_energy", error, phase1, members, routed, pheromone_before, pheromone_after, selected_routes, plan, control_energy=control_energy)
    state_after = HybridState(physical_after, phase1.state_after)
    return HybridRoundResult(
        round_index, phase1, members, tuple(routed), dict(selected_routes), None, plan,
        combined_e_m, combined_sum, state, state_after, newly_dead, True, None, None,
        data_energy=energy.e_sum, ant_control_energy=control_energy,
    )


def run_hybrid(
    nodes: Sequence[object], dist_matrix: Sequence[Sequence[float]], base_dists: Sequence[float],
    initial_state: HybridState, round_contexts: Sequence[HybridRoundContext],
    parameters: HybridParameters, ac_aco_rng: object, mrp_rng: object,
    pheromone_lifecycle: PheromoneLifecycle,
) -> HybridRunResult:
    """Run caller-provided Hybrid contexts until an atomic round failure."""

    if ac_aco_rng is mrp_rng:
        raise ValueError("Hybrid requires separate caller-owned AC-ACO and MRP RNG streams")
    state, rounds = initial_state, []
    for round_index, context in enumerate(round_contexts):
        result = run_hybrid_round(
            nodes, dist_matrix, base_dists, state, context, parameters,
            ac_aco_rng, mrp_rng, pheromone_lifecycle, round_index,
        )
        rounds.append(result)
        state = result.state_after
        if not result.success:
            break
    return HybridRunResult(tuple(rounds), state)


def _failure(
    round_index, state, stage, reason, phase1=None, members=None, routed=(),
    pheromone_before=None, pheromone_after=None, selected_routes=None, plan=None,
    failed_cluster_head=None, routing_failure_classification=None,
    failed_phase2_result=None,
    control_energy=None,
) -> HybridRoundResult:
    physical_after, newly_dead = commit_control_energy_after_failure(
        state.physical_state, control_energy,
    )
    state_after = state if physical_after is state.physical_state else HybridState(
        physical_after, state.ac_aco_state,
    )
    charged = control_energy is not None and not control_energy.is_zero
    return HybridRoundResult(
        round_index, phase1, members, tuple(routed),
        None if selected_routes is None else dict(selected_routes), None, plan,
        control_energy.e_m_list if charged else None,
        control_energy.total_energy if charged else None,
        state, state_after, newly_dead, False, stage, str(reason),
        failed_cluster_head, routing_failure_classification,
        failed_phase2_result,
        data_energy=None, ant_control_energy=control_energy,
    )


def _require_contract(nodes, state, context, parameters, pheromone_lifecycle) -> None:
    if not isinstance(state, HybridState) or not isinstance(context, HybridRoundContext):
        raise ValueError("Hybrid round requires explicit state and hop-count context")
    if not isinstance(parameters, HybridParameters) or not isinstance(pheromone_lifecycle, PheromoneLifecycle):
        raise ValueError("Hybrid parameters and MRP pheromone lifecycle are required")
    if len(nodes) != len(state.physical_state.residual_e):
        raise ValueError("Hybrid residual snapshot must match node count")
    validate_ant_energy_parameters(
        parameters.mrp.charge_ant_energy,
        parameters.mrp.ant_control_packet_bits,
    )
    if parameters.phase3_top_k is not None and (
        not isinstance(parameters.phase3_top_k, int)
        or isinstance(parameters.phase3_top_k, bool)
        or parameters.phase3_top_k < 0
    ):
        raise ValueError("Hybrid phase3_top_k must be None or a non-negative integer")
