"""AC-ACO Direct-first routing with MRP fallback and Phase-III Top-K."""

from collections.abc import Sequence
from functools import partial

from mrp.multi_ch_routing import MRPRoutingFailure, route_cluster_heads_with_mrp
from mrp.phase2.routing import discover_mrp_phase2_routes
from mrp.phase2.control_energy import (
    validate_ant_energy_parameters, zero_ant_control_energy,
)
from mrp.phase2.sant_validation import sink_is_reachable
from mrp.phase3.path_selection import select_phase3_route
from mrp.runner_lifecycle import (
    apply_baseline_lifecycle,
    combine_data_and_control_energy,
    commit_control_energy_after_failure,
    copy_pheromone_state,
)
from mrp.runner_types import PheromoneLifecycle
from mrp.topology import assign_members_to_cluster_heads

from .multiflow import build_hybrid_multiflow_plan
from .multiflow_energy import evaluate_hybrid_multiflow_energy
from .phase1 import select_ac_aco_cluster_heads
from .types import (
    DirectMRPTopKDiagnostics,
    DirectMRPTopKParameters,
    DirectMRPTopKRoundResult,
    HybridRoundContext,
    HybridState,
)


def run_direct_mrp_topk_round(
    nodes: Sequence[object],
    dist_matrix: Sequence[Sequence[float]],
    base_dists: Sequence[float],
    state: HybridState,
    context: HybridRoundContext,
    parameters: DirectMRPTopKParameters,
    ac_aco_rng: object,
    mrp_rng: object,
    pheromone_lifecycle: PheromoneLifecycle,
    round_index: int,
) -> DirectMRPTopKRoundResult:
    """Run the project Direct-first + MRP-fallback + Top-K adaptation.

    Direct reachability is decided before the common MRP engine is called, so
    a directly reachable CH creates no SANT, BANT, AANT, or discovery attempt.
    """

    _require_contract(nodes, state, context, parameters, pheromone_lifecycle)
    if ac_aco_rng is mrp_rng:
        raise ValueError("Direct-first mode requires separate AC-ACO and MRP RNG streams")

    try:
        phase1 = select_ac_aco_cluster_heads(
            nodes,
            dist_matrix,
            base_dists,
            state.physical_state.residual_e,
            state.physical_state.live_nodes,
            state.ac_aco_state,
            parameters.ac_aco,
            parameters.mrp,
            ac_aco_rng,
            round_index,
        )
    except Exception as error:
        return _failure(round_index, state, "ac_aco_phase1", error)

    heads = tuple(phase1.selected_cluster_heads)
    direct_heads = tuple(
        head
        for head in heads
        if sink_is_reachable(
            head, base_dists, parameters.mrp.communication_radius,
        )
    )
    direct_set = set(direct_heads)
    fallback_heads = tuple(head for head in heads if head not in direct_set)

    members = assign_members_to_cluster_heads(
        nodes,
        heads,
        parameters.mrp.communication_radius,
        dist_matrix,
        state.physical_state.residual_e,
    )
    if members is None:
        return _failure(
            round_index,
            state,
            "member_assignment",
            "nearest-CH assignment failed",
            phase1,
            direct_heads=direct_heads,
            fallback_heads=fallback_heads,
        )
    members = {head: tuple(member_ids) for head, member_ids in members.items()}

    routed = ()
    control_energy = zero_ant_control_energy(len(nodes))
    fallback_routes: dict[int, tuple[int, ...]] = {}
    pheromone_before = pheromone_after = None
    if fallback_heads:
        try:
            routing = route_cluster_heads_with_mrp(
                fallback_heads,
                state.physical_state.live_nodes,
                state.physical_state,
                dist_matrix,
                base_dists,
                context.hop_counts,
                parameters.mrp,
                mrp_rng,
                pheromone_lifecycle,
                phase2_discovery=discover_mrp_phase2_routes,
                phase3_selector=partial(
                    select_phase3_route, top_k=parameters.phase3_top_k,
                ),
            )
        except MRPRoutingFailure as error:
            stage = "mrp_phase2" if error.stage == "phase2" else "mrp_routing"
            return _failure(
                round_index,
                state,
                stage,
                error,
                phase1,
                members,
                error.partial_results,
                error.pheromone_before,
                error.pheromone_after,
                failed_cluster_head=error.cluster_head,
                routing_failure_classification=error.classification,
                failed_phase2_result=error.phase2_result,
                direct_heads=direct_heads,
                fallback_heads=fallback_heads,
                discovery_failed=True,
                control_energy=error.control_energy,
            )
        routed = routing.ch_routing_results
        fallback_routes = dict(routing.selected_routes)
        pheromone_before = routing.pheromone_before
        pheromone_after = routing.pheromone_after
        control_energy = getattr(routing, "control_energy", control_energy)
    elif pheromone_lifecycle is PheromoneLifecycle.PERSIST_ACROSS_ROUNDS:
        pheromone_before = copy_pheromone_state(state.physical_state.pheromone_state)
        pheromone_after = copy_pheromone_state(state.physical_state.pheromone_state)

    selected_routes = {
        head: (head, -1) if head in direct_set else fallback_routes[head]
        for head in heads
    }
    diagnostics = _diagnostics(direct_heads, fallback_heads, routed)
    try:
        plan = build_hybrid_multiflow_plan(
            heads,
            members,
            selected_routes,
            state.physical_state.live_nodes,
            len(nodes),
        )
    except Exception as error:
        return _failure(
            round_index,
            state,
            "multiflow_plan",
            error,
            phase1,
            members,
            routed,
            pheromone_before,
            pheromone_after,
            selected_routes,
            direct_heads=direct_heads,
            fallback_heads=fallback_heads,
            control_energy=control_energy,
        )
    try:
        energy = evaluate_hybrid_multiflow_energy(
            plan, dist_matrix, base_dists, parameters.mrp,
        )
        combined_e_m, combined_sum = combine_data_and_control_energy(
            energy.e_m_list, energy.e_sum, control_energy,
        )
        persisted = (
            pheromone_after
            if pheromone_lifecycle is PheromoneLifecycle.PERSIST_ACROSS_ROUNDS
            else None
        )
        physical_after, newly_dead = apply_baseline_lifecycle(
            state.physical_state, combined_e_m, combined_sum, persisted,
        )
    except Exception as error:
        return _failure(
            round_index,
            state,
            "final_energy",
            error,
            phase1,
            members,
            routed,
            pheromone_before,
            pheromone_after,
            selected_routes,
            plan,
            direct_heads=direct_heads,
            fallback_heads=fallback_heads,
            control_energy=control_energy,
        )

    state_after = HybridState(physical_after, phase1.state_after)
    return DirectMRPTopKRoundResult(
        round_index=round_index, phase1_result=phase1, members_by_head=members,
        ch_routing_results=tuple(routed), selected_routes=dict(selected_routes),
        final_topology=None, final_plan=plan, e_m_list=combined_e_m,
        e_sum=combined_sum, state_before=state, state_after=state_after,
        newly_dead=newly_dead, success=True, failure_stage=None,
        failure_reason=None, data_energy=energy.e_sum,
        ant_control_energy=control_energy, direct_mrp_diagnostics=diagnostics,
    )


def _failure(
    round_index,
    state,
    stage,
    reason,
    phase1=None,
    members=None,
    routed=(),
    pheromone_before=None,
    pheromone_after=None,
    selected_routes=None,
    plan=None,
    *,
    failed_cluster_head=None,
    routing_failure_classification=None,
    failed_phase2_result=None,
    direct_heads=(),
    fallback_heads=(),
    discovery_failed=False,
    control_energy=None,
) -> DirectMRPTopKRoundResult:
    diagnostics = _diagnostics(
        direct_heads,
        fallback_heads,
        routed,
        failed_phase2_result,
        discovery_failed,
    )
    physical_after, newly_dead = commit_control_energy_after_failure(
        state.physical_state, control_energy,
    )
    state_after = (
        state if physical_after is state.physical_state
        else HybridState(physical_after, state.ac_aco_state)
    )
    charged = control_energy is not None and not control_energy.is_zero
    return DirectMRPTopKRoundResult(
        round_index=round_index, phase1_result=phase1, members_by_head=members,
        ch_routing_results=tuple(routed),
        selected_routes=None if selected_routes is None else dict(selected_routes),
        final_topology=None, final_plan=plan,
        e_m_list=control_energy.e_m_list if charged else None,
        e_sum=control_energy.total_energy if charged else None,
        state_before=state, state_after=state_after, newly_dead=newly_dead,
        success=False, failure_stage=stage, failure_reason=str(reason),
        failed_cluster_head=failed_cluster_head,
        routing_failure_classification=routing_failure_classification,
        failed_phase2_result=failed_phase2_result, data_energy=None,
        ant_control_energy=control_energy, direct_mrp_diagnostics=diagnostics,
    )


def _diagnostics(
    direct_heads,
    fallback_heads,
    routed,
    failed_phase2_result=None,
    discovery_failed=False,
) -> DirectMRPTopKDiagnostics:
    completed = tuple(routed)
    phase2_results = [item.phase2_result for item in completed]
    if failed_phase2_result is not None:
        phase2_results.append(failed_phase2_result)
    sant_count = sum(item.num_sants_executed for item in phase2_results)
    bant_count = sum(
        ant.bant_preparation is not None
        for item in phase2_results
        for ant in item.ant_results
    )
    aant_count = sum(
        step.mode == "AANT"
        for item in phase2_results
        for ant in item.ant_results
        for step in ant.sant_result.trace
    )
    routes_before = sum(item.phase2_result.num_unique_routes for item in completed)
    routes_after = sum(len(item.phase3_result.candidates) for item in completed)
    discoveries = len(completed) + int(discovery_failed)
    return DirectMRPTopKDiagnostics(
        tuple(direct_heads),
        tuple(fallback_heads),
        discoveries,
        0,
        routes_before,
        routes_after,
        routes_before - routes_after,
        sant_count,
        bant_count,
        aant_count,
    )


def _require_contract(nodes, state, context, parameters, pheromone_lifecycle) -> None:
    if not isinstance(state, HybridState) or not isinstance(context, HybridRoundContext):
        raise ValueError("Direct-first round requires explicit state and hop-count context")
    if not isinstance(parameters, DirectMRPTopKParameters):
        raise ValueError("Direct-first mode requires DirectMRPTopKParameters")
    if not isinstance(pheromone_lifecycle, PheromoneLifecycle):
        raise ValueError("Direct-first mode requires an MRP pheromone lifecycle")
    if len(nodes) != len(state.physical_state.residual_e):
        raise ValueError("Direct-first residual snapshot must match node count")
    validate_ant_energy_parameters(
        parameters.mrp.charge_ant_energy,
        parameters.mrp.ant_control_packet_bits,
    )
    if (
        not isinstance(parameters.phase3_top_k, int)
        or isinstance(parameters.phase3_top_k, bool)
        or parameters.phase3_top_k <= 0
    ):
        raise ValueError("Direct-first phase3_top_k must be a positive integer")
    if (
        not isinstance(parameters.mrp.ttl, int)
        or isinstance(parameters.mrp.ttl, bool)
        or parameters.mrp.ttl <= 0
    ):
        raise ValueError("Direct-first MRP TTL must be a positive integer")
