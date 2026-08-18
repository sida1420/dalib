"""Experimental Hybrid round with joint lifetime-aware route-set selection."""

from collections.abc import Sequence

from mrp.multi_ch_routing import MRPRoutingFailure
from mrp.phase2.routing import discover_mrp_phase2_routes
from mrp.runner_lifecycle import (
    apply_baseline_lifecycle,
    copy_pheromone_state,
    pheromone_for_discovery,
)
from mrp.runner_types import PheromoneLifecycle
from mrp.topology import assign_members_to_cluster_heads

from .lifetime_selector import select_lifetime_aware_routes
from .phase1 import select_ac_aco_cluster_heads
from .types import (
    HybridLifetimeRoundResult,
    HybridParameters,
    HybridRoundContext,
    HybridState,
    LifetimeAwareCHRoutingResult,
    LifetimeAwareRouteSelection,
    LifetimeRouteCandidate,
)


def run_hybrid_lifetime_round(
    nodes: Sequence[object], dist_matrix, base_dists, state: HybridState,
    context: HybridRoundContext, parameters: HybridParameters,
    ac_aco_rng: object, mrp_rng: object, pheromone_lifecycle: PheromoneLifecycle,
    round_index: int,
) -> HybridLifetimeRoundResult:
    """Run one atomic experimental round without invoking normal Phase III."""

    if ac_aco_rng is mrp_rng:
        raise ValueError("Hybrid requires separate caller-owned AC-ACO and MRP RNG streams")
    phase1 = members = None
    try:
        phase1 = select_ac_aco_cluster_heads(
            nodes, dist_matrix, base_dists, state.physical_state.residual_e,
            state.physical_state.live_nodes, state.ac_aco_state, parameters.ac_aco,
            parameters.mrp, ac_aco_rng, round_index,
        )
        assigned = assign_members_to_cluster_heads(
            nodes, phase1.selected_cluster_heads, parameters.mrp.communication_radius,
            dist_matrix, state.physical_state.residual_e,
        )
        if assigned is None:
            return _failure(round_index, state, "member_assignment", "nearest-CH assignment failed", phase1)
        members = {head: tuple(values) for head, values in assigned.items()}
        phase2_results, pheromone_before, pheromone_after = _discover_candidates(
            phase1.selected_cluster_heads, state, dist_matrix, base_dists,
            context, parameters, mrp_rng, pheromone_lifecycle,
        )
        candidates = {
            head: phase2.unique_routes
            for head, phase2 in zip(phase1.selected_cluster_heads, phase2_results)
        }
        selection = select_lifetime_aware_routes(
            phase1.selected_cluster_heads, members, candidates,
            state.physical_state.live_nodes, state.physical_state.residual_e,
            dist_matrix, base_dists, parameters.mrp,
        )
        routed = tuple(
            LifetimeAwareCHRoutingResult(
                head, phase2,
                LifetimeAwareRouteSelection((LifetimeRouteCandidate(
                    _route_length(selection.selected_routes[head], dist_matrix, base_dists),
                ),), selection.selected_routes[head]),
            )
            for head, phase2 in zip(phase1.selected_cluster_heads, phase2_results)
        )
        persisted = (
            pheromone_after
            if pheromone_lifecycle is PheromoneLifecycle.PERSIST_ACROSS_ROUNDS else None
        )
        physical_after, newly_dead = apply_baseline_lifecycle(
            state.physical_state, selection.energy.e_m_list,
            selection.energy.e_sum, persisted,
        )
    except MRPRoutingFailure as error:
        return _failure(
            round_index, state, "mrp_phase2" if error.stage == "phase2" else "mrp_routing",
            error, phase1, members, failed_cluster_head=error.cluster_head,
            routing_failure_classification=error.classification,
            failed_phase2_result=error.phase2_result,
        )
    except Exception as error:
        stage = "ac_aco_phase1" if phase1 is None else "lifetime_route_selection"
        return _failure(round_index, state, stage, error, phase1, members)

    return HybridLifetimeRoundResult(
        round_index, phase1, members, routed, dict(selection.selected_routes), None,
        selection.plan, selection.energy.e_m_list, selection.energy.e_sum, state,
        HybridState(physical_after, phase1.state_after), newly_dead, True, None, None,
        lifetime_selection=selection,
    )


def _discover_candidates(
    heads, state, dist_matrix, base_dists, context, parameters, rng, lifecycle,
):
    mrp = parameters.mrp
    before = pheromone_for_discovery(
        state.physical_state, lifecycle, dist_matrix, mrp.communication_radius, mrp.config,
    )
    shared, results = before, []
    for head in heads:
        try:
            phase2 = discover_mrp_phase2_routes(
                head, state.physical_state.live_nodes, state.physical_state.residual_e,
                dist_matrix, base_dists, context.hop_counts, mrp.communication_radius,
                mrp.heuristic_bounds, mrp.config, mrp.lambda_coefficient, mrp.ttl,
                mrp.num_sants, rng, mrp.c0, mrp.e_elec, mrp.free_space_coeff,
                mrp.multipath_coeff, mrp.d0, mrp.bit_count, mrp.c, mrp.c1, shared,
            )
        except Exception as error:
            raise MRPRoutingFailure(
                "phase2", head, "MRP_DOMAIN_FAILURE", error,
                pheromone_before=before, pheromone_after=shared,
            ) from error
        shared = phase2.final_pheromone_state
        if phase2.status != "completed":
            raise MRPRoutingFailure(
                "phase2", head, "MRP_DOMAIN_FAILURE", phase2.status,
                phase2_result=phase2, pheromone_before=before, pheromone_after=shared,
            )
        if not phase2.unique_routes:
            classification = (
                "PHYSICAL_CONNECTIVITY_EXHAUSTED"
                if head not in context.hop_counts else "MRP_SEARCH_FAILURE"
            )
            raise MRPRoutingFailure(
                "phase3", head, classification, "Phase II produced no unique successful routes",
                phase2_result=phase2, pheromone_before=before, pheromone_after=shared,
            )
        results.append(phase2)
    return tuple(results), copy_pheromone_state(before), copy_pheromone_state(shared)


def _route_length(route, dist_matrix, base_dists):
    return sum(
        base_dists[source] if target == -1 else dist_matrix[source][target]
        for source, target in zip(route, route[1:])
    )


def _failure(
    round_index, state, stage, reason, phase1=None, members=None,
    failed_cluster_head=None, routing_failure_classification=None,
    failed_phase2_result=None,
):
    return HybridLifetimeRoundResult(
        round_index, phase1, members, (), None, None, None, None, None,
        state, state, (), False, stage, str(reason), failed_cluster_head,
        routing_failure_classification, failed_phase2_result,
    )
