"""Import-safe Pure-MRP integration with the repository's common evaluator."""

from collections.abc import Sequence

from evaluate import energy_consumption
from mrp.phase1.clustering import select_cluster_head
from mrp.phase2.control_energy import validate_ant_energy_parameters
from mrp.phase2.routing import discover_mrp_phase2_routes
from mrp.phase3.path_selection import select_phase3_route
from mrp.multi_ch_routing import MRPRoutingFailure, route_cluster_heads_with_mrp
from mrp.runner_lifecycle import (
    PureMRPIntegrationError,
    apply_baseline_lifecycle,
    combine_data_and_control_energy,
    commit_control_energy_after_failure,
    copy_pheromone_state,
    validate_pure_mrp_state,
)
from mrp.runner_types import (
    PheromoneLifecycle,
    PureMRPParameters,
    PureMRPRoundContext,
    PureMRPRoundResult,
    PureMRPRunResult,
    PureMRPState,
)
from mrp.topology import materialize_topology


def run_pure_mrp_round(
    nodes: Sequence[object],
    dist_matrix: Sequence[Sequence[float]],
    base_dists: Sequence[float],
    state: PureMRPState,
    context: PureMRPRoundContext,
    parameters: PureMRPParameters,
    rng: object,
    pheromone_lifecycle: PheromoneLifecycle,
    round_index: int,
) -> PureMRPRoundResult:
    """Run one event-defined MRP round and commit data plus optional ant energy."""

    _require_round_contract(
        nodes, dist_matrix, state, context, parameters, pheromone_lifecycle,
    )
    phase1 = phase2 = phase3 = topology = None
    pheromone_before = pheromone_after = None
    try:
        phase1 = select_cluster_head(
            nodes, state.live_nodes, state.residual_e, context.event_nodes,
            context.event_signal_strengths, dist_matrix,
            parameters.communication_radius, parameters.config,
        )
    except Exception as error:
        return _failure(round_index, state, "phase1", error)
    try:
        routing = route_cluster_heads_with_mrp(
            (phase1.cluster_head,), state.live_nodes, state, dist_matrix,
            base_dists, context.hop_counts, parameters, rng,
            pheromone_lifecycle,
            phase2_discovery=discover_mrp_phase2_routes,
            phase3_selector=select_phase3_route,
        )
    except MRPRoutingFailure as error:
        return _failure(
            round_index, state, error.stage, error, phase1,
            error.pheromone_before, error.phase2_result,
            error.pheromone_after, control_energy=error.control_energy,
        )
    route_result = routing.ch_routing_results[0]
    phase2, phase3 = route_result.phase2_result, route_result.phase3_result
    pheromone_before, pheromone_after = (
        routing.pheromone_before, routing.pheromone_after,
    )
    try:
        topology = materialize_topology(
            [phase1.cluster_head], {phase1.cluster_head: phase1.members},
            {phase1.cluster_head: phase3.selected_route}, state.live_nodes,
            len(nodes),
        )
        if topology is None:
            raise PureMRPIntegrationError(
                "materialize_topology rejected selected MRP route"
            )
    except Exception as error:
        return _failure(
            round_index, state, "topology", error, phase1,
            pheromone_before, phase2, pheromone_after, phase3,
            control_energy=routing.control_energy,
        )
    try:
        data_e_m, data_sum = energy_consumption(
            nodes, topology, parameters.d0, parameters.bit_count,
            parameters.ctrl_bit, base_dists, dist_matrix, parameters.e_elec,
            parameters.e_agg, parameters.free_space_coeff,
            parameters.multipath_coeff,
        )
        combined_e_m, combined_sum = combine_data_and_control_energy(
            data_e_m, data_sum, routing.control_energy,
        )
        persisted = (
            pheromone_after
            if pheromone_lifecycle is PheromoneLifecycle.PERSIST_ACROSS_ROUNDS
            else None
        )
        state_after, newly_dead = apply_baseline_lifecycle(
            state, combined_e_m, combined_sum, persisted,
        )
    except Exception as error:
        return _failure(
            round_index, state, "energy", error, phase1, pheromone_before,
            phase2, pheromone_after, phase3, topology,
            control_energy=routing.control_energy,
        )
    return PureMRPRoundResult(
        round_index, phase1, phase2, phase3, phase3.selected_route, topology,
        combined_e_m, combined_sum, state, state_after,
        copy_pheromone_state(pheromone_before),
        copy_pheromone_state(pheromone_after), newly_dead, True, None, None,
        data_energy=data_sum, ant_control_energy=routing.control_energy,
    )


def run_pure_mrp(
    nodes, dist_matrix, base_dists, initial_state, round_contexts, parameters,
    rng, pheromone_lifecycle,
) -> PureMRPRunResult:
    """Run caller-provided event contexts until one round fails."""

    state, rounds = initial_state, []
    for round_index, context in enumerate(round_contexts):
        result = run_pure_mrp_round(
            nodes, dist_matrix, base_dists, state, context, parameters, rng,
            pheromone_lifecycle, round_index,
        )
        rounds.append(result)
        state = result.state_after
        if not result.success:
            break
    return PureMRPRunResult(tuple(rounds), state)


def _failure(
    round_index, state, stage, reason, phase1=None, pheromone_before=None,
    phase2=None, pheromone_after=None, phase3=None, topology=None,
    control_energy=None,
) -> PureMRPRoundResult:
    state_after, newly_dead = commit_control_energy_after_failure(
        state, control_energy,
    )
    charged = control_energy is not None and not control_energy.is_zero
    return PureMRPRoundResult(
        round_index, phase1, phase2, phase3, None, topology,
        control_energy.e_m_list if charged else None,
        control_energy.total_energy if charged else None,
        state, state_after, copy_pheromone_state(pheromone_before),
        copy_pheromone_state(pheromone_after), newly_dead, False, stage,
        str(reason), data_energy=None, ant_control_energy=control_energy,
    )


def _require_round_contract(
    nodes, dist_matrix, state, context, parameters, pheromone_lifecycle,
) -> None:
    if not isinstance(state, PureMRPState) or not isinstance(
        context, PureMRPRoundContext,
    ):
        raise PureMRPIntegrationError(
            "Pure-MRP round requires state and explicit event context"
        )
    if not isinstance(parameters, PureMRPParameters):
        raise PureMRPIntegrationError("Pure-MRP round requires complete parameters")
    if not isinstance(pheromone_lifecycle, PheromoneLifecycle):
        raise PureMRPIntegrationError("pheromone lifecycle policy is required")
    if len(nodes) != len(state.residual_e):
        raise PureMRPIntegrationError("residual energy must match node count")
    validate_ant_energy_parameters(
        parameters.charge_ant_energy, parameters.ant_control_packet_bits,
    )
    validate_pure_mrp_state(
        state, len(nodes), dist_matrix, parameters.communication_radius,
        parameters.config,
    )
