"""Centralized Phase-II coordination of the existing MRP ant primitives."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import math

from mrp.config import MRPConfig
from mrp.phase2.control_energy import (
    AntControlEnergy,
    AntControlEnergyLedger,
    validate_ant_energy_parameters,
    zero_ant_control_energy,
)
from mrp.phase2.bant_feedback import (
    BANTFeedbackInputError,
    BANTFeedbackResult,
    _apply_bant_feedback_to_validated_state,
    _copy_sensor_pheromone_state,
)
from mrp.phase2.heuristic import (
    HeuristicBounds,
    HeuristicInputError,
    _validate_heuristic_parameters,
)
from mrp.phase2.pheromone import initialize_sensor_pheromone_state
from mrp.phase2.route_quality import (
    BANTPreparation,
    RouteQuality,
    RouteQualityInputError,
    _evaluate_route_quality_from_validated_snapshot,
    _validate_parameters,
    _validate_snapshot,
    prepare_bant,
)
from mrp.phase2.sant import SANTRouteResult, _construct_sant_from_validated_snapshot


class MRPPhase2InputError(ValueError):
    """Raised when the caller does not provide a complete Phase-II contract."""


@dataclass(frozen=True)
class MRPPhase2AntResult:
    """One sequential SANT attempt and any BANT work it produced."""

    ant_index: int
    sant_result: SANTRouteResult
    route_quality: RouteQuality | None
    bant_preparation: BANTPreparation | None
    bant_feedback: BANTFeedbackResult | None
    f_best_before: float | None
    f_best_after: float | None
    feedback_status: str


@dataclass(frozen=True)
class MRPPhase2Result:
    """Phase-II discovery output; no Phase-III path selection is performed."""

    start_cluster_head: int
    ant_results: tuple[MRPPhase2AntResult, ...]
    successful_ant_results: tuple[MRPPhase2AntResult, ...]
    failed_ant_results: tuple[MRPPhase2AntResult, ...]
    unique_routes: tuple[tuple[int, ...], ...]
    final_pheromone_state: dict[int, dict[int, float]]
    f_best: float | None
    num_sants_requested: int
    num_sants_executed: int
    num_sants_succeeded: int
    num_unique_routes: int
    multipath_ready: bool
    status: str
    halted_ant_index: int | None
    control_energy: AntControlEnergy | None = None


def discover_mrp_phase2_routes(
    start_cluster_head: int,
    live_nodes: Collection[int],
    residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]],
    base_dists: Sequence[float],
    hop_counts: Mapping[int, int],
    communication_radius: float,
    bounds: HeuristicBounds,
    config: MRPConfig,
    lambda_coefficient: float,
    ttl: int,
    num_sants: int,
    rng: object,
    c0: float,
    e_elec: float,
    free_space_coeff: float,
    multipath_coeff: float,
    d0: float,
    bit_count: float,
    c: float,
    c1: float,
    initial_pheromone_state: Mapping[int, Mapping[int, float]] | None = None,
    *,
    charge_ant_energy: bool = False,
    ant_control_packet_bits: int | None = None,
) -> MRPPhase2Result:
    """Discover MRP CH-to-Sink paths with stable, sequential ant execution.

    This is a SIMULATOR_IMPLEMENTATION_MAPPING of the paper's distributed
    behavior.  Shared pheromone evolves after each ant, while each SANT call
    creates its own visited route and TTL.  The first successful route only
    initializes historical best quality, per the approved compatibility rule.
    """

    _validate_orchestration_inputs(num_sants, c, c1)
    state = _validated_discovery_state(
        live_nodes, residual_e, dist_matrix, base_dists, communication_radius,
        bounds, config, c0, e_elec, free_space_coeff, multipath_coeff, d0, bit_count,
        initial_pheromone_state,
    )
    return _run_validated_phase2_discovery(
        start_cluster_head, live_nodes, residual_e, dist_matrix, base_dists,
        hop_counts, communication_radius, bounds, config, lambda_coefficient,
        ttl, num_sants, rng, c0, e_elec, free_space_coeff, multipath_coeff,
        d0, bit_count, c, c1, state,
        charge_ant_energy=charge_ant_energy,
        ant_control_packet_bits=ant_control_packet_bits,
    )


def _run_validated_phase2_discovery(
    start_cluster_head, live_nodes, residual_e, dist_matrix, base_dists,
    hop_counts, communication_radius, bounds, config, lambda_coefficient,
    ttl, num_sants, rng, c0, e_elec, free_space_coeff, multipath_coeff,
    d0, bit_count, c, c1, state, *, charge_ant_energy=False,
    ant_control_packet_bits=None, control_ledger=None,
) -> MRPPhase2Result:
    """Run ants after the public or multi-CH boundary validated shared inputs."""

    ant_results: list[MRPPhase2AntResult] = []
    successful: list[MRPPhase2AntResult] = []
    failed: list[MRPPhase2AntResult] = []
    best_quality: float | None = None
    status, halted_ant_index = "completed", None
    if control_ledger is None:
        control_ledger = _control_ledger(
            charge_ant_energy, ant_control_packet_bits, len(residual_e),
            dist_matrix, base_dists, e_elec, free_space_coeff, multipath_coeff, d0,
        )

    for ant_index in range(num_sants):
        sant_result = _construct_sant_from_validated_snapshot(
            start_cluster_head, live_nodes, residual_e, dist_matrix, base_dists,
            hop_counts, communication_radius, state, bounds, config,
            lambda_coefficient, ttl, rng,
            None if control_ledger is None else control_ledger.record_sant_hop,
        )
        state = sant_result.pheromone_state
        if not sant_result.success:
            ant = MRPPhase2AntResult(
                ant_index, sant_result, None, None, None, None, best_quality, "sant_failed"
            )
            ant_results.append(ant)
            failed.append(ant)
            continue

        quality = _evaluate_route_quality_from_validated_snapshot(
            sant_result.path, live_nodes, residual_e, dist_matrix, base_dists, config,
            c0, e_elec, free_space_coeff, multipath_coeff, d0, bit_count,
        )
        preparation = prepare_bant(quality)
        if control_ledger is not None:
            control_ledger.record_bant_route(preparation.reverse_path)
        previous_best = best_quality
        feedback: BANTFeedbackResult | None = None
        feedback_status = "first_route_best_initialized"
        if previous_best is None:
            best_quality = quality.quality
        else:
            feedback = _apply_bant_feedback_to_validated_state(
                preparation, previous_best, state, config, c, c1,
            )
            state = feedback.pheromone_state
            best_quality = feedback.updated_best_quality
            feedback_status = "bant_feedback_applied"
            if not feedback.transition_compatible:
                feedback_status = "invalid_negative_pheromone_state"
                status, halted_ant_index = feedback_status, ant_index

        ant = MRPPhase2AntResult(
            ant_index, sant_result, quality, preparation, feedback,
            previous_best, best_quality, feedback_status,
        )
        ant_results.append(ant)
        successful.append(ant)
        if halted_ant_index is not None:
            break

    unique_routes = _unique_paths(successful)
    return MRPPhase2Result(
        start_cluster_head,
        tuple(ant_results),
        tuple(successful),
        tuple(failed),
        unique_routes,
        _copy_pheromone_state(state),
        best_quality,
        num_sants,
        len(ant_results),
        len(successful),
        len(unique_routes),
        len(unique_routes) >= 2,
        status,
        halted_ant_index,
        (
            zero_ant_control_energy(len(residual_e))
            if control_ledger is None else control_ledger.snapshot()
        ),
    )


def _control_ledger(
    enabled, packet_bits, node_count, dist_matrix, base_dists,
    e_elec, free_space_coeff, multipath_coeff, d0,
):
    try:
        validate_ant_energy_parameters(enabled, packet_bits)
    except ValueError as error:
        raise MRPPhase2InputError(str(error)) from error
    if not enabled:
        return None
    try:
        return AntControlEnergyLedger(
            node_count, packet_bits, dist_matrix, base_dists,
            e_elec, free_space_coeff, multipath_coeff, d0,
        )
    except ValueError as error:
        raise MRPPhase2InputError(str(error)) from error


def _validate_orchestration_inputs(num_sants: int, c: float, c1: float) -> None:
    if not isinstance(num_sants, int) or isinstance(num_sants, bool) or num_sants <= 0:
        raise MRPPhase2InputError("num_sants must be a positive integer")
    _require_finite(c, "c")
    _require_finite(c1, "c1")


def _validated_discovery_state(
    live_nodes, residual_e, dist_matrix, base_dists, communication_radius,
    bounds, config, c0, e_elec, free_space_coeff, multipath_coeff, d0, bit_count,
    initial_pheromone_state,
):
    """Validate immutable discovery inputs once before trusted per-ant work."""

    try:
        _validate_snapshot(residual_e, dist_matrix, base_dists)
        _validate_parameters(
            config, c0, e_elec, free_space_coeff, multipath_coeff, d0, bit_count,
        )
        _validate_heuristic_parameters(bounds, config)
        expected = initialize_sensor_pheromone_state(
            live_nodes, dist_matrix, communication_radius, config,
        )
        if initial_pheromone_state is None:
            return expected
        state = _copy_sensor_pheromone_state(initial_pheromone_state)
    except (
        RouteQualityInputError, BANTFeedbackInputError, HeuristicInputError,
        ValueError, TypeError,
    ) as error:
        raise MRPPhase2InputError(str(error)) from error
    if set(state) != set(expected) or any(
        set(state[source]) != set(expected[source]) for source in expected
    ):
        raise MRPPhase2InputError(
            "initial pheromone state must cover every live directed sensor link"
        )
    return state


def _unique_paths(results: Sequence[MRPPhase2AntResult]) -> tuple[tuple[int, ...], ...]:
    seen: set[tuple[int, ...]] = set()
    unique: list[tuple[int, ...]] = []
    for result in results:
        path = result.sant_result.path
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return tuple(unique)


def _copy_pheromone_state(state: Mapping[int, Mapping[int, float]]) -> dict[int, dict[int, float]]:
    """Detach the aggregate snapshot from the final per-ant diagnostic state."""

    return {source_id: dict(outgoing) for source_id, outgoing in state.items()}


def _require_finite(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MRPPhase2InputError(f"{label} must be finite")
