"""MRP_PHASE_II Eq. (27)–(29), without ants or pheromone mutation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from mrp.config import MRPConfig
from mrp.topology_validation import is_valid_sensor_id


class HeuristicInputError(ValueError):
    """Raised when paper-defined heuristic inputs are mathematically invalid."""


@dataclass(frozen=True)
class HeuristicBounds:
    """Caller-provided Eq. (27) bounds; paper supplies no numeric values.

    ``mu_*`` and ``eta_*`` are PAPER_DEFINED_VALUE_UNREPORTED / later
    USER_SELECTED_PARAMETER values.  They must never receive guessed defaults.
    """

    mu_min: float
    mu_max: float
    eta_min: float
    eta_max: float


def calculate_theta(
    source_id: int,
    candidate_id: int,
    residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]],
    base_dists: Sequence[float],
    hop_counts: Mapping[int, int],
    config: MRPConfig,
) -> float:
    """MRP_PHASE_II Eq. (29), using existing energy/distance snapshots."""

    values = _validated_values(
        source_id, candidate_id, residual_e, dist_matrix, base_dists, hop_counts,
        config, False,
    )
    return _calculate_theta_from_values(values, config.k6)


def calculate_mu(
    source_id: int,
    candidate_id: int,
    residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]],
    base_dists: Sequence[float],
    hop_counts: Mapping[int, int],
    config: MRPConfig,
) -> float:
    """MRP_PHASE_II Eq. (28), μ_ij = E_j^k4 / d_ij^k5 + θ_ij."""

    return _calculate_mu_from_snapshot(
        source_id, candidate_id, residual_e, dist_matrix, base_dists,
        hop_counts, config, False,
    )


def _calculate_mu_from_validated_snapshot(
    source_id, candidate_id, residual_e, dist_matrix, base_dists, hop_counts, config,
) -> float:
    """Internal Eq. (28) hot path after shared snapshot validation."""

    return _calculate_mu_from_snapshot(
        source_id, candidate_id, residual_e, dist_matrix, base_dists,
        hop_counts, config, True,
    )


def _calculate_mu_from_snapshot(
    source_id, candidate_id, residual_e, dist_matrix, base_dists, hop_counts,
    config, snapshot_validated,
) -> float:
    values = _validated_values(
        source_id, candidate_id, residual_e, dist_matrix, base_dists, hop_counts,
        config, snapshot_validated,
    )
    mu = values.base_component + _calculate_theta_from_values(values, config.k6)
    if not math.isfinite(mu) or mu < 0:
        raise HeuristicInputError("Eq. (28) heuristic is not finite")
    return mu


def calculate_eta(mu: float, bounds: HeuristicBounds) -> float:
    """MRP_PHASE_II Eq. (27), including its paper-defined piecewise bounds."""

    _require_finite_non_negative(mu, "mu")
    _validate_bounds(bounds)
    return _calculate_eta_from_validated_bounds(mu, bounds)


def _calculate_eta_from_validated_bounds(mu: float, bounds: HeuristicBounds) -> float:
    """Internal Eq. (27) hot path for already validated bounds and finite mu."""

    if mu > bounds.mu_max:
        return bounds.eta_max
    if mu >= bounds.mu_min:
        return mu
    return bounds.eta_min


@dataclass(frozen=True)
class _HeuristicValues:
    base_component: float
    distance_to_sink_source: float
    distance_to_sink_candidate: float
    hop_source: int
    hop_candidate: int


def _validated_values(
    source_id: int,
    candidate_id: int,
    residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]],
    base_dists: Sequence[float],
    hop_counts: Mapping[int, int],
    config: MRPConfig,
    snapshot_validated: bool,
) -> _HeuristicValues:
    node_count = len(residual_e)
    _validate_sensor_pair(source_id, candidate_id, node_count)
    if not snapshot_validated:
        if len(base_dists) != node_count or len(dist_matrix) != node_count:
            raise HeuristicInputError("base distances and distance matrix must match residual energy")
        if any(len(row) != node_count for row in dist_matrix):
            raise HeuristicInputError("distance matrix must be square over sensor nodes")
        _validate_heuristic_parameters(None, config)
        for energy in residual_e:
            _require_finite_non_negative(energy, "residual energy")
    _require_finite_non_negative(residual_e[candidate_id], "candidate residual energy")
    distance = dist_matrix[source_id][candidate_id]
    _require_positive_finite(distance, "d_ij")
    source_sink_distance = base_dists[source_id]
    candidate_sink_distance = base_dists[candidate_id]
    _require_finite_non_negative(source_sink_distance, "d_is")
    _require_positive_finite(candidate_sink_distance, "d_js")
    hop_source = _hop_count(hop_counts, source_id)
    hop_candidate = _hop_count(hop_counts, candidate_id)
    base_component = residual_e[candidate_id] ** config.k4 / distance ** config.k5
    if not math.isfinite(base_component):
        raise HeuristicInputError("Eq. (28) base heuristic is not finite")
    return _HeuristicValues(
        base_component,
        source_sink_distance,
        candidate_sink_distance,
        hop_source,
        hop_candidate,
    )


def _validate_heuristic_parameters(bounds: HeuristicBounds | None, config: MRPConfig) -> None:
    if not isinstance(config, MRPConfig):
        raise HeuristicInputError("heuristic calculation requires an MRPConfig")
    _require_positive_finite(config.k4, "k4")
    _require_positive_finite(config.k5, "k5")
    _require_positive_finite(config.k6, "k6")
    if bounds is not None:
        _validate_bounds(bounds)


def _validate_sensor_pair(source_id: int, candidate_id: int, node_count: int) -> None:
    if candidate_id == -1:
        raise HeuristicInputError("Eq. (29) is undefined for a Sink candidate (d_js=0)")
    if not _is_valid_phase2_sensor_id(source_id, node_count) or not _is_valid_phase2_sensor_id(candidate_id, node_count):
        raise HeuristicInputError("source and candidate must be valid sensor IDs")
    if source_id == candidate_id:
        raise HeuristicInputError("Eq. (29) requires distinct source and candidate sensors")


def _is_valid_phase2_sensor_id(sensor_id: int, node_count: int) -> bool:
    """Keep the existing sensor-ID range contract, excluding bool aliases."""

    return not isinstance(sensor_id, bool) and is_valid_sensor_id(sensor_id, node_count)


def _calculate_theta_from_values(values: _HeuristicValues, k6: float) -> float:
    if values.hop_candidate > values.hop_source:
        return 0.0
    ratio = values.distance_to_sink_source / values.distance_to_sink_candidate
    theta = values.base_component * ratio**k6
    if not math.isfinite(theta) or theta < 0:
        raise HeuristicInputError("Eq. (29) direction term is not finite")
    return theta


def _hop_count(hop_counts: Mapping[int, int], sensor_id: int) -> int:
    if sensor_id not in hop_counts:
        raise HeuristicInputError(f"missing hop count for sensor {sensor_id}")
    hop_count = hop_counts[sensor_id]
    if isinstance(hop_count, bool) or not isinstance(hop_count, int) or hop_count < 0:
        raise HeuristicInputError("hop count must be a non-negative integer")
    return hop_count


def _validate_bounds(bounds: HeuristicBounds) -> None:
    if not isinstance(bounds, HeuristicBounds):
        raise HeuristicInputError("Eq. (27) requires caller-provided heuristic bounds")
    _require_finite_non_negative(bounds.mu_min, "mu_min")
    _require_finite_non_negative(bounds.mu_max, "mu_max")
    _require_finite_non_negative(bounds.eta_min, "eta_min")
    _require_finite_non_negative(bounds.eta_max, "eta_max")
    if bounds.mu_min > bounds.mu_max or bounds.eta_min > bounds.eta_max:
        raise HeuristicInputError("heuristic min bound cannot exceed max bound")


def _require_positive_finite(value: float, label: str) -> None:
    _require_finite_non_negative(value, label)
    if value == 0:
        raise HeuristicInputError(f"{label} must be finite and positive")


def _require_finite_non_negative(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HeuristicInputError(f"{label} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise HeuristicInputError(f"{label} must be finite and non-negative")
