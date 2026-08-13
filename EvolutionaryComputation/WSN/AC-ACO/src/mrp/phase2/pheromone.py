"""MRP_PHASE_II sensor-to-sensor local pheromone primitives, Eq. (30)–(31)."""

from collections.abc import Collection, Sequence
import math

from mrp.config import MRPConfig
from mrp.topology_validation import is_valid_sensor_id


class PheromoneInputError(ValueError):
    """Raised when a sensor-to-sensor local-update input is invalid."""


def initialize_sensor_pheromone_state(
    live_nodes: Collection[int],
    dist_matrix: Sequence[Sequence[float]],
    communication_radius: float,
    config: MRPConfig,
) -> dict[int, dict[int, float]]:
    """Table 2 initialization for sensor links; Sink is intentionally absent."""

    node_count = len(dist_matrix)
    live_ids = tuple(live_nodes)
    if not isinstance(config, MRPConfig):
        raise PheromoneInputError("config must be an MRPConfig")
    _require_finite_positive(communication_radius, "communication radius")
    _require_finite_non_negative(config.initial_pheromone, "initial pheromone")
    if any(not _is_sensor_id(sensor_id, node_count) for sensor_id in live_ids):
        raise PheromoneInputError("live_nodes must contain only valid sensor IDs")
    live_ids = tuple(sorted(live_ids))
    if any(len(row) != node_count for row in dist_matrix):
        raise PheromoneInputError("distance matrix must be square over sensor nodes")
    state: dict[int, dict[int, float]] = {}
    for source_id in live_ids:
        state[source_id] = {}
        for candidate_id in live_ids:
            if source_id == candidate_id:
                continue
            distance = dist_matrix[source_id][candidate_id]
            _require_finite_positive(distance, "d_ij")
            state[source_id][candidate_id] = (
                config.initial_pheromone if distance <= communication_radius else 0.0
            )
    return state


def calculate_local_pheromone_deposit(
    source_id: int,
    candidate_id: int,
    residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]],
    lambda_coefficient: float,
) -> float:
    """MRP_PHASE_II Eq. (31): Δτ_ij = λ(E_i + E_j) / d_ij².

    ``lambda_coefficient`` is caller supplied because the paper calls λ a
    coefficient but does not publish a numeric value.  Sink IDs are rejected
    temporarily: this sensor-only primitive makes no claim about final-hop
    MRP semantics.
    """

    source_energy, candidate_energy, distance = _sensor_link_values(
        source_id, candidate_id, residual_e, dist_matrix
    )
    _require_finite_non_negative(lambda_coefficient, "lambda coefficient")
    energy_sum = source_energy + candidate_energy
    distance_squared = distance * distance
    if not math.isfinite(energy_sum) or not math.isfinite(distance_squared):
        raise PheromoneInputError("Eq. (31) intermediate value is not finite")
    deposit = lambda_coefficient * energy_sum / distance_squared
    if not math.isfinite(deposit) or deposit < 0:
        raise PheromoneInputError("Eq. (31) local pheromone deposit is not finite")
    return deposit


def update_local_pheromone(
    source_id: int,
    candidate_id: int,
    old_pheromone: float,
    residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]],
    lambda_coefficient: float,
    config: MRPConfig,
) -> float:
    """MRP_PHASE_II Eq. (30), for an already-decided sensor link only."""

    _require_finite_non_negative(old_pheromone, "old pheromone")
    rho = _rho(config)
    deposit = calculate_local_pheromone_deposit(
        source_id, candidate_id, residual_e, dist_matrix, lambda_coefficient
    )
    updated = (1 - rho) * old_pheromone + rho * deposit
    if not math.isfinite(updated) or updated < 0:
        raise PheromoneInputError("Eq. (30) local pheromone value is not finite")
    return updated


def _sensor_link_values(
    source_id: int,
    candidate_id: int,
    residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    node_count = len(residual_e)
    _validate_sensor_link_ids(source_id, candidate_id, node_count)
    if len(dist_matrix) != node_count or any(len(row) != node_count for row in dist_matrix):
        raise PheromoneInputError("distance matrix must be square over sensor nodes")
    source_energy = residual_e[source_id]
    candidate_energy = residual_e[candidate_id]
    _require_finite_positive(source_energy, "source residual energy")
    _require_finite_positive(candidate_energy, "candidate residual energy")
    distance = dist_matrix[source_id][candidate_id]
    _require_finite_positive(distance, "d_ij")
    return source_energy, candidate_energy, distance


def _validate_sensor_link_ids(source_id: int, candidate_id: int, node_count: int) -> None:
    if source_id == -1 or candidate_id == -1:
        raise PheromoneInputError(
            "Sink IDs are a temporary source-boundary restriction of this local pheromone primitive"
        )
    if not _is_sensor_id(source_id, node_count) or not _is_sensor_id(candidate_id, node_count):
        raise PheromoneInputError("local pheromone link requires valid sensor IDs")
    if source_id == candidate_id:
        raise PheromoneInputError("local pheromone update requires distinct sensor IDs")


def _rho(config: MRPConfig) -> float:
    if not isinstance(config, MRPConfig):
        raise PheromoneInputError("config must be an MRPConfig")
    rho = config.rho
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho):
        raise PheromoneInputError("rho must be finite")
    if not 0 <= rho <= 1:
        raise PheromoneInputError("rho must be in [0, 1] for Eq. (30)")
    return rho


def _is_sensor_id(sensor_id: int, node_count: int) -> bool:
    return not isinstance(sensor_id, bool) and is_valid_sensor_id(sensor_id, node_count)


def _require_finite_positive(value: float, label: str) -> None:
    _require_finite_non_negative(value, label)
    if value == 0:
        raise PheromoneInputError(f"{label} must be positive for a live sensor link")


def _require_finite_non_negative(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PheromoneInputError(f"{label} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise PheromoneInputError(f"{label} must be finite and non-negative")
