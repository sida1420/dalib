"""MRP_PHASE_II Eq. (26), without ant movement or random sampling."""

from collections.abc import Iterable, Mapping, Sequence
import math

from mrp.config import MRPConfig
from mrp.phase2.heuristic import HeuristicInputError
from mrp.topology_validation import is_valid_sensor_id


class NoValidTransitionError(HeuristicInputError):
    """Raised when Eq. (26) has no positive normalizing denominator."""


def transition_probabilities(
    source_id: int,
    candidate_ids: Iterable[int],
    visited_nodes: Iterable[int],
    pheromones: Mapping[int, float],
    heuristic_values: Mapping[int, float],
    dist_matrix: Sequence[Sequence[float]],
    communication_radius: float,
    config: MRPConfig,
) -> dict[int, float]:
    """MRP_PHASE_II Eq. (26) distribution over unvisited radio neighbors.

    Omitted candidates have paper probability zero: they are visited or not in
    N_i.  This function neither selects a next hop nor changes pheromone.
    """

    # Materialize once: callers may provide generator-based ant state later.
    # Eq. (26) must see the same candidate and visited sets during validation
    # and distribution construction.
    candidates = tuple(candidate_ids)
    visited_ids = tuple(visited_nodes)
    _validate_inputs(source_id, candidates, visited_ids, dist_matrix, communication_radius, config)
    visited = set(visited_ids)
    weights: dict[int, float] = {}
    for candidate_id in candidates:
        # A node is not its own next-hop neighbor.  It is omitted exactly like
        # any other candidate outside N_i, with probability zero.
        if candidate_id == source_id or candidate_id in visited:
            continue
        if dist_matrix[source_id][candidate_id] > communication_radius:
            continue
        tau = _non_negative_finite(pheromones, candidate_id, "pheromone")
        eta = _non_negative_finite(heuristic_values, candidate_id, "eta")
        weight = tau**config.alpha * eta**config.beta
        if not math.isfinite(weight) or weight < 0:
            raise HeuristicInputError("Eq. (26) transition weight is invalid")
        weights[candidate_id] = weight

    denominator = sum(weights.values())
    if not math.isfinite(denominator) or denominator <= 0:
        raise NoValidTransitionError("Eq. (26) has no positive transition denominator")
    return {candidate_id: weight / denominator for candidate_id, weight in weights.items()}


def _validate_inputs(
    source_id: int,
    candidate_ids: Iterable[int],
    visited_nodes: Iterable[int],
    dist_matrix: Sequence[Sequence[float]],
    communication_radius: float,
    config: MRPConfig,
) -> None:
    candidates = tuple(candidate_ids)
    node_count = len(dist_matrix)
    if not _is_valid_phase2_sensor_id(source_id, node_count):
        raise HeuristicInputError("transition source must be a valid sensor ID")
    if len(set(candidates)) != len(candidates):
        raise HeuristicInputError("transition candidates must be unique")
    if any(candidate_id == -1 for candidate_id in candidates):
        raise HeuristicInputError("Sink transition handling is not defined by Eq. (26)–(29)")
    if any(not _is_valid_phase2_sensor_id(candidate_id, node_count) for candidate_id in candidates):
        raise HeuristicInputError("transition candidates must be valid sensor IDs")
    if any(not _is_valid_phase2_sensor_id(node_id, node_count) for node_id in visited_nodes):
        raise HeuristicInputError("visited nodes must be valid sensor IDs")
    if len(dist_matrix) == 0 or any(len(row) != node_count for row in dist_matrix):
        raise HeuristicInputError("distance matrix must be square over sensor nodes")
    _positive_finite(communication_radius, "communication radius")
    _positive_finite(config.alpha, "alpha")
    _positive_finite(config.beta, "beta")
    for candidate_id in candidates:
        if candidate_id == source_id:
            continue
        _positive_finite(dist_matrix[source_id][candidate_id], "d_ij")


def _is_valid_phase2_sensor_id(sensor_id: int, node_count: int) -> bool:
    """Keep the shared sensor-ID range contract, excluding bool aliases."""

    return not isinstance(sensor_id, bool) and is_valid_sensor_id(sensor_id, node_count)


def _non_negative_finite(values: Mapping[int, float], sensor_id: int, label: str) -> float:
    if sensor_id not in values:
        raise HeuristicInputError(f"missing {label} for valid transition candidate {sensor_id}")
    value = values[sensor_id]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise HeuristicInputError(f"{label} must be finite and non-negative")
    return value


def _positive_finite(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise HeuristicInputError(f"{label} must be finite and positive")
