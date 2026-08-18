"""Explicit network-wide multi-CH adaptation of event-oriented MRP Phase I."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

from .config import MRPConfig
from .topology_validation import is_valid_sensor_id


NETWORK_WIDE_MRP_ADAPTATION = "NETWORK_WIDE_MRP_ADAPTATION"
NETWORK_WIDE_SELECTION_POLICY = (
    "COVERAGE_FIRST_THEN_EQ24_NON_EVENT_FACTORS_THEN_SENSOR_ID"
)


class NetworkWideMRPClusteringError(ValueError):
    """Raised when exactly N valid network-wide MRP CHs cannot be selected."""


@dataclass(frozen=True)
class NetworkWideClusterFormationResult:
    """Deterministic multi-CH result; this is not paper event Phase I."""

    selected_cluster_heads: tuple[int, ...]
    target_cluster_head_count: int
    scores: Mapping[int, float]
    live_neighbor_counts: Mapping[int, int]
    selection_policy: str = NETWORK_WIDE_SELECTION_POLICY
    classification: str = NETWORK_WIDE_MRP_ADAPTATION


def configured_target_cluster_head_count(node_count: int, ch_proportion: float) -> int:
    """Match AC-ACO's ``while len(path) < proportion * node_count`` count."""

    if node_count <= 0 or not math.isfinite(ch_proportion) or not 0 < ch_proportion <= 1:
        raise NetworkWideMRPClusteringError("invalid configured CH proportion")
    return math.ceil(ch_proportion * node_count)


def select_network_wide_cluster_heads(
    nodes: Sequence[object], live_nodes: Collection[int], residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]], communication_radius: float,
    config: MRPConfig, target_cluster_head_count: int,
) -> NetworkWideClusterFormationResult:
    """Select N CHs without fabricating the event-signal term from Eq. (24).

    Paper-backed components retained: residual energy exponent ``k1`` and live
    neighbor-count exponent ``k2``.  Experimental adaptation: network-wide
    coverage is ranked first, the partial score second, and sensor ID breaks
    ties.  The event-only ``SE_i ** k3`` term is deliberately unavailable.
    """

    node_count = len(nodes)
    live = set(live_nodes)
    _validate_inputs(
        node_count, live, residual_e, dist_matrix, communication_radius,
        target_cluster_head_count,
    )
    neighbor_counts = {
        sensor: sum(
            neighbor != sensor and dist_matrix[sensor][neighbor] <= communication_radius
            for neighbor in live
        )
        for sensor in live
    }
    scores = {
        sensor: residual_e[sensor] ** config.k1 * count ** config.k2
        for sensor, count in neighbor_counts.items()
    }
    candidates = {sensor for sensor, score in scores.items() if math.isfinite(score) and score > 0}
    if len(candidates) < target_cluster_head_count:
        raise NetworkWideMRPClusteringError(
            f"MRP_CLUSTERING_EXHAUSTED: need {target_cluster_head_count} positive-score live CHs; "
            f"found {len(candidates)}"
        )

    footprints = {
        sensor: {other for other in live if other == sensor or dist_matrix[sensor][other] <= communication_radius}
        for sensor in candidates
    }
    uncovered = set(live)
    selected = []
    for _ in range(target_cluster_head_count):
        winner = max(
            candidates - set(selected),
            key=lambda sensor: (len(footprints[sensor] & uncovered), scores[sensor], -sensor),
        )
        selected.append(winner)
        uncovered -= footprints[winner]
    if uncovered:
        raise NetworkWideMRPClusteringError(
            f"MRP_CLUSTERING_EXHAUSTED: {len(uncovered)} live sensors cannot be assigned "
            f"within radius using exactly {target_cluster_head_count} CHs"
        )
    return NetworkWideClusterFormationResult(
        tuple(selected), target_cluster_head_count, MappingProxyType(scores),
        MappingProxyType(neighbor_counts),
    )


def _validate_inputs(node_count, live, residual_e, dist_matrix, radius, target):
    if not live or len(residual_e) != node_count:
        raise NetworkWideMRPClusteringError("network-wide MRP requires live residual state")
    if any(not is_valid_sensor_id(sensor, node_count) or residual_e[sensor] <= 0 for sensor in live):
        raise NetworkWideMRPClusteringError("live CH candidates must be valid positive-energy sensors")
    if len(dist_matrix) != node_count or any(len(row) != node_count for row in dist_matrix):
        raise NetworkWideMRPClusteringError("network-wide MRP requires a square distance matrix")
    if not math.isfinite(radius) or radius <= 0:
        raise NetworkWideMRPClusteringError("communication radius must be finite and positive")
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        raise NetworkWideMRPClusteringError("target CH count must be a positive integer")
