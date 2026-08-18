"""MRP_PHASE_I centralized dynamic-clustering core from Eq. (24) and (25)."""

from collections.abc import Collection, Mapping, Sequence
import math

from mrp.config import MRPConfig
from mrp.topology_validation import is_valid_sensor_id
from mrp.types import ClusterFormationResult


class ClusterFormationInputError(ValueError):
    """Raised when caller-supplied event information cannot support Phase I."""


def calculate_cluster_head_score(
    residual_energy: float,
    event_neighbor_count: int,
    event_signal_strength: float,
    config: MRPConfig,
) -> float:
    """MRP_PHASE_I: calculate Eq. (24), q_i = E_i^k1 K_i^k2 SE_i^k3."""

    _require_positive_finite(residual_energy, "residual energy")
    _require_positive_finite(event_signal_strength, "event signal strength")
    if not isinstance(event_neighbor_count, int) or event_neighbor_count < 0:
        raise ClusterFormationInputError("event neighbor count must be a non-negative integer")

    score = (
        residual_energy**config.k1
        * event_neighbor_count**config.k2
        * event_signal_strength**config.k3
    )
    if not math.isfinite(score) or score < 0:
        raise ClusterFormationInputError("Eq. (24) produced an invalid q_i")
    return score


def calculate_cluster_head_timer(score: float, timer_scale_q: float) -> float:
    """MRP_PHASE_I: calculate Eq. (25) only when caller supplies paper's q."""

    _require_positive_finite(score, "q_i score")
    _require_positive_finite(timer_scale_q, "timer scale q")
    return timer_scale_q / score


def select_cluster_head(
    nodes: Sequence[object],
    live_nodes: Collection[int],
    residual_e: Sequence[float],
    event_nodes: Sequence[int],
    event_signal_strengths: Mapping[int, float],
    dist_matrix: Sequence[Sequence[float]],
    communication_radius: float,
    config: MRPConfig,
) -> ClusterFormationResult:
    """MRP_PHASE_I: choose one CH for one caller-defined event area.

    ``event_nodes`` is S_e and signals are caller-owned SE_i input.  The
    distributed T_a interval is represented by direct snapshot counting.  With
    a common positive Eq. (25) coefficient, the unique largest positive q_i
    has the shortest timer and is therefore the sole advertised CH.  Ties and
    all-zero scores are rejected because the paper provides no tie rule.  A
    lone survivor of a previously multi-sensor event is selected directly.
    """

    node_count = len(nodes)
    event_ids = tuple(event_nodes)
    live_ids = set(live_nodes)
    _validate_inputs(node_count, residual_e, event_ids, live_ids, dist_matrix, communication_radius)

    candidates = tuple(
        sensor_id
        for sensor_id in event_ids
        if sensor_id in live_ids and residual_e[sensor_id] > 0
    )
    if not candidates:
        raise ClusterFormationInputError("event area has no live CH candidate")

    neighbor_counts = {
        sensor_id: _count_event_neighbors(
            sensor_id, candidates, dist_matrix, communication_radius
        )
        for sensor_id in candidates
    }
    scores: dict[int, float] = {}
    for sensor_id in candidates:
        if sensor_id not in event_signal_strengths:
            raise ClusterFormationInputError(
                f"missing event signal strength for event candidate {sensor_id}"
            )
        scores[sensor_id] = calculate_cluster_head_score(
            residual_e[sensor_id],
            neighbor_counts[sensor_id],
            event_signal_strengths[sensor_id],
            config,
        )

    positive_scores = {sensor_id: score for sensor_id, score in scores.items() if score > 0}
    if not positive_scores and len(candidates) == 1 and len(event_ids) > 1:
        winner = candidates[0]
        return ClusterFormationResult.create(
            winner, (), scores, neighbor_counts
        )
    if not positive_scores:
        raise ClusterFormationInputError("no event candidate has a finite positive q_i")
    highest_score = max(positive_scores.values())
    winners = [
        sensor_id for sensor_id, score in positive_scores.items() if score == highest_score
    ]
    if len(winners) != 1:
        raise ClusterFormationInputError("Eq. (25) timer tie has no paper-defined resolution")

    winner = winners[0]
    _require_winner_heard_by_all(winner, candidates, dist_matrix, communication_radius)
    members = tuple(sensor_id for sensor_id in candidates if sensor_id != winner)
    return ClusterFormationResult.create(winner, members, scores, neighbor_counts)


def _count_event_neighbors(
    sensor_id: int,
    event_ids: Sequence[int],
    dist_matrix: Sequence[Sequence[float]],
    communication_radius: float,
) -> int:
    return sum(
        1
        for neighbor_id in event_ids
        if neighbor_id != sensor_id
        and dist_matrix[sensor_id][neighbor_id] <= communication_radius
    )


def _require_winner_heard_by_all(
    winner: int,
    candidates: Sequence[int],
    dist_matrix: Sequence[Sequence[float]],
    communication_radius: float,
) -> None:
    """Require the hearing condition behind paper's one-CH event-area claim.

    The paper does not give a separate centralized tie/advertisement model.
    Global q_i ranking is equivalent to Algorithm 1 only if the earliest CH
    advertisement reaches every later live event candidate.  This direct
    winner-to-candidate radio check is the simulator mapping available from
    the current snapshot.  If it fails, a later candidate could advertise and
    the paper's one-CH observable result is not guaranteed.
    """

    if any(
        dist_matrix[winner][candidate_id] > communication_radius
        for candidate_id in candidates
        if candidate_id != winner
    ):
        raise ClusterFormationInputError(
            "event area cannot guarantee Algorithm 1's single CH advertisement"
        )


def _validate_inputs(
    node_count: int,
    residual_e: Sequence[float],
    event_ids: tuple[int, ...],
    live_ids: set[int],
    dist_matrix: Sequence[Sequence[float]],
    communication_radius: float,
) -> None:
    if len(residual_e) != node_count:
        raise ClusterFormationInputError("residual energy length must match nodes")
    if not event_ids or len(set(event_ids)) != len(event_ids):
        raise ClusterFormationInputError("event_nodes must be a non-empty unique sequence")
    if any(not is_valid_sensor_id(sensor_id, node_count) for sensor_id in event_ids):
        raise ClusterFormationInputError("event_nodes contains an invalid sensor ID")
    if any(not is_valid_sensor_id(sensor_id, node_count) for sensor_id in live_ids):
        raise ClusterFormationInputError("live_nodes contains an invalid sensor ID")
    if len(dist_matrix) != node_count or any(len(row) != node_count for row in dist_matrix):
        raise ClusterFormationInputError("dist_matrix must be square over nodes")
    _validate_event_distances(event_ids, dist_matrix)
    _require_positive_finite(communication_radius, "communication radius")
    for sensor_id in event_ids:
        if residual_e[sensor_id] > 0:
            _require_positive_finite(residual_e[sensor_id], "residual energy")


def _validate_event_distances(
    event_ids: Sequence[int], dist_matrix: Sequence[Sequence[float]]
) -> None:
    """Fail closed when Phase I cannot evaluate an event-area radio link."""

    for source_id in event_ids:
        for target_id in event_ids:
            if source_id == target_id:
                continue
            distance = dist_matrix[source_id][target_id]
            if (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not math.isfinite(distance)
                or distance < 0
            ):
                raise ClusterFormationInputError(
                    "event distance must be finite and non-negative"
                )


def _require_positive_finite(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClusterFormationInputError(f"{label} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ClusterFormationInputError(f"{label} must be finite and positive")
