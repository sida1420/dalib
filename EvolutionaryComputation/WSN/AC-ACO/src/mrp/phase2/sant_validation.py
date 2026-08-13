"""MRP_PHASE_II normal-SANT input and sensor-neighbor validation."""

from collections.abc import Collection, Mapping, Sequence
import math

from mrp.config import MRPConfig
from mrp.phase2.heuristic import HeuristicBounds
from mrp.topology_validation import is_valid_sensor_id


class SANTInputError(ValueError):
    """Raised when SANT construction lacks a valid snapshot input."""


def validate_sant_route_inputs(
    start_ch: int,
    live_nodes: Collection[int],
    residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]],
    base_dists: Sequence[float],
    communication_radius: float,
    pheromone_state: Mapping[int, Mapping[int, float]],
    bounds: HeuristicBounds,
    config: MRPConfig,
    ttl: int,
    rng: object,
) -> set[int]:
    """Validate state shared by normal-SANT and AANT forwarding branches."""

    node_count = len(residual_e)
    live_set = set(live_nodes)
    if not _sensor_id(start_ch, node_count) or start_ch not in live_set:
        raise SANTInputError("start CH must be a live valid sensor ID")
    if not live_set or any(not _sensor_id(node_id, node_count) for node_id in live_set):
        raise SANTInputError("live_nodes must contain only valid sensor IDs")
    if len(dist_matrix) != node_count or any(len(row) != node_count for row in dist_matrix):
        raise SANTInputError("distance matrix must be square over sensor nodes")
    if len(base_dists) != node_count:
        raise SANTInputError("base distances must match residual energy")
    _positive_finite(communication_radius, "communication radius")
    if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl < 0:
        raise SANTInputError("TTL must be a caller-supplied non-negative integer")
    if not isinstance(pheromone_state, Mapping):
        raise SANTInputError("pheromone state must map source IDs to outgoing link values")
    if not isinstance(bounds, HeuristicBounds):
        raise SANTInputError("SANT construction requires caller-provided heuristic bounds")
    if not isinstance(config, MRPConfig):
        raise SANTInputError("SANT construction requires an MRPConfig")
    if not callable(getattr(rng, "random", None)):
        raise SANTInputError("SANT construction requires a caller-provided RNG with random()")
    for energy in residual_e:
        _non_negative_finite(energy, "residual energy")
    if any(residual_e[node_id] <= 0 for node_id in live_set):
        raise SANTInputError("live_nodes cannot contain a depleted sensor")
    return live_set


def sink_is_reachable(current_id: int, base_dists: Sequence[float], radius: float) -> bool:
    """SIMULATOR_COMPATIBILITY_ASSUMPTION: terminate at directly reachable Sink."""

    distance_to_sink = base_dists[current_id]
    _non_negative_finite(distance_to_sink, "current distance to Sink")
    return distance_to_sink <= radius


def eligible_sensor_neighbors(
    current_id: int,
    live_nodes: Collection[int],
    visited_nodes: Collection[int],
    dist_matrix: Sequence[Sequence[float]],
    radius: float,
) -> tuple[int, ...]:
    """Return only unvisited live sensor neighbors; Sink is never a candidate."""

    visited_set = set(visited_nodes)
    candidates: list[int] = []
    for candidate_id in sorted(live_nodes):
        if candidate_id == current_id or candidate_id in visited_set:
            continue
        distance = dist_matrix[current_id][candidate_id]
        _positive_finite(distance, "d_ij")
        if distance <= radius:
            candidates.append(candidate_id)
    return tuple(candidates)


def _sensor_id(sensor_id: int, node_count: int) -> bool:
    return not isinstance(sensor_id, bool) and is_valid_sensor_id(sensor_id, node_count)


def _positive_finite(value: float, label: str) -> None:
    _non_negative_finite(value, label)
    if value == 0:
        raise SANTInputError(f"{label} must be positive for a sensor-to-sensor link")


def _non_negative_finite(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SANTInputError(f"{label} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise SANTInputError(f"{label} must be finite and non-negative")


def draw_unit_interval(rng: object) -> float:
    """Read one reproducible random draw, rejecting invalid RNG implementations."""

    draw = rng.random()
    if isinstance(draw, bool) or not isinstance(draw, (int, float)) or not math.isfinite(draw):
        raise SANTInputError("RNG random() result must be finite")
    if not 0 <= draw < 1:
        raise SANTInputError("RNG random() result must be in [0, 1)")
    return draw
