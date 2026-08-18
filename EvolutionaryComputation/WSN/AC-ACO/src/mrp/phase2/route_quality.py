"""MRP Phase-II BANT route metrics, Eq. (33)--(36), without feedback."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
import math

from evaluate import E_data_receiving, E_transmitting
from mrp.config import MRPConfig
from mrp.topology_validation import is_valid_sensor_id

class RouteQualityInputError(ValueError): pass

@dataclass(frozen=True)
class RouteQuality:
    path: tuple[int, ...]
    f1: float
    f2: float
    f3: float
    quality: float


@dataclass(frozen=True)
class BANTPreparation:
    forward_path: tuple[int, ...]
    reverse_path: tuple[int, ...]
    route_quality: RouteQuality


def evaluate_route_quality(
    path: Sequence[int],
    live_nodes: Collection[int],
    residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]],
    base_dists: Sequence[float],
    config: MRPConfig,
    c0: float,
    e_elec: float,
    free_space_coeff: float,
    multipath_coeff: float,
    d0: float,
    bit_count: float,
) -> RouteQuality:
    """Evaluate one completed route with Eq. (33)--(36), without state mutation.

    Eq. (35)'s ``E_ij`` is the paper's link communication energy: the
    existing transmitter and receiver primitives are composed for every route
    edge.  The terminal Sink edge uses ``base_dists`` and needs no ``E_sink``.
    Eq. (34) is mapped to physical sensors only because this simulator's Sink
    sentinel has no residual-energy state.
    """

    return _evaluate_route_quality(
        path, live_nodes, residual_e, dist_matrix, base_dists, config, c0,
        e_elec, free_space_coeff, multipath_coeff, d0, bit_count, False,
    )


def _evaluate_route_quality_from_validated_snapshot(
    path, live_nodes, residual_e, dist_matrix, base_dists, config, c0,
    e_elec, free_space_coeff, multipath_coeff, d0, bit_count,
) -> RouteQuality:
    """Internal hot path after Phase-II validated the shared snapshot once."""

    return _evaluate_route_quality(
        path, live_nodes, residual_e, dist_matrix, base_dists, config, c0,
        e_elec, free_space_coeff, multipath_coeff, d0, bit_count, True,
    )


def _evaluate_route_quality(
    path, live_nodes, residual_e, dist_matrix, base_dists, config, c0,
    e_elec, free_space_coeff, multipath_coeff, d0, bit_count, snapshot_validated,
) -> RouteQuality:
    route = _validate_route(
        path, live_nodes, residual_e, dist_matrix, base_dists,
        snapshot_validated=snapshot_validated,
    )
    _validate_parameters(config, c0, e_elec, free_space_coeff, multipath_coeff, d0, bit_count)
    f1 = min(residual_e[sensor_id] for sensor_id in route[:-1])
    _require_positive_finite(f1, "f1")
    distances = [_link_distance(source, destination, dist_matrix, base_dists) for source, destination in zip(route, route[1:])]
    f3 = sum(distances)
    _require_positive_finite(f3, "f3")
    f2 = sum(
        _link_communication_energy(
            distance, e_elec, free_space_coeff, multipath_coeff, d0, bit_count
        )
        for distance in distances
    )
    _require_positive_finite(f2, "f2")
    try:
        numerator = c0 * f1**config.k7
        denominator = f2**config.k8 * f3**config.k9
    except OverflowError as error:
        raise RouteQualityInputError("route quality calculation overflowed") from error
    _require_positive_finite(numerator, "Eq. (33) numerator")
    _require_positive_finite(denominator, "Eq. (33) denominator")
    quality = numerator / denominator
    _require_positive_finite(quality, "route quality")
    return RouteQuality(route, f1, f2, f3, quality)


def prepare_bant(route_quality: RouteQuality) -> BANTPreparation:
    if not isinstance(route_quality, RouteQuality):
        raise RouteQualityInputError("BANT preparation requires a RouteQuality result")
    path = route_quality.path
    if len(path) < 2 or path[-1] != -1 or len(set(path[:-1])) != len(path) - 1:
        raise RouteQualityInputError("BANT preparation requires a complete acyclic Sink route")
    if any(not isinstance(sensor_id, int) or isinstance(sensor_id, bool) or sensor_id < 0 for sensor_id in path[:-1]):
        raise RouteQualityInputError("BANT preparation requires sensor IDs before the Sink")
    for label, value in (("f1", route_quality.f1), ("f2", route_quality.f2), ("f3", route_quality.f3), ("route quality", route_quality.quality)):
        _require_positive_finite(value, label)
    return BANTPreparation(route_quality.path, tuple(reversed(route_quality.path)), route_quality)


def _validate_route(
    path: Sequence[int],
    live_nodes: Collection[int],
    residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]],
    base_dists: Sequence[float],
    *,
    snapshot_validated: bool = False,
) -> tuple[int, ...]:
    try:
        route = tuple(path)
        live_set = set(live_nodes)
        node_count = len(residual_e)
    except (TypeError, ValueError) as error:
        raise RouteQualityInputError("route and snapshot inputs must be collections") from error
    if len(route) < 2 or route[-1] != -1:
        raise RouteQualityInputError("route must be non-empty, complete, and end at Sink -1")
    if any(not _is_sensor_id(sensor_id, node_count) for sensor_id in route[:-1]):
        raise RouteQualityInputError("route must start with and otherwise contain valid sensor IDs")
    if len(set(route[:-1])) != len(route) - 1:
        raise RouteQualityInputError("route contains a duplicate physical sensor or cycle")
    if not live_set or any(not _is_sensor_id(sensor_id, node_count) for sensor_id in live_set):
        raise RouteQualityInputError("live_nodes must contain valid sensor IDs")
    if not snapshot_validated:
        _validate_snapshot(residual_e, dist_matrix, base_dists)
        if any(residual_e[sensor_id] <= 0 for sensor_id in live_set):
            raise RouteQualityInputError("live_nodes cannot contain a depleted sensor")
    if any(sensor_id not in live_set for sensor_id in route[:-1]):
        raise RouteQualityInputError("route contains a dead sensor")
    return route


def _validate_snapshot(
    residual_e: Sequence[float], dist_matrix: Sequence[Sequence[float]], base_dists: Sequence[float]
) -> None:
    node_count = len(residual_e)
    try:
        matrix_is_square = len(dist_matrix) == node_count and all(len(row) == node_count for row in dist_matrix)
        base_distances_match = len(base_dists) == node_count
    except TypeError as error:
        raise RouteQualityInputError("distance inputs must be sized collections") from error
    if not matrix_is_square:
        raise RouteQualityInputError("distance matrix must be square over sensors")
    if not base_distances_match:
        raise RouteQualityInputError("base distances must match residual energy")
    for energy in residual_e:
        _require_finite_non_negative(energy, "residual energy")
    for distance in base_dists:
        _require_finite_non_negative(distance, "base distance")
    for row in dist_matrix:
        for distance in row:
            _require_finite_non_negative(distance, "distance matrix value")


def _validate_parameters(
    config: MRPConfig, c0: float, e_elec: float, free_space_coeff: float,
    multipath_coeff: float, d0: float, bit_count: float,
) -> None:
    if not isinstance(config, MRPConfig):
        raise RouteQualityInputError("route quality requires an MRPConfig")
    _require_positive_finite(config.k7, "k7")
    _require_positive_finite(config.k8, "k8")
    _require_positive_finite(config.k9, "k9")
    _require_positive_finite(c0, "c0")
    _require_finite_non_negative(e_elec, "E_elec")
    _require_finite_non_negative(free_space_coeff, "free-space coefficient")
    _require_finite_non_negative(multipath_coeff, "multipath coefficient")
    _require_positive_finite(d0, "d0")
    _require_positive_finite(bit_count, "bit count")


def _link_distance(
    source_id: int, destination_id: int, dist_matrix: Sequence[Sequence[float]], base_dists: Sequence[float]
) -> float:
    distance = base_dists[source_id] if destination_id == -1 else dist_matrix[source_id][destination_id]
    _require_finite_non_negative(distance, "route link distance")
    return distance


def _link_communication_energy(
    distance: float, e_elec: float, free_space_coeff: float, multipath_coeff: float,
    d0: float, bit_count: float,
) -> float:
    try:
        transmit = E_transmitting(e_elec, free_space_coeff, multipath_coeff, distance, d0, bit_count)
        receive = E_data_receiving(e_elec, bit_count)
    except OverflowError as error:
        raise RouteQualityInputError("route link communication energy overflowed") from error
    energy = transmit + receive
    _require_finite_non_negative(energy, "route link communication energy")
    return energy


def _is_sensor_id(sensor_id: int, node_count: int) -> bool:
    return not isinstance(sensor_id, bool) and is_valid_sensor_id(sensor_id, node_count)


def _require_positive_finite(value: float, label: str) -> None:
    _require_finite_non_negative(value, label)
    if value == 0:
        raise RouteQualityInputError(f"{label} must be positive")


def _require_finite_non_negative(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RouteQualityInputError(f"{label} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise RouteQualityInputError(f"{label} must be finite and non-negative")
