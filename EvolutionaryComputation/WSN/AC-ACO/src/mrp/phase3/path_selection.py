"""MRP Phase-III Eq. (38)--(40), plus an optional Hybrid-only project adapter."""
from collections.abc import Mapping
from dataclasses import dataclass
import math

from mrp.config import MRPConfig
from mrp.phase2.route_quality import RouteQuality
from mrp.phase2.routing import MRPPhase2Result
class MRPPhase3InputError(ValueError):
    """Raised when Phase-II output cannot support paper-defined Phase III."""
@dataclass(frozen=True)
class MRPPhase3Candidate:
    """One exact unique Phase-II path with its Phase-III decision values."""

    route_quality: RouteQuality
    phase3_fitness: float
    probability: float

    @property
    def route(self) -> tuple[int, ...]:
        return self.route_quality.path

    @property
    def minimum_energy(self) -> float:
        return self.route_quality.f1

    @property
    def communication_energy(self) -> float:
        return self.route_quality.f2

    @property
    def path_length(self) -> float:
        return self.route_quality.f3
@dataclass(frozen=True)
class MRPPhase3Result:
    """One probabilistically selected route; Phase III does not transmit it."""

    candidates: tuple[MRPPhase3Candidate, ...]
    selected_route: tuple[int, ...]
    selected_index: int
    random_draw: float | None
    probability_sum: float
    source_phase2_multipath_ready: bool


def select_phase3_route(
    phase2_result: MRPPhase2Result,
    config: MRPConfig,
    rng: object,
    *,
    top_k: int | None = None,
) -> MRPPhase3Result:
    """Apply Eq. (38)--(40) to unique Phase-II paths and sample one route.

    Eq. (38)'s ``Emin(i)``, ``E(i)``, and ``Length_i`` reuse, respectively,
    the already validated Phase-II ``RouteQuality.f1``, ``.f2``, and ``.f3``.
    They are never recalculated here.

    ``top_k`` is a project adaptation activated only by AC-ACO + MRP Hybrid.
    Original/Pure MRP callers omit it and evaluate every discovered route.
    """

    _validate_phase2_result(phase2_result)
    _validate_weights(config)
    _validate_rng(rng)
    _validate_top_k(top_k)
    qualities = _unique_route_qualities(phase2_result)
    fitnesses = tuple(calculate_phase3_fitness(quality, config) for quality in qualities)
    if top_k:
        ranked = sorted(
            zip(qualities, fitnesses), key=lambda item: item[1], reverse=True,
        )[:top_k]
        qualities = tuple(item[0] for item in ranked)
        fitnesses = tuple(item[1] for item in ranked)
    probabilities, probability_sum = _probabilities(fitnesses)
    candidates = tuple(
        MRPPhase3Candidate(quality, fitness, probability)
        for quality, fitness, probability in zip(qualities, fitnesses, probabilities)
    )
    selected_index, draw = _select_index(probabilities, rng)
    return MRPPhase3Result(
        candidates,
        candidates[selected_index].route,
        selected_index,
        draw,
        probability_sum,
        phase2_result.multipath_ready,
    )


def calculate_phase3_fitness(route_quality: RouteQuality, config: MRPConfig) -> float:
    """Calculate only Phase-III Eq. (38), not Phase-II Eq. (33) quality."""

    _validate_weights(config)
    _validate_route_quality(route_quality)
    try:
        fitness = (
            route_quality.f1 ** config.k10
            + 1 / (route_quality.f2 ** config.k11)
            + 1 / (route_quality.f3 ** config.k12)
        )
    except (OverflowError, ZeroDivisionError) as error:
        raise MRPPhase3InputError("Phase-III fitness calculation overflowed") from error
    _require_positive_finite(fitness, "Phase-III fitness")
    return fitness


def _validate_phase2_result(result: MRPPhase2Result) -> None:
    if not isinstance(result, MRPPhase2Result):
        raise MRPPhase3InputError("Phase III requires an MRPPhase2Result")
    if result.status != "completed":
        raise MRPPhase3InputError(f"Phase II did not complete: {result.status}")
    if not result.unique_routes:
        raise MRPPhase3InputError("Phase II produced no unique successful routes")


def _unique_route_qualities(result: MRPPhase2Result) -> tuple[RouteQuality, ...]:
    try:
        routes = tuple(result.unique_routes)
        if len(set(routes)) != len(routes):
            raise MRPPhase3InputError("Phase-II unique routes must not contain duplicates")
    except TypeError as error:
        raise MRPPhase3InputError("Phase-II unique routes must be exact path tuples") from error
    by_route: dict[tuple[int, ...], RouteQuality] = {}
    for ant in result.successful_ant_results:
        quality = ant.route_quality
        if quality is None or ant.sant_result.path != quality.path:
            raise MRPPhase3InputError("successful ant route lacks matching RouteQuality")
        _validate_route_quality(quality)
        by_route.setdefault(quality.path, quality)
    try:
        qualities = tuple(by_route[path] for path in routes)
    except KeyError as error:
        raise MRPPhase3InputError("unique route lacks a successful RouteQuality") from error
    return qualities


def _probabilities(fitnesses: tuple[float, ...]) -> tuple[tuple[float, ...], float]:
    try:
        total = math.fsum(fitnesses)
    except OverflowError as error:
        raise MRPPhase3InputError("Phase-III total fitness overflowed") from error
    _require_positive_finite(total, "Phase-III total fitness")
    probabilities = tuple(fitness / total for fitness in fitnesses)
    if any(not math.isfinite(probability) or probability <= 0 for probability in probabilities):
        raise MRPPhase3InputError("Phase-III probabilities must be finite and positive")
    probability_sum = math.fsum(probabilities)
    if not math.isclose(probability_sum, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise MRPPhase3InputError("Phase-III probabilities do not sum to one")
    return probabilities, probability_sum


def _select_index(probabilities: tuple[float, ...], rng: object) -> tuple[int, float | None]:
    if len(probabilities) == 1:
        return 0, None
    draw = _draw_unit_interval(rng)
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index, draw
    return len(probabilities) - 1, draw


def _validate_weights(config: MRPConfig) -> None:
    if not isinstance(config, MRPConfig):
        raise MRPPhase3InputError("Phase III requires an MRPConfig")
    weights = (config.k10, config.k11, config.k12)
    if any(not _is_finite_number(weight) or weight < 0 for weight in weights):
        raise MRPPhase3InputError("k10, k11, and k12 must be finite non-negative weights")
    try:
        weights_sum = math.fsum(weights)
    except OverflowError as error:
        raise MRPPhase3InputError("k10 + k11 + k12 must be finite") from error
    if not math.isclose(weights_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise MRPPhase3InputError("k10 + k11 + k12 must equal one")


def _validate_route_quality(quality: RouteQuality) -> None:
    if not isinstance(quality, RouteQuality):
        raise MRPPhase3InputError("Phase-III candidate requires RouteQuality")
    path = quality.path
    if not isinstance(path, tuple) or len(path) < 2 or path[-1] != -1 or len(set(path[:-1])) != len(path) - 1:
        raise MRPPhase3InputError("RouteQuality path must be a complete acyclic Sink route")
    if any(not isinstance(node_id, int) or isinstance(node_id, bool) or node_id < 0 for node_id in path[:-1]):
        raise MRPPhase3InputError("RouteQuality path must contain sensor IDs before Sink")
    for label, value in (("minimum energy", quality.f1), ("communication energy", quality.f2), ("path length", quality.f3), ("route quality", quality.quality)):
        _require_positive_finite(value, label)


def _validate_rng(rng: object) -> None:
    if not callable(getattr(rng, "random", None)):
        raise MRPPhase3InputError("Phase III requires a caller-provided RNG with random()")


def _validate_top_k(top_k: int | None) -> None:
    if top_k is not None and (
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0
    ):
        raise MRPPhase3InputError("top_k must be None or a non-negative integer")


def _draw_unit_interval(rng: object) -> float:
    draw = rng.random()
    if not _is_finite_number(draw) or not 0 <= draw < 1:
        raise MRPPhase3InputError("RNG draw must be finite and in [0, 1)")
    return float(draw)


def _require_positive_finite(value: float, label: str) -> None:
    if not _is_finite_number(value) or value <= 0:
        raise MRPPhase3InputError(f"{label} must be finite and positive")


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
