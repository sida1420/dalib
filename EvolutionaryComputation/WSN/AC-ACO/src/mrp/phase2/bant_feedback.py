"""MRP Phase-II BANT Eq. (30), (32), and (37), without orchestration."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

from mrp.config import MRPConfig
from mrp.phase2.pheromone import (
    PheromoneInputError,
    update_pheromone_with_deposit,
    validate_pheromone_config,
)
from mrp.phase2.route_quality import BANTPreparation, RouteQualityInputError, prepare_bant


class BANTFeedbackInputError(ValueError):
    """Raised when a BANT feedback snapshot lacks paper-required inputs."""


@dataclass(frozen=True)
class BANTLinkFeedbackTrace:
    """One reverse BANT traversal step mapped to its directed forward link."""

    reverse_link: tuple[int, int]
    forward_link: tuple[int, int]
    tau_before: float
    global_delta_tau: float
    tau_after: float


@dataclass(frozen=True)
class BANTFeedbackResult:
    """Read-only result of BANT sensor-link feedback; Sink pheromone is absent."""

    preparation: BANTPreparation
    previous_best_quality: float
    updated_best_quality: float
    global_delta_tau: float
    pheromone_state: dict[int, dict[int, float]]
    trace: tuple[BANTLinkFeedbackTrace, ...]
    skipped_terminal_link: tuple[int, int]
    negative_pheromone_links: tuple[tuple[int, int], ...]
    transition_compatible: bool


def apply_bant_global_feedback(
    preparation: BANTPreparation,
    previous_best_quality: float,
    pheromone_state: Mapping[int, Mapping[int, float]],
    config: MRPConfig,
    c: float,
    c1: float,
) -> BANTFeedbackResult:
    """Apply paper Eq. (32), then shared Eq. (30), then Eq. (37).

    The caller must supply a previous positive ``f_best`` because the paper
    does not state how its first BANT initializes that denominator.  Only
    sensor-to-sensor forward routing links are updated; the terminal link is
    returned as skipped because no Sink pheromone contract exists yet.
    """

    return _apply_bant_global_feedback(
        preparation, previous_best_quality, pheromone_state, config, c, c1, False,
    )


def _apply_bant_feedback_to_validated_state(
    preparation, previous_best_quality, pheromone_state, config, c, c1,
) -> BANTFeedbackResult:
    """Internal hot path after Phase-II validated the complete state once."""

    return _apply_bant_global_feedback(
        preparation, previous_best_quality, pheromone_state, config, c, c1, True,
    )


def _apply_bant_global_feedback(
    preparation, previous_best_quality, pheromone_state, config, c, c1,
    state_validated,
) -> BANTFeedbackResult:
    validated = _validated_preparation(preparation)
    _require_positive_finite(previous_best_quality, "previous best quality")
    _require_finite(c, "c")
    _require_finite(c1, "c1")
    if not isinstance(config, MRPConfig):
        raise BANTFeedbackInputError("BANT feedback requires an MRPConfig")
    try:
        validate_pheromone_config(config)
    except PheromoneInputError as error:
        raise BANTFeedbackInputError(str(error)) from error
    current_quality = validated.route_quality.quality
    delta_tau = calculate_global_pheromone_delta(current_quality, previous_best_quality, c, c1)
    state = _copy_sensor_pheromone_state(
        pheromone_state, validate_all=not state_validated,
    )
    sensor_links = tuple(zip(validated.forward_path[:-2], validated.forward_path[1:-1]))
    traces: list[BANTLinkFeedbackTrace] = []
    negative_links: list[tuple[int, int]] = []
    for source_id, destination_id in reversed(sensor_links):
        old_tau = _directed_tau(state, source_id, destination_id)
        try:
            new_tau = update_pheromone_with_deposit(old_tau, delta_tau, config)
        except PheromoneInputError as error:
            raise BANTFeedbackInputError(str(error)) from error
        state[source_id][destination_id] = new_tau
        if new_tau < 0:
            negative_links.append((source_id, destination_id))
        traces.append(BANTLinkFeedbackTrace(
            (destination_id, source_id), (source_id, destination_id), old_tau, delta_tau, new_tau
        ))
    return BANTFeedbackResult(
        validated,
        previous_best_quality,
        max(previous_best_quality, current_quality),
        delta_tau,
        state,
        tuple(traces),
        (validated.forward_path[-2], -1),
        tuple(negative_links),
        not negative_links,
    )


def calculate_global_pheromone_delta(
    current_quality: float, previous_best_quality: float, c: float, c1: float,
) -> float:
    """MRP_PHASE_II Eq. (32), preserving its paper-defined signed feedback."""

    _require_positive_finite(current_quality, "current route quality")
    _require_positive_finite(previous_best_quality, "previous best quality")
    _require_finite(c, "c")
    _require_finite(c1, "c1")
    delta_tau = c * (current_quality - previous_best_quality) / previous_best_quality + c1 * current_quality
    _require_finite(delta_tau, "Eq. (32) global delta tau")
    return delta_tau


def _validated_preparation(preparation: BANTPreparation) -> BANTPreparation:
    if not isinstance(preparation, BANTPreparation):
        raise BANTFeedbackInputError("BANT feedback requires a BANTPreparation")
    try:
        validated = prepare_bant(preparation.route_quality)
    except RouteQualityInputError as error:
        raise BANTFeedbackInputError(str(error)) from error
    if preparation != validated:
        raise BANTFeedbackInputError("BANT preparation does not match its validated route quality")
    return validated


def _copy_sensor_pheromone_state(
    pheromone_state: Mapping[int, Mapping[int, float]],
    *,
    validate_all: bool = True,
) -> dict[int, dict[int, float]]:
    if not isinstance(pheromone_state, Mapping):
        raise BANTFeedbackInputError("pheromone state must map directed sensor links")
    try:
        state = {source: dict(outgoing) for source, outgoing in pheromone_state.items()}
    except (AttributeError, TypeError, ValueError) as error:
        raise BANTFeedbackInputError("pheromone state must map directed sensor links") from error
    if not validate_all:
        return state
    if any(not _sensor_id(source) for source in state):
        raise BANTFeedbackInputError("pheromone state cannot contain Sink or invalid sources")
    if any(not _sensor_id(destination) for outgoing in state.values() for destination in outgoing):
        raise BANTFeedbackInputError("pheromone state cannot contain Sink or invalid destinations")
    if any(not _valid_tau(value) for outgoing in state.values() for value in outgoing.values()):
        raise BANTFeedbackInputError("pheromone state values must be finite and non-negative")
    return state


def _directed_tau(state: Mapping[int, Mapping[int, float]], source_id: int, destination_id: int) -> float:
    if source_id not in state or destination_id not in state[source_id]:
        raise BANTFeedbackInputError(f"missing directed pheromone for routing link {source_id}->{destination_id}")
    old_tau = state[source_id][destination_id]
    if not _valid_tau(old_tau):
        raise BANTFeedbackInputError("old pheromone must be finite and non-negative")
    return old_tau


def _sensor_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_tau(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _require_positive_finite(value: float, label: str) -> None:
    _require_finite(value, label)
    if value <= 0:
        raise BANTFeedbackInputError(f"{label} must be positive")


def _require_finite(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BANTFeedbackInputError(f"{label} must be finite")
