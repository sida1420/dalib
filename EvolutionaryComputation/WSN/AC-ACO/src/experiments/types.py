"""Experiment records; unavailable values remain ``None`` rather than zero."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .config import EvaluationView, ExperimentMode, TrialSeedStreams


@dataclass(frozen=True)
class RoundMetrics:
    algorithm: ExperimentMode
    trial_id: int
    trial_seed: int
    round_index: int
    live_before: tuple[int, ...]
    live_after: tuple[int, ...]
    dead_count: int
    newly_dead: tuple[int, ...]
    declared_source_sensors: tuple[int, ...] | None
    represented_payload_sensor_count: int | None
    delivered_payload_count: int | None
    delivered_bits: float | None
    selected_cluster_heads: tuple[int, ...] | None
    selected_routes: tuple[tuple[int, ...], ...] | None
    cluster_sizes: Mapping[int, int] | None
    actual_round_energy: float | None
    energy_per_delivered_payload: float | None
    energy_per_delivered_bit: float | None
    energy_per_represented_sensor: float | None
    cumulative_actual_energy: float
    residual_total: float
    residual_mean: float
    residual_std_population: float
    residual_min: float
    selected_route_hops: tuple[int, ...] | None
    selected_route_lengths: tuple[float, ...] | None
    discovered_route_counts: tuple[int, ...] | None
    multipath_ready: tuple[bool, ...] | None
    success: bool
    failure_stage: str | None
    failure_reason: str | None


@dataclass(frozen=True)
class LifetimeMetrics:
    fnd_round: int | None
    hnd_round: int | None
    lnd_round: int | None
    fnd_censored: bool
    hnd_censored: bool
    lnd_censored: bool


@dataclass(frozen=True)
class ModeTrialResult:
    algorithm: ExperimentMode
    rounds: tuple[RoundMetrics, ...]
    raw_round_results: tuple[object, ...]
    lifetime: LifetimeMetrics


@dataclass(frozen=True)
class FairnessAudit:
    raw_energy_status: str
    normalized_energy_status: str
    lifetime_status: str
    reason: str
    reporting_view: EvaluationView


@dataclass(frozen=True)
class ComparisonTrialResult:
    trial_id: int
    seeds: TrialSeedStreams
    mode_results: Mapping[ExperimentMode, ModeTrialResult]
    fairness: FairnessAudit


@dataclass(frozen=True)
class ExperimentResult:
    trials: tuple[ComparisonTrialResult, ...]
    aggregate: Mapping[str, Mapping[str, object]]
