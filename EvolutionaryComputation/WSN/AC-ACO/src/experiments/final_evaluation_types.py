"""Immutable experiment-only records for the frozen Checkpoint-10 evaluation."""

from dataclasses import dataclass

from .config import ExperimentMode, TrialSeedStreams
from .config import ExperimentConfig
from .scenario_generation import ExistingMap, FrozenScenarioSets
from .types import ModeTrialResult


@dataclass(frozen=True)
class FNDObservation:
    """First-node-death result without treating a censoring cap as a death."""

    round_index_zero_based: int | None
    round_number_one_based: int | None
    censored_at_cap: bool
    rounds_executed: int
    algorithm_failure_before_fnd: bool


@dataclass(frozen=True)
class NativeEvaluationTrial:
    """One native observation: seed-only or seed plus frozen event scenario."""

    algorithm: ExperimentMode
    trial_seed: int
    scenario_id: str | None
    scenario_anchor_sensor: int | None
    seed_streams: TrialSeedStreams
    mode_result: ModeTrialResult
    fnd: FNDObservation


@dataclass(frozen=True)
class FinalEvaluationPreflight:
    """Validated immutable boundary before reserved observations may execute."""

    selected_config_id: str
    selected_config_hash: str
    evaluation_scenario_hash: str
    calibration_scenario_hash: str
    max_rounds: int
    baseline_scenario_independent: bool
    hybrid_scenario_independent: bool
    multipath_denominator_audit: str


@dataclass(frozen=True)
class FinalEvaluationContext:
    """Validated inputs retained through execution and publication; never reloaded."""

    preflight: FinalEvaluationPreflight
    configuration: ExperimentConfig
    existing_map: ExistingMap
    scenarios: FrozenScenarioSets


@dataclass(frozen=True)
class FinalEvaluationResult:
    """Raw native records plus the preflight provenance that authorized them."""

    context: FinalEvaluationContext
    native_trials: tuple[NativeEvaluationTrial, ...]
