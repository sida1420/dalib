"""Immutable experiment-only contracts for parameter calibration."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .config import ExperimentRoundScenario


class ParameterProvenance(str, Enum):
    PAPER_EQUATION = "PAPER_EQUATION"
    PAPER_ALGORITHM_RULE = "PAPER_ALGORITHM_RULE"
    PAPER_SIMULATION_PARAMETER = "PAPER_SIMULATION_PARAMETER"
    EXISTING_SIMULATOR_PARAMETER = "EXISTING_SIMULATOR_PARAMETER"
    SIMULATOR_IMPLEMENTATION_MAPPING = "SIMULATOR_IMPLEMENTATION_MAPPING"
    SIMULATOR_COMPATIBILITY_ASSUMPTION = "SIMULATOR_COMPATIBILITY_ASSUMPTION"
    PAPER_DEFINED_VALUE_UNREPORTED = "PAPER_DEFINED_VALUE_UNREPORTED"
    USER_SELECTED_PARAMETER = "USER_SELECTED_PARAMETER"
    USER_SELECTED_CALIBRATION_RANGE = "USER_SELECTED_CALIBRATION_RANGE"
    USER_SELECTED_CALIBRATION_CONSTRAINT = "USER_SELECTED_CALIBRATION_CONSTRAINT"
    USER_SELECTED_SIMULATION_POLICY = "USER_SELECTED_SIMULATION_POLICY"
    USER_SELECTED_CALIBRATION_BUDGET = "USER_SELECTED_CALIBRATION_BUDGET"


@dataclass(frozen=True)
class ParameterAuditEntry:
    name: str
    value: object | None
    provenance: ParameterProvenance
    calibratable: bool
    reason: str


@dataclass(frozen=True)
class CalibrationPlan:
    calibration_seed_set: tuple[int, ...]
    evaluation_seed_set: tuple[int, ...]
    calibration_scenario_set_id: str
    evaluation_scenario_set_id: str
    calibration_scenarios: tuple[ExperimentRoundScenario, ...]
    evaluation_scenarios: tuple[ExperimentRoundScenario, ...]
    search_space: Mapping[str, tuple[object, ...]]
    search_space_provenance: Mapping[str, str]
    minimum_completion_rate: float | None = None
    approved_candidate_overrides: tuple[tuple[tuple[str, object], ...], ...] | None = None


@dataclass(frozen=True)
class CalibrationCandidate:
    config_id: str
    overrides: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class CalibrationRoundRecord:
    config_id: str
    trial_seed: int
    round_index: int
    success: bool
    failure_stage: str | None
    failure_reason: str | None
    phase2_status: str | None
    route_discovery_success: bool
    multipath_ready: bool | None
    unique_route_count: int | None
    sants_requested: int | None
    successful_sants: int | None
    mean_ant_hops: float | None
    negative_pheromone_failure: bool
    negative_pheromone_ant_index: int | None
    domain_failure: bool
    topology_conflict: bool
    no_route: bool
    minimum_observed_tau: float | None
    actual_round_energy: float | None
    delivered_payload_count: int | None
    energy_per_delivered_payload: float | None
    scenario_id: str | None = None


@dataclass(frozen=True)
class CalibrationTrialResult:
    config_id: str
    trial_seed: int
    completed: bool
    route_discovery_attempts: int
    route_discovery_successes: int
    multipath_ready_count: int
    negative_pheromone_failures: int
    domain_failure_count: int
    topology_conflicts: int
    no_route_count: int
    minimum_observed_tau: float | None
    normalized_energy: float | None
    failure_reasons: tuple[str, ...]
    scenario_id: str | None = None


@dataclass(frozen=True)
class CalibrationSummary:
    config_id: str
    attempted_trials: int
    completed_trials: int
    failed_trial_count: int
    completion_rate: float
    route_discovery_success_rate: float | None
    negative_pheromone_failure_count: int
    negative_pheromone_failure_rate: float | None
    domain_failure_count: int
    topology_conflict_count: int
    topology_conflict_rate: float | None
    no_route_count: int
    no_route_rate: float | None
    multipath_ready_rate: float | None
    mean_unique_route_count: float | None
    normalized_energy_mean: float | None
    normalized_energy_stddev: float | None
    normalized_energy_sample_count: int
    minimum_observed_tau: float | None
    mean_sants_requested: float | None
    mean_successful_sants: float | None
    mean_ant_hops: float | None
    valid: bool
    invalid_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationCandidateResult:
    candidate: CalibrationCandidate
    rounds: tuple[CalibrationRoundRecord, ...]
    trials: tuple[CalibrationTrialResult, ...]
    summary: CalibrationSummary
    rank: int | None


@dataclass(frozen=True)
class CalibrationResult:
    plan: CalibrationPlan
    parameter_audit: tuple[ParameterAuditEntry, ...]
    candidates: tuple[CalibrationCandidateResult, ...]
    selection_rule: str
    selected_config_id: str | None
    scenario_content_overlap: bool


@dataclass(frozen=True)
class FrozenCalibrationConfig:
    selected_config_id: str
    candidate: CalibrationCandidate
    parameter_snapshot: Mapping[str, object]
    parameter_metadata: Mapping[str, Mapping[str, object]]
    calibrated_parameter_paths: tuple[str, ...]
    calibration_seed_set: tuple[int, ...]
    calibration_scenario_set_id: str
    calibration_scenarios: tuple[ExperimentRoundScenario, ...]
    selection_rule: str
    calibration_summary: CalibrationSummary
    implementation_version: str | None
    created_at_utc: str
    calibration_status: str = "CALIBRATION_FRAMEWORK_SMOKE_ONLY"
    calibration_metadata: Mapping[str, object] | None = None
