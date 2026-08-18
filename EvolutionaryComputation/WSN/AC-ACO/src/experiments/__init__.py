"""Reproducible, workload-aware comparison infrastructure for WSN modes."""

from .baseline_adapter import BaselineState, run_baseline_round
from .calibration import SELECTION_RULE, run_mrp_calibration, run_mrp_calibration_candidates
from .calibration_audit import SEARCH_RANGE_PROVENANCE, TUNABLE_PATHS, audit_calibration_parameters
from .calibration_config import (
    CalibrationInputError,
    apply_parameter_snapshot,
    candidate_configuration,
    enumerate_calibration_candidates,
    validate_calibration_plan,
)
from .calibration_freeze import (
    FrozenCalibrationError,
    freeze_selected_configuration,
    load_frozen_configuration,
)
from .calibration_io import write_calibration_outputs
from .calibration_characterization import CalibrationCharacterization, CharacterizationError, characterize_calibration_space
from .hop_counts import HopCountInputError, HopCountSnapshot, derive_hop_counts, scenario_with_current_hops
from .real_calibration import (
    CALIBRATION_SEEDS,
    CALIBRATION_STATUS,
    EVALUATION_SEEDS,
    RealCalibrationError,
    RealCalibrationRun,
    freeze_scenario_manifest,
    primary_calibration_config,
    run_real_calibration,
    write_real_calibration_outputs,
)
from .recalibration import (
    CURRENT_SEMANTICS,
    IMPLEMENTATION_VERSION,
    RecalibrationError,
    load_checkpoint_9c_lock,
    run_checkpoint_9c,
)
from .checkpoint_9d import (
    CHECKPOINT_9D_DIRECTORY, HYBRID_IMPLEMENTATION_ID, HYBRID_SEMANTICS,
    Checkpoint9DError, create_checkpoint_9d_freeze, load_checkpoint_9d_freeze,
)
from .final_evaluation import (
    FINAL_MAX_ROUNDS,
    FinalEvaluationError,
    load_final_evaluation_config,
    preflight_final_evaluation,
    run_frozen_final_evaluation,
    run_and_write_frozen_final_evaluation,
    validate_final_evaluation_result,
)
from .final_evaluation_reporting import write_final_evaluation_outputs
from .final_evaluation_types import (
    FNDObservation,
    FinalEvaluationPreflight,
    FinalEvaluationResult,
    FinalEvaluationContext,
    NativeEvaluationTrial,
)
from .scenario_generation import (
    CALIBRATION_QUANTILES,
    EVALUATION_QUANTILES,
    EVENT_RADIUS_METERS,
    ExistingMap,
    FrozenEventScenario,
    FrozenScenarioSets,
    ScenarioGenerationError,
    generate_frozen_scenario_sets,
    load_frozen_scenario_sets,
    load_existing_map_network,
    scenario_manifest_payload,
    validate_frozen_scenario_sets,
    validate_scenarios_for_existing_map,
)
from .calibration_types import (
    CalibrationCandidate,
    CalibrationCandidateResult,
    CalibrationPlan,
    CalibrationResult,
    CalibrationRoundRecord,
    CalibrationSummary,
    CalibrationTrialResult,
    FrozenCalibrationConfig,
    ParameterAuditEntry,
    ParameterProvenance,
)
from .config import (
    EvaluationView,
    ExperimentConfig,
    ExperimentMode,
    ExperimentNetwork,
    ExperimentRoundScenario,
    StopCondition,
    TrialSeedStreams,
    derive_seed_streams,
    parameter_manifest,
)
from .io import write_experiment_outputs
from .metrics import assess_trial_fairness, derive_lifetime_metrics
from .network_wide_mrp import NetworkWideMRPRoundResult, run_network_wide_mrp_round
from .runners import run_experiment, run_trial, smoke_trace
from .types import (
    ComparisonTrialResult,
    ExperimentResult,
    FairnessAudit,
    LifetimeMetrics,
    ModeTrialResult,
    RoundMetrics,
)

__all__ = [
    "BaselineState", "EvaluationView", "ExperimentConfig", "ExperimentMode",
    "ExperimentNetwork", "ExperimentRoundScenario", "StopCondition",
    "TrialSeedStreams", "assess_trial_fairness", "derive_lifetime_metrics",
    "derive_seed_streams", "run_baseline_round", "run_experiment", "run_trial",
    "parameter_manifest",
    "smoke_trace", "write_experiment_outputs", "ComparisonTrialResult",
    "ExperimentResult", "FairnessAudit", "LifetimeMetrics", "ModeTrialResult",
    "RoundMetrics",
    "NetworkWideMRPRoundResult", "run_network_wide_mrp_round",
    "CalibrationCandidate", "CalibrationCandidateResult", "CalibrationInputError",
    "CalibrationPlan", "CalibrationResult", "CalibrationRoundRecord", "CalibrationSummary",
    "CalibrationTrialResult", "FrozenCalibrationConfig", "FrozenCalibrationError",
    "ParameterAuditEntry", "ParameterProvenance", "SEARCH_RANGE_PROVENANCE", "SELECTION_RULE",
    "TUNABLE_PATHS", "apply_parameter_snapshot", "audit_calibration_parameters",
    "candidate_configuration", "enumerate_calibration_candidates", "freeze_selected_configuration",
    "load_frozen_configuration", "run_mrp_calibration", "validate_calibration_plan",
    "run_mrp_calibration_candidates", "write_calibration_outputs",
    "CalibrationCharacterization", "CharacterizationError", "characterize_calibration_space",
    "HopCountInputError", "HopCountSnapshot", "derive_hop_counts", "scenario_with_current_hops",
    "EVENT_RADIUS_METERS", "CALIBRATION_QUANTILES", "EVALUATION_QUANTILES", "ExistingMap",
    "FrozenEventScenario", "FrozenScenarioSets", "ScenarioGenerationError",
    "generate_frozen_scenario_sets", "load_frozen_scenario_sets", "load_existing_map_network", "scenario_manifest_payload", "validate_frozen_scenario_sets", "validate_scenarios_for_existing_map",
    "CALIBRATION_SEEDS", "EVALUATION_SEEDS", "CALIBRATION_STATUS", "RealCalibrationError",
    "RealCalibrationRun", "primary_calibration_config", "freeze_scenario_manifest",
    "run_real_calibration", "write_real_calibration_outputs",
    "CURRENT_SEMANTICS", "IMPLEMENTATION_VERSION", "RecalibrationError",
    "load_checkpoint_9c_lock", "run_checkpoint_9c",
    "CHECKPOINT_9D_DIRECTORY", "HYBRID_IMPLEMENTATION_ID", "HYBRID_SEMANTICS", "Checkpoint9DError", "create_checkpoint_9d_freeze", "load_checkpoint_9d_freeze",
    "FINAL_MAX_ROUNDS", "FinalEvaluationError", "load_final_evaluation_config",
    "preflight_final_evaluation", "run_frozen_final_evaluation", "run_and_write_frozen_final_evaluation", "validate_final_evaluation_result", "write_final_evaluation_outputs",
    "FNDObservation", "FinalEvaluationContext", "FinalEvaluationPreflight", "FinalEvaluationResult", "NativeEvaluationTrial",
]
