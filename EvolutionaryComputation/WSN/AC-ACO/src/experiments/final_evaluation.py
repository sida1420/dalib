"""Frozen, FND-censored final evaluation authorized by Checkpoint-9C only."""

import csv
from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path

from .calibration_freeze import FrozenCalibrationError, load_frozen_configuration
from .calibration_audit import parameter_snapshot
from .config import ExperimentConfig, ExperimentMode, StopCondition
from .final_evaluation_types import (
    FNDObservation, FinalEvaluationContext, FinalEvaluationPreflight, FinalEvaluationResult,
    NativeEvaluationTrial,
)
from .hop_counts import scenario_with_current_hops
from .real_calibration import EVALUATION_SEEDS, primary_calibration_config
from .recalibration import RecalibrationError, load_checkpoint_9c_lock
from .scientific_freeze import scientific_freeze_identity
from .checkpoint_9d import Checkpoint9DError, load_checkpoint_9d_freeze
from .runners import run_trial
from .scenario_generation import (
    ExistingMap, FrozenEventScenario, load_frozen_scenario_sets, validate_scenarios_for_existing_map,
)


FINAL_MAX_ROUNDS = 5000
EXPECTED_EVALUATION = (("evaluation-v1-e1", 90), ("evaluation-v1-e2", 72), ("evaluation-v1-e3", 20), ("evaluation-v1-e4", 3))


class FinalEvaluationError(ValueError):
    """Raised before reserved observations if a frozen-evaluation invariant fails."""


def load_final_evaluation_config(existing_map: ExistingMap, selected_config_path: str | Path) -> ExperimentConfig:
    """Load the frozen MRP snapshot and add only approved evaluation controls."""

    frozen = load_frozen_configuration(selected_config_path, primary_calibration_config(existing_map))
    return replace(
        frozen, max_rounds=FINAL_MAX_ROUNDS, trial_count=1,
        stop_condition=StopCondition.FND,
        parameter_provenance={
            **frozen.parameter_provenance,
            "experiment.max_rounds": "USER_SELECTED_EVALUATION_CAP / PAPER_SCALE_REFERENCED",
            "experiment.stop_condition": "PAPER_ALGORITHM_RULE: FIRST NODE DEATH",
        },
        implementation_version="checkpoint-10-frozen-evaluation",
    )


def preflight_final_evaluation(
    existing_map: ExistingMap, selected_config_path: str | Path, scenario_manifest_path: str | Path,
) -> FinalEvaluationPreflight:
    """Verify all immutable boundaries without executing an evaluation seed."""

    return _validated_context(existing_map, selected_config_path, scenario_manifest_path).preflight


def run_frozen_final_evaluation(
    existing_map: ExistingMap, selected_config_path: str | Path, scenario_manifest_path: str | Path,
) -> FinalEvaluationResult:
    """Execute only evaluation-v1 × reserved seeds, stopping each native trial at FND or the cap."""

    context = _validated_context(existing_map, selected_config_path, scenario_manifest_path)
    preflight, scenarios, config = context.preflight, context.scenarios, context.configuration
    pure = replace(config, modes=(ExperimentMode.PURE_MRP,))
    baseline = replace(config, modes=(ExperimentMode.BASELINE,))
    hybrid = replace(config, modes=(ExperimentMode.HYBRID,))
    if _mrp_hash(pure) != _mrp_hash(hybrid):
        raise FinalEvaluationError("Pure-MRP and Hybrid must use the identical frozen MRP configuration")

    native = []
    reference = scenarios.evaluation[0].round_scenario()
    refresh_hops = partial(_refresh_hops, communication_radius=config.mrp_parameters.communication_radius)
    for trial_id, seed in enumerate(EVALUATION_SEEDS):
        native.append(_run_native(existing_map, baseline, reference, seed, trial_id, None))
        native.append(_run_native(existing_map, hybrid, reference, seed, trial_id, None, refresh_hops))
        for scenario in scenarios.evaluation:
            native.append(_run_native(
                existing_map, pure, scenario.round_scenario(), seed, trial_id, scenario, refresh_hops,
            ))
    result = FinalEvaluationResult(context, tuple(native))
    validate_final_evaluation_result(result)
    return result


def run_and_write_frozen_final_evaluation(
    existing_map: ExistingMap, selected_config_path: str | Path, scenario_manifest_path: str | Path,
    output_directory: str | Path,
) -> Path:
    """Run and persist one native trial at a time; never retain 180 histories."""

    from .final_evaluation_reporting import IncrementalFinalEvaluationWriter

    context = _validated_context(existing_map, selected_config_path, scenario_manifest_path)
    config, scenarios = context.configuration, context.scenarios
    pure = replace(config, modes=(ExperimentMode.PURE_MRP,))
    baseline = replace(config, modes=(ExperimentMode.BASELINE,))
    hybrid = replace(config, modes=(ExperimentMode.HYBRID,))
    if _mrp_hash(pure) != _mrp_hash(hybrid):
        raise FinalEvaluationError("Pure-MRP and Hybrid must use the identical frozen MRP configuration")
    reference = scenarios.evaluation[0].round_scenario()
    refresh_hops = partial(_refresh_hops, communication_radius=config.mrp_parameters.communication_radius)
    writer = IncrementalFinalEvaluationWriter(context, output_directory)
    try:
        for trial_id, seed in enumerate(EVALUATION_SEEDS):
            writer.append(_run_native(existing_map, baseline, reference, seed, trial_id, None))
            writer.append(_run_native(existing_map, hybrid, reference, seed, trial_id, None, refresh_hops))
            for scenario in scenarios.evaluation:
                writer.append(_run_native(existing_map, pure, scenario.round_scenario(), seed, trial_id, scenario, refresh_hops))
        return writer.finalize()
    except BaseException:
        writer.abort()
        raise


def _run_native(existing_map, config, scenario, seed, trial_id, frozen_scenario, provider=None):
    comparison = run_trial(
        existing_map.network, (scenario,), config, trial_id, seed,
        repeat_scenarios=True, scenario_provider=provider,
    )
    mode = config.modes[0]
    result = comparison.mode_results[mode]
    return NativeEvaluationTrial(
        mode, seed, None if frozen_scenario is None else frozen_scenario.scenario_id,
        None if frozen_scenario is None else frozen_scenario.event_anchor_sensor,
        comparison.seeds, result, _fnd(result, config.max_rounds),
    )


def _fnd(result, max_rounds):
    index = result.lifetime.fnd_round
    failed = any(not row.success for row in result.rounds)
    censored = index is None and not failed and len(result.rounds) == max_rounds
    return FNDObservation(index, None if index is None else index + 1, censored, len(result.rounds), failed)


def _refresh_hops(scenario, state, network, communication_radius):
    physical = getattr(state, "physical_state", state)
    refreshed, _ = scenario_with_current_hops(
        scenario, network, physical.live_nodes, communication_radius,
    )
    return refreshed


def _load_selected(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalEvaluationError("frozen selected configuration cannot be read") from error


def _validate_selected(selected, config, lock):
    metadata = selected.get("calibration_metadata")
    expected_id, expected_hash = lock["parameter_config_id"], lock["parameter_config_hash"]
    if selected.get("selected_config_id") != expected_id:
        raise FinalEvaluationError("CURRENT semantics config required; historical calibration is stale")
    if selected.get("calibration_status") != "SCIENTIFIC_CALIBRATION_COMPLETE":
        raise FinalEvaluationError("FROZEN ARTIFACT MISMATCH: selected configuration status")
    digest = hashlib.sha256(json.dumps(selected.get("parameter_snapshot"), sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    if not isinstance(metadata, dict) or metadata.get("selected_config_hash") != expected_hash or digest != expected_hash:
        raise FinalEvaluationError("FROZEN ARTIFACT MISMATCH: selected configuration hash")
    freeze_payload = metadata.get("scientific_freeze_payload")
    if metadata.get("implementation_semantics") != "CURRENT_POST_PHASE1_FIX" or not isinstance(freeze_payload, dict):
        raise FinalEvaluationError("CURRENT semantics scientific freeze required")
    freeze_id, freeze_hash = scientific_freeze_identity(freeze_payload)
    if (freeze_id, freeze_hash) != (lock.get("calibration_freeze_id"), lock.get("calibration_freeze_hash")):
        raise FinalEvaluationError("FROZEN ARTIFACT MISMATCH: scientific freeze identity")
    if set(metadata.get("calibration_seed_set", ())) & set(EVALUATION_SEEDS) or tuple(metadata.get("reserved_evaluation_seed_set", ())) != EVALUATION_SEEDS:
        raise FinalEvaluationError("FROZEN ARTIFACT MISMATCH: reserved evaluation seeds")
    if metadata.get("evaluation_seeds_executed") or metadata.get("evaluation_scenarios_executed"):
        raise FinalEvaluationError("frozen calibration metadata reports prior evaluation execution")
    if config.max_rounds != FINAL_MAX_ROUNDS or config.stop_condition is not StopCondition.FND:
        raise FinalEvaluationError("final evaluation requires FND stop with the approved 5000-round cap")
    actual_snapshot = parameter_snapshot(config)
    actual_digest = hashlib.sha256(json.dumps(actual_snapshot, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    if actual_digest != expected_hash:
        raise FinalEvaluationError("FROZEN ARTIFACT MISMATCH: restored parameter snapshot")


def _validate_scenarios(scenarios, lock):
    if dict(scenarios.scenario_hashes) != lock["scenario_hashes"]:
        raise FinalEvaluationError("FROZEN ARTIFACT MISMATCH: scenario hashes")
    actual = tuple((item.scenario_id, item.event_anchor_sensor) for item in scenarios.evaluation)
    if actual != EXPECTED_EVALUATION:
        raise FinalEvaluationError("FROZEN ARTIFACT MISMATCH: reserved evaluation scenarios")


def _audit_calibration_multipath(path, config_id):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["config_id"] == config_id]
        attempts = [row for row in rows if row["phase2_status"]]
        routes = [int(row["unique_route_count"]) for row in rows if row["unique_route_count"]]
        ready = [row for row in attempts if row["multipath_ready"] == "True"]
    except (OSError, KeyError, ValueError) as error:
        raise FinalEvaluationError("cannot audit frozen calibration multipath denominators") from error
    if not attempts or len(routes) != len(attempts):
        raise FinalEvaluationError("calibration multipath denominator audit failed")
    return f"unique-route mean: {len(routes)} Phase-II records; multipath-ready: {len(ready)}/{len(attempts)} Phase-II attempts"


def _mrp_hash(config):
    snapshot = parameter_snapshot(config)
    included = {key: value for key, value in snapshot.items() if key.startswith("mrp.") or key == "mrp_pheromone_lifecycle"}
    return hashlib.sha256(json.dumps(included, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def validate_final_evaluation_result(result: FinalEvaluationResult) -> FinalEvaluationResult:
    """Bind publication to the exact preflight context and native observation plan."""

    if not isinstance(result, FinalEvaluationResult):
        raise FinalEvaluationError("final evaluation result is required")
    context = result.context
    preflight = context.preflight
    if not preflight.selected_config_id or not preflight.selected_config_hash:
        raise FinalEvaluationError("final result does not retain the approved selected configuration")
    _validate_scenarios(context.scenarios, {
        "scenario_hashes": {
            "calibration-v1": preflight.calibration_scenario_hash,
            "evaluation-v1": preflight.evaluation_scenario_hash,
        },
    })
    actual_digest = hashlib.sha256(json.dumps(parameter_snapshot(context.configuration), sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    if actual_digest != preflight.selected_config_hash:
        raise FinalEvaluationError("final result configuration no longer matches its frozen snapshot")
    if context.configuration.max_rounds != FINAL_MAX_ROUNDS or context.configuration.stop_condition is not StopCondition.FND:
        raise FinalEvaluationError("final result does not retain the approved FND/cap policy")
    expected = {
        (ExperimentMode.BASELINE, seed, None) for seed in EVALUATION_SEEDS
    } | {
        (ExperimentMode.HYBRID, seed, None) for seed in EVALUATION_SEEDS
    } | {
        (ExperimentMode.PURE_MRP, seed, scenario_id) for seed in EVALUATION_SEEDS
        for scenario_id, _ in EXPECTED_EVALUATION
    }
    actual = {(item.algorithm, item.trial_seed, item.scenario_id) for item in result.native_trials}
    if actual != expected or len(result.native_trials) != len(expected):
        raise FinalEvaluationError("final result does not contain the required 30/30/120 native observations")
    if any(item.trial_seed not in EVALUATION_SEEDS for item in result.native_trials):
        raise FinalEvaluationError("final result contains an invalid trial seed")
    return result


def _validated_context(existing_map, selected_config_path, scenario_manifest_path):
    selected = _load_selected(selected_config_path)
    try:
        lock = load_checkpoint_9c_lock(Path(selected_config_path).parent)
    except RecalibrationError as error:
        raise FinalEvaluationError("CURRENT semantics config required; historical calibration is stale") from error
    scenarios = load_frozen_scenario_sets(scenario_manifest_path)
    validate_scenarios_for_existing_map(scenarios, existing_map)
    config = load_final_evaluation_config(existing_map, selected_config_path)
    _validate_selected(selected, config, lock)
    _validate_scenarios(scenarios, lock)
    try:
        load_checkpoint_9d_freeze(Path(selected_config_path).resolve().parents[2], selected_config_path, scenario_manifest_path)
    except Checkpoint9DError as error:
        raise FinalEvaluationError(
            "Checkpoint-9D scientific freeze is stale; a reviewed network-wide MRP freeze is required"
        ) from error
    audit = _audit_calibration_multipath(Path(selected_config_path).parent / "calibration_rounds.csv", selected["selected_config_id"])
    identical_hops = len({tuple(sorted(item.initial_hop_counts.items())) for item in scenarios.evaluation}) == 1
    if not identical_hops:
        raise FinalEvaluationError("Hybrid scenario-independence audit failed: evaluation hop snapshots differ")
    preflight = FinalEvaluationPreflight(
        selected["selected_config_id"], selected["calibration_metadata"]["selected_config_hash"],
        scenarios.scenario_hashes["evaluation-v1"], scenarios.scenario_hashes["calibration-v1"],
        config.max_rounds, True, True, audit,
    )
    return FinalEvaluationContext(preflight, config, existing_map, scenarios)
