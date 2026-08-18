"""Checkpoint-9C recalibration after the Phase-I semantic correction."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess

from .io import json_text, publish_output_files
from .real_calibration import (
    CALIBRATION_SEEDS,
    CALIBRATION_STATUS,
    EVALUATION_SEEDS,
    RealCalibrationRun,
    run_real_calibration,
    write_real_calibration_outputs,
)
from .scenario_generation import load_existing_map_network, load_frozen_scenario_sets


CURRENT_SEMANTICS = "CURRENT_POST_PHASE1_FIX"
IMPLEMENTATION_VERSION = "checkpoint-9c-current-post-phase1-fix"
SEMANTIC_FIX_MARKERS = (
    "DEAD_NODE_PHASE1_SCORING_FIX",
    "LONE_SURVIVOR_PHASE1_FIX",
    "PURE_MRP_STATE_VALIDATION_HARDENING",
    "IMMUTABLE_PHEROMONE_SNAPSHOTS",
    "BOOLEAN_SENSOR_ID_REJECTION",
)
HISTORICAL_CONFIG_ID = "62963da9e200e409"
HISTORICAL_CONFIG_HASH = "b6e3ef93ce5de68ce5c4f997454f0cd3d4ef662c71e51fd583e3bc0a48128473"
EXPECTED_SCENARIO_HASHES = {
    "calibration-v1": "f36ee2d98db9264896b2bdc5a686d07b7fac35bd4d81dfac9e8ee8bf6b177cec",
    "evaluation-v1": "9c602cf7b7e7024da49cf9fc22f36866d8673a4a32cbc56fe0c47e3130fdf8c9",
}
class RecalibrationError(ValueError):
    """Raised when Checkpoint 9C cannot preserve its controlled boundaries."""
def run_checkpoint_9c(
    repository_root: str | Path, output_root: str | Path = "real_calibration_v2",
) -> Path:
    """Run the approved calibration policy on current code without evaluation execution."""

    root = Path(repository_root).resolve()
    destination = _destination(root, output_root)
    old_root = root / "real_calibration"
    historical = _read_json(old_root / "calibration_outputs" / "selected_config.json")
    scenarios = load_frozen_scenario_sets(old_root / "scenario_manifest.json")
    _validate_historical_inputs(historical, scenarios)
    existing_map = load_existing_map_network(root / "map.pkl")
    identity = _implementation_identity(root)
    run = run_real_calibration(existing_map, scenarios)
    run = replace(run, configuration=replace(run.configuration, implementation_version=IMPLEMENTATION_VERSION))
    lock = _lock_payload(run, identity)
    report = _report_payload(historical, old_root, run, lock)
    source_manifest = (old_root / "scenario_manifest.json").read_text(encoding="utf-8")
    publish_output_files(destination, {"scenario_manifest.json": source_manifest})
    output = write_real_calibration_outputs(
        run, destination,
        metadata_updates={
            "implementation_semantics": CURRENT_SEMANTICS,
            "semantic_fix_markers": SEMANTIC_FIX_MARKERS,
            "implementation_identity": identity,
            "recalibration_reason": "IMPLEMENTATION_SEMANTIC_CHANGE_REQUIRING_RECALIBRATION",
            "historical_calibration": {
                "status": "SUPERSEDED_BY_IMPLEMENTATION_SEMANTIC_CHANGE",
                "selected_config_id": HISTORICAL_CONFIG_ID,
                "selected_config_hash": HISTORICAL_CONFIG_HASH,
            },
            "evaluation_seeds_executed": False,
            "evaluation_scenarios_executed": False,
        },
        extra_files={
            "checkpoint_9c_lock.json": json_text(lock),
            "checkpoint_9c_report.json": json_text(report),
        },
    )
    return output
def load_checkpoint_9c_lock(calibration_output_directory: str | Path) -> dict[str, object]:
    """Read the only lock that may authorize a future Checkpoint-10 preflight."""

    lock = _read_json(Path(calibration_output_directory) / "checkpoint_9c_lock.json")
    required = {
        "parameter_config_id", "parameter_config_hash", "calibration_freeze_id", "calibration_freeze_hash", "scenario_hashes",
        "calibration_seeds", "reserved_evaluation_seeds", "implementation_semantics",
    }
    if not required <= set(lock) or lock.get("implementation_semantics") != CURRENT_SEMANTICS:
        raise RecalibrationError("Checkpoint-9C evaluation lock is invalid")
    if lock.get("evaluation_seeds_executed") or lock.get("evaluation_scenarios_executed"):
        raise RecalibrationError("Checkpoint-9C lock records forbidden evaluation execution")
    return lock
def _destination(root: Path, output_root: str | Path) -> Path:
    destination = Path(output_root)
    destination = destination if destination.is_absolute() else root / destination
    if destination.exists():
        raise RecalibrationError("Checkpoint-9C output directory already exists")
    return destination
def _validate_historical_inputs(historical, scenarios) -> None:
    metadata = historical.get("calibration_metadata", {})
    if historical.get("selected_config_id") != HISTORICAL_CONFIG_ID:
        raise RecalibrationError("historical selected configuration identity changed")
    if metadata.get("selected_config_hash") != HISTORICAL_CONFIG_HASH:
        raise RecalibrationError("historical selected configuration hash changed")
    if dict(scenarios.scenario_hashes) != EXPECTED_SCENARIO_HASHES:
        raise RecalibrationError("frozen scenario hashes changed")
    if tuple(metadata.get("calibration_seed_set", ())) != CALIBRATION_SEEDS:
        raise RecalibrationError("historical calibration seed set changed")
    if tuple(metadata.get("reserved_evaluation_seed_set", ())) != EVALUATION_SEEDS:
        raise RecalibrationError("historical reserved evaluation seed set changed")
    if metadata.get("evaluation_seeds_executed") or metadata.get("evaluation_scenarios_executed"):
        raise RecalibrationError("historical calibration reports forbidden evaluation execution")


def _implementation_identity(root: Path) -> dict[str, object]:
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_status": _git(root, "status", "--short"),
        "semantic_fix_markers": SEMANTIC_FIX_MARKERS,
    }


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ("git", *args), cwd=root, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _lock_payload(run: RealCalibrationRun, identity: dict[str, object]) -> dict[str, object]:
    return {
        "checkpoint": "9C — RECALIBRATION AFTER PHASE-I SEMANTIC FIX",
        "calibration_status": CALIBRATION_STATUS,
        "implementation_semantics": CURRENT_SEMANTICS,
        "implementation_identity": identity,
        "selected_config_id": run.stage_b.selected_config_id,
        "selected_config_hash": run.selected_config_hash,
        "scenario_hashes": EXPECTED_SCENARIO_HASHES,
        "calibration_seeds": CALIBRATION_SEEDS,
        "reserved_evaluation_seeds": EVALUATION_SEEDS,
        "evaluation_seeds_executed": False,
        "evaluation_scenarios_executed": False,
        "reserved_final_evaluation": {"primary_stop": "FND", "round_cap": 5000},
    }


def _report_payload(historical, old_root: Path, run: RealCalibrationRun, lock) -> dict[str, object]:
    old_metadata = _read_json(old_root / "calibration_outputs" / "real_calibration_metadata.json")
    old_selected = _selected_payload(historical)
    new_selected = _selected_payload(run.stage_b)
    return {
        "historical_calibration_status": "SUPERSEDED_BY_IMPLEMENTATION_SEMANTIC_CHANGE",
        "recalibration_reason": "IMPLEMENTATION_SEMANTIC_CHANGE_REQUIRING_RECALIBRATION",
        "old_characterization": old_metadata["characterization"],
        "new_characterization": asdict(run.characterization),
        "old_stage_a_top_2": _old_top_two(old_metadata["stage_a_ranked_summaries"]),
        "new_stage_a_top_2": _top_two(run.stage_a),
        "old_selected_config": old_selected,
        "new_selected_config": new_selected,
        "stage_a_candidate_count": len(run.stage_a.candidates),
        "stage_b_candidate_count": len(run.stage_b.candidates),
        "negative_tau_results": _failure_totals(run, "negative_pheromone_failure_count"),
        "routing_failure_results": _failure_totals(run, "no_route_count"),
        "evaluation_seeds_executed": False,
        "evaluation_scenarios_executed": False,
        "final_evaluation_lock": lock,
    }


def _old_top_two(items):
    return sorted((item for item in items if item.get("rank") is not None), key=lambda item: item["rank"])[:2]


def _top_two(stage):
    ranked = sorted((item for item in stage.candidates if item.rank is not None), key=lambda item: item.rank)
    return [_selected_payload(item) for item in ranked[:2]]


def _selected_payload(value) -> dict[str, object]:
    if hasattr(value, "candidates"):
        selected = next(item for item in value.candidates if item.rank == 1)
        return _selected_payload(selected)
    if hasattr(value, "candidate"):
        return {"config_id": value.candidate.config_id, "overrides": dict(value.candidate.overrides), "rank": value.rank}
    candidate = value["candidate"]
    return {"config_id": value["selected_config_id"], "overrides": dict(candidate["overrides"])}


def _failure_totals(run: RealCalibrationRun, field: str) -> dict[str, int]:
    return {
        "stage_a": sum(getattr(item.summary, field) for item in run.stage_a.candidates),
        "stage_b": sum(getattr(item.summary, field) for item in run.stage_b.candidates),
    }


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecalibrationError(f"cannot read required artifact: {path}") from error
