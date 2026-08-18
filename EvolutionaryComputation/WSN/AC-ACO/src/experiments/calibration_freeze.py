"""Freeze and reload a selected calibration result without post-hoc parameter overrides."""

from datetime import datetime, timezone
import json
from pathlib import Path

from mrp import PheromoneLifecycle

from .calibration_audit import parameter_snapshot
from .calibration_config import (
    CalibrationInputError, apply_parameter_snapshot, candidate_configuration, candidate_from_overrides,
)
from .calibration_types import CalibrationResult, FrozenCalibrationConfig
from .config import ExperimentConfig


class FrozenCalibrationError(ValueError):
    """Raised when a final evaluation attempts to modify a calibrated configuration."""


def freeze_selected_configuration(
    result: CalibrationResult, base_config: ExperimentConfig, *,
    calibration_status: str = "CALIBRATION_FRAMEWORK_SMOKE_ONLY",
    calibration_metadata: dict[str, object] | None = None,
) -> FrozenCalibrationConfig:
    """Create the only artifact later evaluation may use without re-tuning."""

    if result.selected_config_id is None:
        raise FrozenCalibrationError("calibration produced no valid selected configuration")
    selected = next(item for item in result.candidates if item.candidate.config_id == result.selected_config_id)
    selected_config = candidate_configuration(base_config, selected.candidate, len(result.plan.calibration_seed_set))
    calibrated = tuple(path for path, _ in selected.candidate.overrides)
    snapshot = parameter_snapshot(selected_config)
    metadata = {
        entry.name: {
            "value": snapshot.get(entry.name, entry.value),
            "provenance": entry.provenance.value,
            "calibrated": entry.name in calibrated,
        }
        for entry in result.parameter_audit
    }
    return FrozenCalibrationConfig(
        selected.candidate.config_id, selected.candidate, snapshot, metadata, calibrated,
        result.plan.calibration_seed_set, result.plan.calibration_scenario_set_id,
        result.plan.calibration_scenarios, result.selection_rule, selected.summary,
        base_config.implementation_version, datetime.now(timezone.utc).isoformat(),
        calibration_status, calibration_metadata,
    )


def load_frozen_configuration(
    selected_config_path: str | Path, base_config: ExperimentConfig,
    runtime_overrides: dict[str, object] | None = None,
) -> ExperimentConfig:
    """Restore a serialized selected config and reject any calibrated-path override."""

    payload = json.loads(Path(selected_config_path).read_text(encoding="utf-8"))
    try:
        selected_id = payload["selected_config_id"]
        overrides = tuple((name, _restore_value(name, value)) for name, value in payload["candidate"]["overrides"])
        candidate = candidate_from_overrides(overrides)
        if candidate.config_id != selected_id:
            raise FrozenCalibrationError("selected configuration ID does not match its canonical overrides")
        calibrated = set(payload["calibrated_parameter_paths"])
        conflict = set(runtime_overrides or ()) & calibrated
        if conflict:
            raise FrozenCalibrationError(f"frozen calibrated parameter override rejected: {sorted(conflict)}")
        if runtime_overrides:
            raise FrozenCalibrationError("frozen configuration accepts no runtime parameter overrides")
        return apply_parameter_snapshot(base_config, payload["parameter_snapshot"])
    except (KeyError, TypeError, CalibrationInputError) as error:
        raise FrozenCalibrationError("selected configuration artifact is invalid") from error


def _restore_value(path: str, value: object) -> object:
    return PheromoneLifecycle(value) if path == "mrp_pheromone_lifecycle" else value
