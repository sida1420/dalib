"""Machine-readable raw calibration output, isolated from final-evaluation output."""

from dataclasses import asdict
from pathlib import Path

from .calibration_freeze import freeze_selected_configuration
from .calibration_types import CalibrationResult
from .config import ExperimentConfig
from .io import csv_text, json_text, publish_output_files


def write_calibration_outputs(
    result: CalibrationResult, base_config: ExperimentConfig, output_directory: str | Path,
    *, frozen_payload: object | None = None, extra_files: dict[str, str] | None = None,
) -> Path:
    """Atomically write calibration records; never overwrite an experiment result directory."""

    files = calibration_output_files(result, base_config, frozen_payload)
    files.update(extra_files or {})
    return publish_output_files(output_directory, files)


def calibration_output_files(
    result: CalibrationResult, base_config: ExperimentConfig, frozen_payload: object | None = None,
) -> dict[str, str]:
    """Serialize one calibration result without deciding where it is published."""

    rounds = [record for candidate in result.candidates for record in candidate.rounds]
    trials = [record for candidate in result.candidates for record in candidate.trials]
    summaries = [candidate.summary for candidate in result.candidates]
    selected = _selected_payload(result, base_config) if frozen_payload is None else frozen_payload
    return {
        "calibration_manifest.json": json_text({
            "plan": result.plan,
            "parameter_provenance_audit": result.parameter_audit,
            "selection_rule": result.selection_rule,
            "hard_validity_gates": [
                "zero_valid_trials", "invalid_negative_pheromone_state", "topology_conflict",
                "invalid_parameter_or_metric_domain", "caller_supplied_minimum_completion_rate",
            ],
            "scenario_content_overlap": result.scenario_content_overlap,
        }),
        "candidate_configs.json": json_text([candidate.candidate for candidate in result.candidates]),
        "calibration_rounds.csv": csv_text(_csv_record(record) for record in rounds),
        "calibration_trials.csv": csv_text(_csv_record(record) for record in trials),
        "calibration_failures.csv": csv_text(_csv_record(record) for record in rounds if not record.success),
        "calibration_summary.csv": csv_text(_csv_record(record) for record in summaries),
        "selected_config.json": json_text(selected),
    }


def _selected_payload(result, base_config):
    if result.selected_config_id is None:
        return {"selected_config_id": None, "reason": "no valid calibration candidate"}
    return freeze_selected_configuration(result, base_config)


def _csv_record(record) -> dict[str, object]:
    return {name: _csv_value(value) for name, value in asdict(record).items()}


def _csv_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return json_text(value)
