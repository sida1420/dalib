"""Caller-directed raw JSON/CSV output; no files are written beneath ``src``."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
import csv
from io import StringIO
import json
import math
from pathlib import Path
import shutil
import tempfile

from .config import ExperimentConfig, ExperimentNetwork, ExperimentRoundScenario, parameter_manifest
from .types import ExperimentResult


def write_experiment_outputs(
    result: ExperimentResult, config: ExperimentConfig, network: ExperimentNetwork,
    scenarios: Sequence[ExperimentRoundScenario], output_directory: str | Path,
) -> Path:
    """Atomically publish strict-JSON raw results into a new caller-supplied directory."""

    files = {
        "manifest.json": json_text({
            "config": config,
            "parameter_provenance": parameter_manifest(config),
            "network_provenance": {
                "map_id": "USER_SELECTED_PARAMETER",
                "nodes": "EXISTING_SIMULATOR_PARAMETER",
                "dist_matrix": "EXISTING_SIMULATOR_PARAMETER",
                "base_dists": "EXISTING_SIMULATOR_PARAMETER",
                "initial_residual_e": "EXISTING_SIMULATOR_PARAMETER",
            },
            "network": network,
            "scenarios": tuple(scenarios),
            "trial_seeds": tuple(trial.seeds for trial in result.trials),
        }),
    }
    rows = [row for trial in result.trials for mode in trial.mode_results.values() for row in mode.rounds]
    files["round_metrics.csv"] = csv_text(_row_dict(row) for row in rows)
    files["trial_metrics.json"] = json_text({
        "trials": [_trial_summary(trial) for trial in result.trials],
        "aggregate": result.aggregate,
    })
    failures = [row for row in rows if not row.success]
    files["failures.csv"] = csv_text(_row_dict(row) for row in failures)
    return publish_output_files(output_directory, files)


def publish_output_files(output_directory: str | Path, files: Mapping[str, str]) -> Path:
    """Atomically publish already-serialized caller-directed experiment files."""

    directory = Path(output_directory)
    if directory.exists():
        raise FileExistsError("experiment output directory must not already exist")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        for name, content in files.items():
            (temporary / name).write_text(content, encoding="utf-8", newline="")
        temporary.replace(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return directory


def json_text(value: object) -> str:
    return json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)


def csv_text(rows) -> str:
    materialized = list(rows)
    fieldnames = list(materialized[0]) if materialized else ["algorithm", "trial_id", "round_index", "failure_reason"]
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(materialized)
    return output.getvalue()


def _row_dict(record: object) -> dict[str, object]:
    return {key: _csv_value(value) for key, value in asdict(record).items()}


def _trial_summary(trial) -> dict[str, object]:
    return {
        "trial_id": trial.trial_id,
        "seeds": trial.seeds,
        "fairness": trial.fairness,
        "modes": {
            mode.value: {"rounds": mode_result.rounds, "lifetime": mode_result.lifetime}
            for mode, mode_result in trial.mode_results.items()
        },
    }


def _csv_value(value: object) -> object:
    if isinstance(value, (str, int, float)) or value is None:
        return _jsonable(value)
    return json.dumps(_jsonable(value), sort_keys=True, allow_nan=False)


def _jsonable(value: object):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("experiment output cannot contain non-finite floats")
        return value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if hasattr(value, "x") and hasattr(value, "y"):
        return {"x": value.x, "y": value.y}
    return value
