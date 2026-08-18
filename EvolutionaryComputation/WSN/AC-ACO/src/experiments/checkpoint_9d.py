"""Checkpoint-9D freeze binding the corrected Hybrid implementation."""

import hashlib
import json
from pathlib import Path

from .io import json_text, publish_output_files
from .real_calibration import CALIBRATION_SEEDS, EVALUATION_SEEDS
from .scientific_freeze import hybrid_multiflow_implementation_identity, scientific_freeze_identity


CHECKPOINT_9D_DIRECTORY = "checkpoint_9d_freeze"
HYBRID_IMPLEMENTATION_ID = "edefad02ed4242bd"
HYBRID_SEMANTICS = "MULTI_FLOW_PER_SELECTED_CH"
CHECKPOINT_9D_STALE_STATUS = "STALE_BY_NETWORK_WIDE_MRP_ADAPTATION"

_BASELINE_SOURCES = (
    "src/experiments/baseline_adapter.py", "src/ac_aco_mrp/phase1.py",
    "src/evaluate.py", "src/experiments/runners.py",
)


class Checkpoint9DError(ValueError):
    """Raised when the corrected Hybrid evaluation boundary is not intact."""


def create_checkpoint_9d_freeze(root: str | Path, selected_path: str | Path, scenario_manifest: str | Path) -> Path:
    """Atomically create the new experiment-layer freeze without recalibration."""

    root = Path(root).resolve()
    selected = _read_json(Path(selected_path))
    scenarios = _read_json(Path(scenario_manifest))
    metadata = selected.get("calibration_metadata", {})
    old_payload = metadata.get("scientific_freeze_payload", {})
    hybrid = hybrid_multiflow_implementation_identity(root)
    if hybrid["implementation_id"] != HYBRID_IMPLEMENTATION_ID:
        raise Checkpoint9DError("corrected Hybrid implementation identity is not approved")
    payload = {
        "checkpoint": "9D — POST-HYBRID-FIX SCIENTIFIC FREEZE",
        "parameter_config": {"id": metadata.get("parameter_config_id"), "hash": metadata.get("parameter_config_hash")},
        "pure_mrp_implementation": {"source_hashes": old_payload.get("implementation_source_hashes")},
        "hybrid_implementation": hybrid,
        "baseline_implementation": {"source_hashes": _source_hashes(root, _BASELINE_SOURCES)},
        "scenario_hashes": metadata.get("scenario_hashes"),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "reserved_evaluation_seeds": list(EVALUATION_SEEDS),
        "protocol": {
            "search_policy": old_payload.get("search_space_policy"),
            "ranking_policy": old_payload.get("ranking_policy"),
            "pheromone_lifecycle": old_payload.get("pheromone_lifecycle"),
            "primary_stop": "FND", "max_rounds": 5000,
        },
        "traffic_semantics": {
            "baseline": "all-live-node membership / traffic",
            "hybrid": "all-live-node membership / traffic",
            "pure_mrp": "event-scoped source traffic",
            "hybrid_routing_semantics": HYBRID_SEMANTICS,
            "cross_ch_single_parent_constraint": "DISABLED",
            "phase_ii_per_ch": "YES", "phase_iii_per_ch": "YES",
            "all_selected_ch_flows_transmit": "YES",
        },
        "scenario_manifest_hash": hashlib.sha256(Path(scenario_manifest).read_bytes()).hexdigest(),
        "scenario_manifest_set_ids": sorted(scenarios.get("scenario_sets", {})),
    }
    freeze_id, freeze_hash = scientific_freeze_identity(payload)
    record = {
        "checkpoint": payload["checkpoint"], "scientific_freeze_id": freeze_id,
        "scientific_freeze_hash": freeze_hash, "payload": payload,
        "previous_scientific_freeze": {"id": metadata.get("calibration_freeze_id"), "hash": metadata.get("calibration_freeze_hash")},
        "previous_checkpoint_10_authorization": "STALE_BY_HYBRID_IMPLEMENTATION_CHANGE",
        "mrp_parameter_snapshot_changed": False, "pure_mrp_calibration_rerun": False,
        "evaluation_seeds_executed": False, "evaluation_scenarios_executed": False,
    }
    destination = Path(selected_path).resolve().parent.parent / CHECKPOINT_9D_DIRECTORY
    if destination.exists():
        raise Checkpoint9DError("Checkpoint-9D freeze already exists; it is immutable")
    return publish_output_files(destination, {"scientific_freeze.json": json_text(record)})


def load_checkpoint_9d_freeze(root: str | Path, selected_path: str | Path, scenario_manifest: str | Path) -> dict[str, object]:
    """Validate the active freeze against the current source and immutable inputs."""

    selected_path = Path(selected_path).resolve()
    record = _read_json(selected_path.parent.parent / CHECKPOINT_9D_DIRECTORY / "scientific_freeze.json")
    payload = record.get("payload")
    freeze_id, freeze_hash = scientific_freeze_identity(payload) if isinstance(payload, dict) else (None, None)
    if (record.get("scientific_freeze_id"), record.get("scientific_freeze_hash")) != (freeze_id, freeze_hash):
        raise Checkpoint9DError("Checkpoint-9D freeze identity is invalid")
    if record.get("previous_checkpoint_10_authorization") != "STALE_BY_HYBRID_IMPLEMENTATION_CHANGE":
        raise Checkpoint9DError("previous Checkpoint-10 authorization was not marked stale")
    traffic = payload.get("traffic_semantics", {}) if isinstance(payload, dict) else {}
    if isinstance(traffic, dict) and traffic.get("pure_mrp") == "event-scoped source traffic":
        raise Checkpoint9DError(
            f"Checkpoint-9D scientific freeze is {CHECKPOINT_9D_STALE_STATUS}"
        )
    _validate_payload(Path(root).resolve(), selected_path, Path(scenario_manifest).resolve(), payload)
    return record


def _validate_payload(root: Path, selected_path: Path, scenario_path: Path, payload: dict[str, object]) -> None:
    selected = _read_json(selected_path)
    metadata = selected.get("calibration_metadata", {})
    if payload.get("parameter_config") != {"id": metadata.get("parameter_config_id"), "hash": metadata.get("parameter_config_hash")}:
        raise Checkpoint9DError("frozen parameter configuration changed")
    if payload.get("scenario_hashes") != metadata.get("scenario_hashes"):
        raise Checkpoint9DError("frozen scenario hashes changed")
    if payload.get("calibration_seeds") != list(CALIBRATION_SEEDS) or payload.get("reserved_evaluation_seeds") != list(EVALUATION_SEEDS):
        raise Checkpoint9DError("frozen seed split changed")
    if payload.get("hybrid_implementation") != hybrid_multiflow_implementation_identity(root):
        raise Checkpoint9DError("Hybrid implementation does not match the Checkpoint-9D multi-flow freeze")
    pure = payload.get("pure_mrp_implementation", {})
    pure_hashes = pure.get("source_hashes") if isinstance(pure, dict) else None
    if not isinstance(pure_hashes, dict) or pure_hashes != _source_hashes(root, tuple(pure_hashes)):
        raise Checkpoint9DError("Pure-MRP implementation changed")
    if payload.get("baseline_implementation") != {"source_hashes": _source_hashes(root, _BASELINE_SOURCES)}:
        raise Checkpoint9DError("Baseline implementation changed")
    if payload.get("scenario_manifest_hash") != hashlib.sha256(scenario_path.read_bytes()).hexdigest():
        raise Checkpoint9DError("frozen scenario manifest changed")
    traffic = payload.get("traffic_semantics", {})
    if not isinstance(traffic, dict) or traffic.get("hybrid_routing_semantics") != HYBRID_SEMANTICS:
        raise Checkpoint9DError("single-parent Hybrid semantics cannot authorize final evaluation")


def _source_hashes(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in paths}


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Checkpoint9DError(f"cannot read required freeze artifact: {path}") from error
