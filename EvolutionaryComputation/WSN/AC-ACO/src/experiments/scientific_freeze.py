"""Deterministic identity for a complete scientific calibration freeze."""

import hashlib
import json
from pathlib import Path


HYBRID_MULTIFLOW_IMPLEMENTATION_SEMANTICS = "CURRENT_POST_PHASE1_FIX_HYBRID_MULTIFLOW"
HYBRID_MULTIFLOW_PROVENANCE_DIRECTORY = "hybrid_multiflow_provenance"
_HYBRID_MULTIFLOW_SOURCES = (
    "src/ac_aco_mrp/multiflow.py",
    "src/ac_aco_mrp/multiflow_energy.py",
    "src/ac_aco_mrp/run_hybrid.py",
    "src/ac_aco_mrp/types.py",
    "src/experiments/metrics.py",
)


def scientific_freeze_identity(payload: dict[str, object]) -> tuple[str, str]:
    """Return a stable ID/hash; callers supply only non-volatile provenance."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:16], digest


def hybrid_multiflow_implementation_identity(repository_root: str | Path) -> dict[str, object]:
    """Hash the corrected Hybrid representation without changing MRP parameters."""

    root = Path(repository_root).resolve()
    source_hashes = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in _HYBRID_MULTIFLOW_SOURCES
    }
    payload = {
        "classification": "HYBRID_MULTI_FLOW_REPRESENTATION_FIX",
        "implementation_semantics": HYBRID_MULTIFLOW_IMPLEMENTATION_SEMANTICS,
        "source_hashes": source_hashes,
    }
    identity_id, identity_hash = scientific_freeze_identity(payload)
    return {**payload, "implementation_id": identity_id, "implementation_hash": identity_hash}


def write_hybrid_multiflow_provenance(
    repository_root: str | Path,
    selected_config_path: str | Path,
    output_directory: str | Path,
) -> Path:
    """Atomically record the stale final-evaluation lock for review, not execution."""

    from .io import json_text, publish_output_files

    selected = json.loads(Path(selected_config_path).read_text(encoding="utf-8"))
    metadata = selected.get("calibration_metadata", {})
    payload = {
        "checkpoint": "HYBRID MULTI-FLOW ROUTING FIX",
        "classification": "HYBRID_MULTI_FLOW_REPRESENTATION_FIX",
        "implementation_identity": hybrid_multiflow_implementation_identity(repository_root),
        "selected_config_id": selected.get("selected_config_id"),
        "parameter_config_hash": metadata.get("parameter_config_hash"),
        "previous_scientific_freeze_id": metadata.get("calibration_freeze_id"),
        "previous_scientific_freeze_hash": metadata.get("calibration_freeze_hash"),
        "mrp_parameter_snapshot_changed": False,
        "pure_mrp_calibration_rerun": False,
        "reserved_evaluation_executed": False,
        "final_evaluation_authorization": "STALE_PENDING_REVIEW",
    }
    return publish_output_files(output_directory, {"implementation_provenance.json": json_text(payload)})
