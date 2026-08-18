"""Operational launcher for the single authorized Checkpoint-10 evaluation."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from experiments import (  # noqa: E402
    load_existing_map_network,
    preflight_final_evaluation,
    run_and_write_frozen_final_evaluation,
)
from experiments.checkpoint_9d import load_checkpoint_9d_freeze  # noqa: E402


PROGRESS = ROOT / "real_calibration_v2" / "checkpoint10_progress.json"
SELECTED = ROOT / "real_calibration_v2" / "calibration_outputs" / "selected_config.json"
SCENARIOS = ROOT / "real_calibration_v2" / "scenario_manifest.json"
OUTPUT = ROOT / "real_calibration_v2" / "final_evaluation_outputs"

EXPECTED_CONFIG_ID = "62963da9e200e409"
EXPECTED_CONFIG_HASH = "b6e3ef93ce5de68ce5c4f997454f0cd3d4ef662c71e51fd583e3bc0a48128473"
PLANNED_NATIVE_TRIALS = 180
HEARTBEAT_SECONDS = 30.0


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_progress(payload, lock):
    """Write operational state only; never used by the scientific evaluator."""
    try:
        with lock:
            PROGRESS.parent.mkdir(parents=True, exist_ok=True)
            PROGRESS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as error:  # operational observability must not affect evaluation
        print(f"checkpoint10 heartbeat write failed: {type(error).__name__}: {error}", file=sys.stderr)


def _heartbeat(payload, lock, stop):
    while not stop.wait(HEARTBEAT_SECONDS):
        payload["heartbeat_timestamp"] = _timestamp()
        _write_progress(payload, lock)


def _require_frozen_inputs(existing_map):
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite existing final output directory: {OUTPUT}")
    selected_payload = json.loads(SELECTED.read_text(encoding="utf-8"))
    metadata = selected_payload.get("calibration_metadata", {})
    checks = {
        "selected_config_id": selected_payload.get("selected_config_id") == EXPECTED_CONFIG_ID,
        "parameter_config_id": metadata.get("parameter_config_id") == EXPECTED_CONFIG_ID,
        "parameter_config_hash": metadata.get("parameter_config_hash") == EXPECTED_CONFIG_HASH,
        "selected_config_hash": metadata.get("selected_config_hash") == EXPECTED_CONFIG_HASH,
        "evaluation_seeds_unconsumed": not metadata.get("evaluation_seeds_executed"),
        "evaluation_scenarios_unconsumed": not metadata.get("evaluation_scenarios_executed"),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"frozen preflight identity check failed: {failed}")
    freeze = load_checkpoint_9d_freeze(ROOT, SELECTED, SCENARIOS)
    return preflight_final_evaluation(existing_map, SELECTED, SCENARIOS), freeze


def main():
    lock = threading.Lock()
    stop = threading.Event()
    payload = {
        "type": "OPERATIONAL_PROGRESS_ONLY",
        "pid": os.getpid(),
        "status": "PREFLIGHT",
        "start_time": _timestamp(),
        "heartbeat_timestamp": _timestamp(),
        "scientific_freeze_id": None,
        "planned_native_trials": PLANNED_NATIVE_TRIALS,
    }
    try:
        existing_map = load_existing_map_network(ROOT / "map.pkl")
        preflight, freeze = _require_frozen_inputs(existing_map)
        payload["scientific_freeze_id"] = freeze["scientific_freeze_id"]
        if preflight.selected_config_id != EXPECTED_CONFIG_ID or preflight.selected_config_hash != EXPECTED_CONFIG_HASH:
            raise RuntimeError("preflight returned an unexpected frozen configuration identity")

        payload["status"] = "RUNNING"
        payload["heartbeat_timestamp"] = _timestamp()
        _write_progress(payload, lock)
        thread = threading.Thread(target=_heartbeat, args=(payload, lock, stop), name="checkpoint10-heartbeat", daemon=True)
        thread.start()
        try:
            run_and_write_frozen_final_evaluation(existing_map, SELECTED, SCENARIOS, OUTPUT)
        finally:
            stop.set()
            thread.join(timeout=2.0)

        payload["status"] = "COMPLETE"
        payload["heartbeat_timestamp"] = _timestamp()
        payload["final_output_exists"] = OUTPUT.is_dir()
        _write_progress(payload, lock)
        if not payload["final_output_exists"]:
            raise RuntimeError("scientific publication returned without creating final output directory")
    except BaseException as error:
        stop.set()
        payload["status"] = "FAILED"
        payload["exception_type"] = type(error).__name__
        payload["exception_message"] = str(error)
        payload["heartbeat_timestamp"] = _timestamp()
        _write_progress(payload, lock)
        raise


if __name__ == "__main__":
    main()
