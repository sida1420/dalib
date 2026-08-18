import json
from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments import (
    CALIBRATION_SEEDS,
    EVALUATION_SEEDS,
    FrozenCalibrationError,
    load_checkpoint_9c_lock,
    load_existing_map_network,
    load_frozen_configuration,
    primary_calibration_config,
)


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "real_calibration"
CURRENT = ROOT / "real_calibration_v2"
OUTPUTS = CURRENT / "calibration_outputs"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_checkpoint_9c_reuses_the_exact_frozen_scenario_manifest_and_seed_split():
    assert (CURRENT / "scenario_manifest.json").read_bytes() == (HISTORICAL / "scenario_manifest.json").read_bytes()
    lock = load_checkpoint_9c_lock(OUTPUTS)
    assert tuple(lock["calibration_seeds"]) == CALIBRATION_SEEDS
    assert tuple(lock["reserved_evaluation_seeds"]) == EVALUATION_SEEDS
    assert not lock["evaluation_seeds_executed"] and not lock["evaluation_scenarios_executed"]


def test_checkpoint_9c_reruns_the_approved_stage_counts_and_preserves_pure_mrp_ranking():
    report, manifest = _json(OUTPUTS / "checkpoint_9c_report.json"), _json(OUTPUTS / "calibration_manifest.json")
    assert report["stage_a_candidate_count"] == 16 and report["stage_b_candidate_count"] == 162
    assert manifest["plan"]["calibration_seed_set"] == list(CALIBRATION_SEEDS)
    assert manifest["plan"]["evaluation_seed_set"] == list(EVALUATION_SEEDS)
    assert "normalized own-energy" in manifest["selection_rule"]


def test_checkpoint_9c_selected_config_encodes_current_semantics_and_stays_override_locked():
    selected = _json(OUTPUTS / "selected_config.json")
    assert selected["calibration_status"] == "SCIENTIFIC_CALIBRATION_COMPLETE"
    assert selected["implementation_version"] == "checkpoint-9c-current-post-phase1-fix"
    metadata = selected["calibration_metadata"]
    assert metadata["implementation_semantics"] == "CURRENT_POST_PHASE1_FIX"
    assert metadata["historical_calibration"]["status"] == "SUPERSEDED_BY_IMPLEMENTATION_SEMANTIC_CHANGE"
    with pytest.raises(FrozenCalibrationError, match="override"):
        load_frozen_configuration(
            OUTPUTS / "selected_config.json",
            primary_calibration_config(load_existing_map_network(ROOT / "map.pkl")),
            {"mrp.c": 0.01},
        )
