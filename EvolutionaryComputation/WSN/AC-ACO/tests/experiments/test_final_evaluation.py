import json
from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments import (
    FINAL_MAX_ROUNDS, ExperimentMode, FrozenCalibrationError, load_existing_map_network,
    load_final_evaluation_config, preflight_final_evaluation, write_final_evaluation_outputs,
)
from experiments.calibration_freeze import load_frozen_configuration
from experiments.final_evaluation import _fnd
from experiments.scientific_freeze import (
    HYBRID_MULTIFLOW_IMPLEMENTATION_SEMANTICS,
    hybrid_multiflow_implementation_identity,
    write_hybrid_multiflow_provenance,
)
from experiments import final_evaluation as final_module
from experiments.real_calibration import primary_calibration_config


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_SELECTED = ROOT / "real_calibration" / "calibration_outputs" / "selected_config.json"
CURRENT_ROOT = ROOT / "real_calibration_v2"
SELECTED = CURRENT_ROOT / "calibration_outputs" / "selected_config.json"
SCENARIOS = CURRENT_ROOT / "scenario_manifest.json"


def test_historical_frozen_config_is_rejected_as_implementation_stale():
    with pytest.raises(final_module.FinalEvaluationError, match="historical calibration is stale"):
        preflight_final_evaluation(load_existing_map_network(ROOT / "map.pkl"), HISTORICAL_SELECTED, ROOT / "real_calibration" / "scenario_manifest.json")


def test_frozen_preflight_rejects_checkpoint_9d_as_semantically_stale():
    with pytest.raises(final_module.FinalEvaluationError, match="Checkpoint-9D"):
        preflight_final_evaluation(load_existing_map_network(ROOT / "map.pkl"), SELECTED, SCENARIOS)
    assert FINAL_MAX_ROUNDS == 5000


def test_frozen_mrp_configuration_is_identical_for_pure_and_hybrid_and_rejects_override():
    existing = load_existing_map_network(ROOT / "map.pkl")
    config = load_final_evaluation_config(existing, SELECTED)
    assert config.stop_condition.value == "FND"
    assert config.mrp_pheromone_lifecycle.value == "RESET_PER_DISCOVERY"
    with pytest.raises(FrozenCalibrationError):
        load_frozen_configuration(SELECTED, primary_calibration_config(existing), {"mrp.c": 0.01})


def test_fnd_round_indexing_censoring_and_immediate_stop_contract():
    completed = SimpleNamespace(success=True)
    zero = _fnd(SimpleNamespace(lifetime=SimpleNamespace(fnd_round=0), rounds=(completed,)), 5000)
    last = _fnd(SimpleNamespace(lifetime=SimpleNamespace(fnd_round=4999), rounds=(completed,) * 5000), 5000)
    censored = _fnd(SimpleNamespace(lifetime=SimpleNamespace(fnd_round=None), rounds=(completed,) * 5000), 5000)
    failed = _fnd(SimpleNamespace(lifetime=SimpleNamespace(fnd_round=None), rounds=(SimpleNamespace(success=False),)), 5000)
    assert (zero.round_index_zero_based, zero.round_number_one_based, zero.censored_at_cap) == (0, 1, False)
    assert (last.round_index_zero_based, last.round_number_one_based, last.censored_at_cap) == (4999, 5000, False)
    assert censored.round_index_zero_based is None and censored.censored_at_cap and censored.rounds_executed == 5000
    assert failed.algorithm_failure_before_fnd and not failed.censored_at_cap


def test_preflight_rejects_a_restored_snapshot_that_does_not_match_the_validated_json(monkeypatch):
    existing = load_existing_map_network(ROOT / "map.pkl")
    trusted = load_final_evaluation_config(existing, SELECTED)
    monkeypatch.setattr(
        final_module, "load_frozen_configuration",
        lambda *args, **kwargs: replace(trusted, mrp_parameters=replace(trusted.mrp_parameters, c=0.01)),
    )
    with pytest.raises(final_module.FinalEvaluationError, match="restored parameter snapshot"):
        final_module.preflight_final_evaluation(existing, SELECTED, SCENARIOS)


def test_hybrid_multiflow_provenance_marks_final_evaluation_stale_without_changing_parameters(tmp_path):
    first = hybrid_multiflow_implementation_identity(ROOT)
    second = hybrid_multiflow_implementation_identity(ROOT)
    output = write_hybrid_multiflow_provenance(ROOT, SELECTED, tmp_path / "provenance")
    payload = json.loads((output / "implementation_provenance.json").read_text(encoding="utf-8"))
    assert first == second and first["implementation_semantics"] == HYBRID_MULTIFLOW_IMPLEMENTATION_SEMANTICS
    assert payload["parameter_config_hash"] == "b6e3ef93ce5de68ce5c4f997454f0cd3d4ef662c71e51fd583e3bc0a48128473"
    assert payload["pure_mrp_calibration_rerun"] is False
    assert payload["final_evaluation_authorization"] == "STALE_PENDING_REVIEW"
