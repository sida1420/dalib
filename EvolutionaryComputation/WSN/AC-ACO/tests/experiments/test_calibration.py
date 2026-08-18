from dataclasses import replace
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ac_aco_mrp import ACACOParameters
from experiments import (
    CalibrationInputError, CalibrationPlan, ExperimentConfig, ExperimentMode, ExperimentNetwork,
    ExperimentRoundScenario, EvaluationView, ParameterProvenance,
    SEARCH_RANGE_PROVENANCE, StopCondition, TUNABLE_PATHS, audit_calibration_parameters,
    enumerate_calibration_candidates, load_frozen_configuration, run_mrp_calibration,
    run_mrp_calibration_candidates,
    run_trial,
    validate_calibration_plan, write_calibration_outputs,
)
from experiments import calibration as calibration_module
from experiments import real_calibration
from experiments.calibration_config import candidate_from_overrides
from mrp import HeuristicBounds, MRPConfig, PheromoneLifecycle, PureMRPParameters
from point import Point


def _network():
    return ExperimentNetwork(
        "calibration-two-node", (Point(0, 0), Point(3, 0)), ((0.0, 3.0), (3.0, 0.0)),
        (7.0, 3.0), (10.0, 8.0),
    )


def _parameters():
    return PureMRPParameters(
        5.0, HeuristicBounds(0.1, 1000.0, 0.1, 1000.0), MRPConfig(aant_probability=0.0),
        0.1, 2, 3, 1.0, 0.0, 0.5, 10.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0,
    )


def _config():
    return ExperimentConfig(
        1, 1, (ExperimentMode.PURE_MRP,), EvaluationView.NORMALIZED_REPORTING,
        StopCondition.MAX_ROUNDS, ACACOParameters(Point(0, 7), 100.0, candidate_count=1),
        _parameters(), PheromoneLifecycle.RESET_PER_DISCOVERY, {}, "test-version",
    )


def _scenario():
    return ExperimentRoundScenario((0, 1), {0: 10.0, 1: 8.0}, {0: 2, 1: 1})


def _plan(**overrides):
    values = dict(
        calibration_seed_set=(1, 2), evaluation_seed_set=(3, 4),
        calibration_scenario_set_id="calibration-events", evaluation_scenario_set_id="evaluation-events",
        calibration_scenarios=(_scenario(),), evaluation_scenarios=(_scenario(),),
        search_space={"mrp.c1": (0.01, -1.0)},
        search_space_provenance={"mrp.c1": SEARCH_RANGE_PROVENANCE}, minimum_completion_rate=None,
    )
    values.update(overrides)
    return CalibrationPlan(**values)


def test_audit_fixes_paper_parameters_and_exposes_only_unresolved_paths():
    audit = {item.name: item for item in audit_calibration_parameters(_config())}
    assert audit["mrp.config.alpha"].provenance is ParameterProvenance.PAPER_SIMULATION_PARAMETER
    assert not audit["mrp.config.alpha"].calibratable and audit["mrp.c"].calibratable
    assert audit["mrp.c"].provenance is ParameterProvenance.PAPER_DEFINED_VALUE_UNREPORTED
    assert "mrp.config.alpha" not in TUNABLE_PATHS


def test_plan_rejects_seed_leakage_and_paper_fixed_search_paths():
    with pytest.raises(CalibrationInputError, match="disjoint"):
        validate_calibration_plan(_plan(evaluation_seed_set=(2, 3)))
    with pytest.raises(CalibrationInputError, match="not an unresolved"):
        enumerate_calibration_candidates(_plan(
            search_space={"mrp.config.alpha": (2.0,)},
            search_space_provenance={"mrp.config.alpha": SEARCH_RANGE_PROVENANCE},
        ))
    assert len(enumerate_calibration_candidates(_plan())) == 2


def test_calibration_is_deterministic_isolated_and_rejects_negative_tau_candidate():
    physical = _network()
    one = run_mrp_calibration(physical, _config(), _plan())
    two = run_mrp_calibration(_network(), _config(), _plan())
    assert one.candidates == two.candidates
    assert {trial.trial_seed for item in one.candidates for trial in item.trials} == {1, 2}
    assert physical.initial_residual_e == (10.0, 8.0) and one.scenario_content_overlap
    invalid, valid = sorted(one.candidates, key=lambda item: dict(item.candidate.overrides)["mrp.c1"])
    assert invalid.summary.negative_pheromone_failure_count == 2 and not invalid.summary.valid
    assert invalid.summary.minimum_observed_tau < 0 and invalid.summary.normalized_energy_sample_count == 0
    assert invalid.rank is None and valid.summary.completed_trials == 2 and valid.rank == 1
    assert valid.summary.completion_rate == valid.summary.route_discovery_success_rate == 1.0
    assert valid.summary.multipath_ready_rate == 0.0
    assert one.selected_config_id == valid.candidate.config_id


def test_ranking_has_no_baseline_inputs_and_uses_only_internal_metrics():
    assert tuple(inspect.signature(run_mrp_calibration).parameters) == ("network", "base_config", "plan")
    result = run_mrp_calibration(_network(), _config(), _plan())
    selected = next(item for item in result.candidates if item.rank == 1)
    assert selected.summary.normalized_energy_sample_count == 2
    assert selected.summary.route_discovery_success_rate == 1.0


def test_frozen_selected_config_round_trips_and_rejects_runtime_override(tmp_path):
    result = run_mrp_calibration(_network(), _config(), _plan())
    directory = write_calibration_outputs(result, _config(), tmp_path / "calibration")
    assert {item.name for item in directory.iterdir()} == {
        "calibration_manifest.json", "candidate_configs.json", "calibration_rounds.csv",
        "calibration_trials.csv", "calibration_failures.csv", "calibration_summary.csv", "selected_config.json",
    }
    restored = load_frozen_configuration(directory / "selected_config.json", _config())
    assert _config().mrp_parameters.c1 == 0.5 and restored.mrp_parameters.c1 == 0.01
    with pytest.raises(Exception, match="override"):
        load_frozen_configuration(directory / "selected_config.json", _config(), {"mrp.c1": 0.5})


def test_raw_negative_tau_is_a_hard_gate_even_if_phase2_status_is_completed():
    metric = SimpleNamespace(
        round_index=0, success=True, failure_stage=None, failure_reason=None,
        actual_round_energy=10.0, delivered_payload_count=2, energy_per_delivered_payload=5.0,
    )
    phase2 = SimpleNamespace(
        status="completed", num_unique_routes=1, ant_results=(), multipath_ready=False,
        num_sants_requested=1, num_sants_succeeded=1, halted_ant_index=None,
        final_pheromone_state={0: {1: -0.01}},
    )
    record = calibration_module._round_record("raw-tau", 1, metric, SimpleNamespace(phase2_result=phase2))
    trial = calibration_module._trial_record("raw-tau", 1, (record,))
    summary = calibration_module._summary("raw-tau", (record,), (trial,), None)
    assert record.negative_pheromone_failure and not summary.valid
    assert "invalid_negative_pheromone_state" in summary.invalid_reasons


def test_partial_failed_trial_energy_is_unavailable_to_summary_ranking():
    completed = calibration_module.CalibrationRoundRecord(
        "partial", 1, 0, True, None, None, "completed", True, False, 1, 1, 1, 1.0,
        False, None, False, False, False, 0.01, 10.0, 2, 5.0,
    )
    failed = replace(
        completed, trial_seed=2, round_index=1, success=False, failure_stage="phase2",
        failure_reason="no route", route_discovery_success=False, actual_round_energy=None,
        delivered_payload_count=None, energy_per_delivered_payload=None,
    )
    complete_trial = calibration_module._trial_record("partial", 1, (completed,))
    partial_trial = calibration_module._trial_record("partial", 2, (completed, failed))
    summary = calibration_module._summary("partial", (completed, failed), (complete_trial, partial_trial), None)
    assert partial_trial.normalized_energy is None
    assert summary.normalized_energy_sample_count == 1
    assert summary.normalized_energy_mean == 5.0


def test_explicit_staged_candidates_cannot_escape_the_declared_grid_or_pair_list():
    plan = _plan(
        approved_candidate_overrides=((('mrp.c1', 0.01),),),
    )
    outside = candidate_from_overrides((('mrp.c1', 0.5),))
    with pytest.raises(ValueError, match="outside the approved"):
        run_mrp_calibration_candidates(_network(), _config(), plan, (outside,))

    allowed_but_unapproved = candidate_from_overrides((('mrp.c1', -1.0),))
    with pytest.raises(ValueError, match="do not match the approved"):
        run_mrp_calibration_candidates(_network(), _config(), plan, (allowed_but_unapproved,))


def test_stage_b_pair_handoff_uses_stage_a_rank_not_construction_order():
    low = candidate_from_overrides((('mrp.c', 0.005), ('mrp.c1', 0.0)))
    best = candidate_from_overrides((('mrp.c', 0.04), ('mrp.c1', 0.01)))
    second = candidate_from_overrides((('mrp.c', 0.02), ('mrp.c1', 0.005)))
    fake_result = SimpleNamespace(candidates=(
        SimpleNamespace(candidate=low, rank=3),
        SimpleNamespace(candidate=best, rank=1),
        SimpleNamespace(candidate=second, rank=2),
    ))

    assert real_calibration._top_pairs(fake_result) == ((0.04, 0.01), (0.02, 0.005))


def test_calibration_hop_provider_cannot_change_frozen_event_membership_or_signal():
    def invalid_provider(scenario, state, network):
        return ExperimentRoundScenario((0,), {0: 1.0}, scenario.hop_counts)

    with pytest.raises(ValueError, match="hop counts only"):
        run_trial(
            _network(), (_scenario(),), _config(), 0, 1,
            repeat_scenarios=True, scenario_provider=invalid_provider,
        )


def test_independent_scenario_trials_form_the_required_seed_scenario_cross_product():
    first = ExperimentRoundScenario((0, 1), {0: 10.0, 1: 8.0}, {0: 2, 1: 1}, "calibration-v1-c1")
    second = ExperimentRoundScenario((0, 1), {0: 10.0, 1: 8.0}, {0: 2, 1: 1}, "calibration-v1-c2")
    candidate = candidate_from_overrides((('mrp.c1', 0.01),))
    plan = _plan(
        calibration_scenarios=(first, second),
        search_space={"mrp.c1": (0.01,)},
        search_space_provenance={"mrp.c1": SEARCH_RANGE_PROVENANCE},
        approved_candidate_overrides=(candidate.overrides,),
    )

    result = run_mrp_calibration_candidates(
        _network(), _config(), plan, (candidate,),
        repeat_scenarios=True, independent_scenario_trials=True,
    )

    trials = result.candidates[0].trials
    assert len(trials) == 4
    assert {(item.trial_seed, item.scenario_id) for item in trials} == {
        (1, "calibration-v1-c1"), (1, "calibration-v1-c2"),
        (2, "calibration-v1-c1"), (2, "calibration-v1-c2"),
    }


def test_file_backed_candidate_workers_preserve_deterministic_results_without_shared_state():
    plan = _plan()
    candidates = tuple(enumerate_calibration_candidates(plan))

    sequential = run_mrp_calibration_candidates(_network(), _config(), plan, candidates)
    isolated = real_calibration._run_candidates_with_file_workers(_network(), _config(), plan, candidates, None)

    assert isolated.candidates == sequential.candidates


def test_file_backed_workers_serialize_the_real_live_hop_provider():
    plan = _plan()
    candidates = tuple(enumerate_calibration_candidates(plan))
    provider = real_calibration._current_hop_provider(5.0)

    sequential = run_mrp_calibration_candidates(
        _network(), _config(), plan, candidates,
        repeat_scenarios=True, scenario_provider=provider, independent_scenario_trials=True,
    )
    isolated = real_calibration._run_candidates_with_file_workers(
        _network(), _config(), plan, candidates, provider,
    )

    assert isolated.candidates == sequential.candidates
