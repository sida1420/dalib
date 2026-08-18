from pathlib import Path
import sys
from dataclasses import replace

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ac_aco_mrp import ACACOParameters, HybridParameters, HybridState, initialize_ac_aco_phase1_state, run_hybrid_round, select_ac_aco_cluster_heads
from experiments import (
    EvaluationView, ExperimentConfig, ExperimentMode, ExperimentNetwork, ExperimentRoundScenario,
    StopCondition, derive_lifetime_metrics, run_baseline_round, run_experiment,
    run_trial, smoke_trace, write_experiment_outputs,
)
from experiments.baseline_adapter import BaselineState
from experiments.metrics import aggregate_mode_trials, assess_trial_fairness
from experiments.types import ModeTrialResult, RoundMetrics
from mrp import HeuristicBounds, MRPConfig, PheromoneLifecycle, PureMRPParameters, PureMRPState
from mrp.runner_lifecycle import apply_baseline_lifecycle
from point import Point
from evaluate import energy_consumption, network_config
import random



def network(nodes=None, base_x=3.0, energy=100.0):
    nodes = tuple(nodes or (Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)))
    base = Point(base_x, 0)
    return ExperimentNetwork("fixed-four-node", nodes, tuple(tuple(abs(left - right) for right in nodes) for left in nodes), tuple(abs(base - node) for node in nodes), tuple(energy for _ in nodes))


def mrp_parameters(**overrides):
    values = dict(communication_radius=50.0, heuristic_bounds=HeuristicBounds(0.1, 1000.0, 0.1, 1000.0), config=MRPConfig(aant_probability=0.0), lambda_coefficient=0.1, ttl=2, num_sants=1, c0=1.0, c=0.0, c1=0.01, d0=10.0, bit_count=1.0, ctrl_bit=0.0, e_elec=2.0, e_agg=0.0, free_space_coeff=1.0, multipath_coeff=1.0)
    values.update(overrides)
    return PureMRPParameters(**values)


def config(modes=(ExperimentMode.BASELINE, ExperimentMode.MRP_NETWORK_WIDE, ExperimentMode.HYBRID), **overrides):
    ac = ACACOParameters(Point(0, 3), 400.0, candidate_count=2, ch_proportion=0.5)
    values = dict(max_rounds=2, trial_count=1, modes=tuple(modes), evaluation_view=EvaluationView.NATIVE_ALGORITHM_SEMANTICS, stop_condition=StopCondition.MAX_ROUNDS, ac_aco_parameters=ac, mrp_parameters=mrp_parameters(), mrp_pheromone_lifecycle=PheromoneLifecycle.RESET_PER_DISCOVERY, parameter_provenance={"num_sants": "USER_SELECTED_PARAMETER", "alpha": "PAPER_SIMULATION_PARAMETER"})
    values.update(overrides)
    return ExperimentConfig(**values)


def scenario(event_nodes=(0, 1)):
    return ExperimentRoundScenario(tuple(event_nodes), {node: 10.0 - node for node in event_nodes}, {0: 1, 1: 1, 2: 1, 3: 1})


def test_fair_modes_have_equal_workload_and_event_reference_remains_scoped():
    trial = run_trial(network(), (scenario(),), config(), 0, 123)
    baseline = trial.mode_results[ExperimentMode.BASELINE].rounds[0]
    pure = trial.mode_results[ExperimentMode.MRP_NETWORK_WIDE].rounds[0]
    hybrid = trial.mode_results[ExperimentMode.HYBRID].rounds[0]
    assert (baseline.represented_payload_sensor_count, pure.represented_payload_sensor_count, hybrid.represented_payload_sensor_count) == (4, 4, 4)
    assert trial.fairness.raw_energy_status == "DIRECTLY_COMPARABLE"
    assert all(row.energy_per_delivered_payload is not None for row in (baseline, pure, hybrid))
    assert "FAIRNESS" in smoke_trace(trial)
    event = run_trial(network(), (scenario(),), config((ExperimentMode.PURE_MRP,)), 0, 123)
    assert event.mode_results[ExperimentMode.PURE_MRP].rounds[0].represented_payload_sensor_count == 2
    normalized = run_trial(network(), (scenario(),), config(evaluation_view=EvaluationView.NORMALIZED_REPORTING), 0, 123)
    assert normalized.fairness.reporting_view is EvaluationView.NORMALIZED_REPORTING
    assert normalized.fairness.normalized_energy_status == "DIRECTLY_COMPARABLE"


def test_mode_ordering_and_input_state_are_isolated():
    initial = network()
    for mode in ExperimentMode:
        alone = run_trial(initial, (scenario(),), config((mode,)), 0, 99)
        combined = run_trial(initial, (scenario(),), config(tuple(ExperimentMode)), 0, 99)
        assert alone.mode_results[mode].rounds == combined.mode_results[mode].rounds
    alone = run_trial(initial, (scenario(),), config((ExperimentMode.PURE_MRP,)), 0, 99)
    combined = run_trial(initial, (scenario(),), config(tuple(ExperimentMode)), 0, 99)
    assert alone.mode_results[ExperimentMode.PURE_MRP].rounds == combined.mode_results[ExperimentMode.PURE_MRP].rounds
    assert initial.initial_residual_e == (100.0,) * 4
    assert all(node.x == value for node, value in zip(initial.nodes, (0, 1, 2, 3)))
    with pytest.raises(ValueError, match="modes"):
        run_trial(initial, (scenario(),), config(()), 0, 99)


def test_equal_workload_is_marked_directly_comparable():
    trial = run_trial(network(), (scenario((0, 1, 2, 3)),), config(), 0, 123)
    assert trial.fairness.raw_energy_status == "DIRECTLY_COMPARABLE"
    assert {result.rounds[0].delivered_payload_count for result in trial.mode_results.values()} == {4}


def test_baseline_adapter_matches_safe_baseline_final_greedy_reconstruction():
    fixed, params = network(), mrp_parameters()
    ac = config((ExperimentMode.BASELINE,)).ac_aco_parameters
    rng_one, rng_two, init_rng = random.Random(31), random.Random(31), random.Random(7)
    state = BaselineState(PureMRPState((100.0,) * 4, (0, 1, 2, 3), frozenset()), initialize_ac_aco_phase1_state(4, ac, init_rng))
    adapter = run_baseline_round(fixed.nodes, fixed.dist_matrix, fixed.base_dists, state, ac, params, rng_one, 0)
    phase1 = select_ac_aco_cluster_heads(fixed.nodes, fixed.dist_matrix, fixed.base_dists, state.physical_state.residual_e, state.physical_state.live_nodes, state.ac_aco_state, ac, params, rng_two, 0)
    topology = network_config(fixed.nodes, list(phase1.selected_cluster_heads), params.communication_radius, params.d0, ac.base_pos, phase1.state_after.hopping_factor, fixed.base_dists, fixed.dist_matrix, state.physical_state.residual_e)
    e_m, e_sum = energy_consumption(fixed.nodes, topology, params.d0, params.bit_count, params.ctrl_bit, fixed.base_dists, fixed.dist_matrix, params.e_elec, params.e_agg, params.free_space_coeff, params.multipath_coeff)
    expected_state, _ = apply_baseline_lifecycle(state.physical_state, e_m, e_sum, None)
    assert adapter.phase1_result.selected_cluster_heads == phase1.selected_cluster_heads
    assert adapter.e_sum == e_sum and adapter.e_m_list == tuple(e_m)
    assert adapter.state_after.physical_state == expected_state


def test_actual_energy_is_final_only_and_failures_add_none():
    trial = run_trial(network(), (scenario(),), config(), 0, 123)
    for result in trial.mode_results.values():
        row = result.rounds[0]
        assert row.actual_round_energy == pytest.approx(row.cumulative_actual_energy)
    failed = run_trial(network(), (scenario((0,)),), config((ExperimentMode.PURE_MRP,)), 0, 123)
    row = failed.mode_results[ExperimentMode.PURE_MRP].rounds[0]
    assert not row.success and row.actual_round_energy is None and row.cumulative_actual_energy == 0


def test_hybrid_final_mrp_energy_differs_from_legacy_greedy_fitness_and_is_the_only_charge():
    nodes = (Point(0, 0), Point(4, 0), Point(8, 0), Point(12, 0))
    fixed = network(nodes, base_x=16.0)
    parameters = mrp_parameters(
        communication_radius=8.0, ttl=8, d0=10.0, c=0.01, c1=0.01,
        free_space_coeff=0.01, multipath_coeff=0.01,
    )
    ac = ACACOParameters(Point(16, 0), 400.0, candidate_count=2, ch_proportion=0.5)
    state = HybridState(PureMRPState((100.0,) * 4, (0, 1, 2, 3), frozenset()), initialize_ac_aco_phase1_state(4, ac, __import__("random").Random(99)))
    result = run_hybrid_round(
        fixed.nodes, fixed.dist_matrix, fixed.base_dists, state, __import__("ac_aco_mrp").HybridRoundContext({0: 3, 1: 2, 2: 1, 3: 1}),
        HybridParameters(ac, parameters), __import__("random").Random(1), __import__("random").Random(1), PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )
    assert result.success and result.phase1_result.selected_legacy_energy_sum == pytest.approx(17.28)
    assert result.e_sum == pytest.approx(13.6) and result.e_sum != result.phase1_result.selected_legacy_energy_sum
    assert result.state_after.physical_state.residual_e == tuple(100.0 - item for item in result.e_m_list)


def test_seed_reproducibility_manifest_and_raw_outputs(tmp_path):
    one = run_experiment(network(), (scenario(),), config(), (17,))
    two = run_experiment(network(), (scenario(),), config(), (17,))
    assert one.trials[0].seeds == two.trials[0].seeds and one.trials[0].seeds.trial_seed == 17
    for mode in one.trials[0].mode_results:
        assert one.trials[0].mode_results[mode].rounds == two.trials[0].mode_results[mode].rounds
    directory = write_experiment_outputs(one, config(), network(), (scenario(),), tmp_path / "raw")
    assert {path.name for path in directory.iterdir()} == {"manifest.json", "round_metrics.csv", "trial_metrics.json", "failures.csv"}
    with pytest.raises(FileExistsError):
        write_experiment_outputs(one, config(), network(), (scenario(),), directory)
    invalid = tmp_path / "invalid"
    with pytest.raises(ValueError, match="non-finite"):
        write_experiment_outputs(one, replace(config(), parameter_provenance={"bad": float("nan")}), network(), (scenario(),), invalid)
    assert not invalid.exists()


def _metric(round_index, dead_count):
    return RoundMetrics(
        algorithm=ExperimentMode.BASELINE, trial_id=0, trial_seed=1, round_index=round_index,
        live_before=(), live_after=(), dead_count=dead_count, newly_dead=(),
        declared_source_sensors=(), represented_payload_sensor_count=1, delivered_payload_count=1,
        delivered_bits=1.0, selected_cluster_heads=(), selected_routes=(), cluster_sizes={},
        actual_round_energy=1.0, energy_per_delivered_payload=1.0, energy_per_delivered_bit=1.0,
        energy_per_represented_sensor=1.0, cumulative_actual_energy=1.0, residual_total=1.0,
        residual_mean=1.0, residual_std_population=0.0, residual_min=1.0, selected_route_hops=(), selected_route_lengths=(),
        discovered_route_counts=(), multipath_ready=(), success=True, failure_stage=None,
        failure_reason=None,
    )


def test_lifetime_metrics_define_fnd_hnd_lnd_and_censoring_for_odd_even_networks():
    even = derive_lifetime_metrics((_metric(0, 0), _metric(1, 1), _metric(2, 2), _metric(3, 4)), 4)
    odd = derive_lifetime_metrics((_metric(0, 0), _metric(1, 2)), 5)
    assert (even.fnd_round, even.hnd_round, even.lnd_round) == (1, 2, 3)
    assert odd.fnd_round == 1 and odd.hnd_round is None and odd.lnd_censored


def test_aggregate_excludes_incomplete_trials_and_censored_lifetime_is_not_directly_comparable():
    complete = ModeTrialResult(ExperimentMode.BASELINE, (_metric(0, 0),), (), derive_lifetime_metrics((_metric(0, 0),), 2))
    failed_row = replace(_metric(0, 0), success=False, actual_round_energy=None, delivered_payload_count=None)
    incomplete = ModeTrialResult(ExperimentMode.BASELINE, (failed_row,), (), derive_lifetime_metrics((failed_row,), 2))
    aggregate = aggregate_mode_trials((complete, incomplete), EvaluationView.NORMALIZED_REPORTING)
    assert aggregate["actual_energy"]["sample_count"] == 1
    assert aggregate["completion"] == {"complete_trial_count": 1, "incomplete_trial_count": 1, "status": "INCOMPLETE_TRIALS_EXCLUDED"}
    audit = assess_trial_fairness({ExperimentMode.BASELINE: complete, ExperimentMode.PURE_MRP: complete})
    assert audit.lifetime_status == "CENSORED_SURVIVAL_ANALYSIS_REQUIRED"
