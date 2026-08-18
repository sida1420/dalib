import importlib
import json
import math
from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments import (
    EvaluationView, ExperimentMode, StopCondition, load_existing_map_network,
    load_frozen_configuration, load_frozen_scenario_sets, primary_calibration_config,
    run_trial,
)
from experiments.manual_runner import execute_route_exhaustion
from experiments.metrics import network_wide_mrp_metrics
from experiments.native_case import NATIVE_CASE_ID, build_native_round_context
from experiments.network_wide_mrp import _failure
from mrp import (
    MRPConfig, NETWORK_WIDE_MRP_ADAPTATION, NetworkWideMRPClusteringError,
    select_network_wide_cluster_heads,
)
from mrp import PureMRPState
from point import Point


ROOT = Path(__file__).resolve().parents[2]


def test_network_wide_phase1_selects_exactly_n_distinct_live_mrp_heads():
    nodes = tuple(Point(index, 0) for index in range(6))
    distances = tuple(tuple(abs(left - right) for right in nodes) for left in nodes)
    result = select_network_wide_cluster_heads(
        nodes, range(6), (10.0,) * 6, distances, 3.0, MRPConfig(), 2,
    )
    assert len(result.selected_cluster_heads) == len(set(result.selected_cluster_heads)) == 2
    assert result.classification == NETWORK_WIDE_MRP_ADAPTATION
    with pytest.raises(NetworkWideMRPClusteringError, match="MRP_CLUSTERING_EXHAUSTED"):
        select_network_wide_cluster_heads(
            nodes, range(6), (10.0,) * 6, distances, 0.5, MRPConfig(), 2,
        )


def test_fair_initial_map_has_ten_chs_ten_flows_and_equal_full_network_workload():
    existing = load_existing_map_network(ROOT / "map.pkl")
    frozen = load_frozen_configuration(
        ROOT / "real_calibration_v2" / "calibration_outputs" / "selected_config.json",
        primary_calibration_config(existing),
    )
    config = replace(
        frozen, max_rounds=1, trial_count=1,
        modes=(ExperimentMode.BASELINE, ExperimentMode.MRP_NETWORK_WIDE, ExperimentMode.HYBRID),
        evaluation_view=EvaluationView.NATIVE_ALGORITHM_SEMANTICS,
        stop_condition=StopCondition.MAX_ROUNDS,
    )
    scenarios = load_frozen_scenario_sets(ROOT / "real_calibration_v2" / "scenario_manifest.json")
    scenario = next(item for item in scenarios.calibration if item.scenario_id == "calibration-v1-c1")
    trial = run_trial(existing.network, (scenario.round_scenario(),), config, 0, 33001)
    target = math.ceil(config.ac_aco_parameters.ch_proportion * len(existing.network.nodes))
    rows = {
        mode: trial.mode_results[mode].rounds[0]
        for mode in config.modes
    }
    assert target == 10
    assert all(row.success for row in rows.values())
    assert {mode: len(rows[mode].selected_cluster_heads) for mode in config.modes} == {
        ExperimentMode.BASELINE: 10,
        ExperimentMode.MRP_NETWORK_WIDE: 10,
        ExperimentMode.HYBRID: 10,
    }
    assert all(len(row.declared_source_sensors) == 100 for row in rows.values())
    assert all(row.represented_payload_sensor_count == 100 for row in rows.values())
    assert len(rows[ExperimentMode.MRP_NETWORK_WIDE].selected_routes) == 10
    assert len(rows[ExperimentMode.HYBRID].selected_routes) == 10


def test_fair_pure_and_hybrid_share_router_phases_and_multiflow_energy_while_event_reference_remains():
    fair = importlib.import_module("experiments.network_wide_mrp")
    hybrid = importlib.import_module("ac_aco_mrp.run_hybrid")
    engine = importlib.import_module("mrp.multi_ch_routing")
    event = importlib.import_module("mrp.run_mrp")
    assert fair.route_cluster_heads_with_mrp is hybrid.route_cluster_heads_with_mrp
    assert fair.route_cluster_heads_with_mrp is event.route_cluster_heads_with_mrp
    assert fair.discover_mrp_phase2_routes is hybrid.discover_mrp_phase2_routes
    assert fair.discover_mrp_phase2_routes is engine.discover_mrp_phase2_routes
    assert fair.select_phase3_route is hybrid.select_phase3_route
    assert fair.select_phase3_route is engine.select_phase3_route
    assert fair.evaluate_hybrid_multiflow_energy is hybrid.evaluate_hybrid_multiflow_energy
    assert callable(event.run_pure_mrp_round)


def test_shared_relay_with_different_flow_next_hops_remains_legal_for_common_multiflow_plan():
    builder = importlib.import_module("ac_aco_mrp.multiflow").build_hybrid_multiflow_plan
    plan = builder(
        (66, 84, 10), {66: (), 84: (), 10: (23,)},
        {66: (66, -1), 84: (84, 66, -1), 10: (10, 84, 23, -1)},
        {10, 23, 66, 84}, 100,
    )
    assert len(plan.selected_flows) == 3
    assert {flow.source_ch for flow in plan.selected_flows} == {10, 66, 84}


def test_partial_required_ch_failure_is_terminal_and_metrics_safe(tmp_path):
    heads = (29, 18, 44, 66, 75, 8, 59, 37, 99, 11)
    completed_heads = heads[:8]
    state = PureMRPState((0.6,) * 100, tuple(range(100)), frozenset(), 7.91302873669)
    phase2 = SimpleNamespace(
        status="completed", num_sants_requested=20, num_sants_succeeded=20,
        num_unique_routes=1, multipath_ready=False,
    )
    phase3 = SimpleNamespace(
        selected_route=(29, -1), selected_index=0,
        candidates=(SimpleNamespace(path_length=1.0),),
    )
    completed = tuple(
        SimpleNamespace(cluster_head=head, phase2_result=phase2, phase3_result=phase3)
        for head in completed_heads
    )
    selected_routes = {head: (head, -1) for head in completed_heads}
    failed_phase2 = SimpleNamespace(
        status="completed", num_sants_requested=20, num_sants_succeeded=0,
        num_unique_routes=0, multipath_ready=False,
    )
    raw = _failure(
        1, state, "mrp_routing", "MRP_SEARCH_FAILURE: CH 99: no unique successful routes",
        SimpleNamespace(selected_cluster_heads=heads), {head: () for head in heads},
        completed, selected_routes, failed_cluster_head=99,
        routing_failure_classification="MRP_SEARCH_FAILURE",
        failed_phase2_result=failed_phase2,
    )

    terminal_row = network_wide_mrp_metrics(raw, 0, 33001, 2000.0)

    assert terminal_row.success is False
    assert terminal_row.selected_routes == tuple(selected_routes.values())
    assert terminal_row.actual_round_energy is None
    assert terminal_row.delivered_payload_count is None
    assert raw.state_after is raw.state_before
    assert raw.routing_status_by_ch[99] == "MRP_SEARCH_FAILURE"
    assert raw.routing_status_by_ch[11] == "NOT_ATTEMPTED_AFTER_REQUIRED_FAILURE"

    successful = SimpleNamespace(
        success=True, phase1_result=SimpleNamespace(selected_cluster_heads=heads),
        selected_routes={head: (head, -1) for head in heads},
    )
    successful_row = SimpleNamespace(
        round_index=0, live_after=tuple(range(100)), dead_count=0,
        cumulative_actual_energy=0.5, delivered_payload_count=100,
        residual_total=59.5, residual_mean=0.595,
    )
    rows = iter(((successful, successful_row), (raw, terminal_row)))
    existing = load_existing_map_network(ROOT / "map.pkl")
    scenario = build_native_round_context(existing, 100.0)
    summary = execute_route_exhaustion(
        canonical_mode="mrp_pure", experiment_mode=ExperimentMode.MRP_NETWORK_WIDE,
        network=existing.network, scenario=scenario, runtime=SimpleNamespace(state=state),
        config=SimpleNamespace(mrp_parameters=SimpleNamespace(communication_radius=100.0)),
        seeds=object(), seed=33001, scenario_id=NATIVE_CASE_ID, anchor=None,
        safety_max_rounds=10, round_executor=lambda *_: next(rows),
    )
    output = tmp_path / "compact_terminal.json"
    output.write_text(json.dumps(summary), encoding="utf-8")

    assert summary["successful_rounds"] == 1
    assert summary["last_successful_round_zero_based"] == 0
    assert summary["routing_failure_round_zero_based"] == 1
    assert summary["physical_energy"] == 0.5
    assert summary["delivered_payloads"] == 100
    assert summary["failure_details"]["selected_flow_count"] == 8
    assert summary["failure_details"]["failed_ch_ids"] == [99]
    assert summary["failure_details"]["routing_status_by_ch"]["99"] == "MRP_SEARCH_FAILURE"
    assert summary["failure_details"]["routing_status_by_ch"]["11"] == "NOT_ATTEMPTED_AFTER_REQUIRED_FAILURE"
    assert output.exists()


def test_each_fair_mode_builds_and_commits_exactly_one_complete_round(monkeypatch):
    existing = load_existing_map_network(ROOT / "map.pkl")
    frozen = load_frozen_configuration(
        ROOT / "real_calibration_v2" / "calibration_outputs" / "selected_config.json",
        primary_calibration_config(existing),
    )
    scenario = build_native_round_context(existing, frozen.mrp_parameters.communication_radius)
    modules = {
        ExperimentMode.BASELINE: importlib.import_module("experiments.baseline_adapter"),
        ExperimentMode.MRP_NETWORK_WIDE: importlib.import_module("experiments.network_wide_mrp"),
        ExperimentMode.HYBRID: importlib.import_module("ac_aco_mrp.run_hybrid"),
    }
    observed = {}

    for mode, module in modules.items():
        names = (
            ("select_ac_aco_cluster_heads", "network_config", "energy_consumption", "apply_baseline_lifecycle")
            if mode is ExperimentMode.BASELINE else
            ("select_network_wide_cluster_heads", "route_cluster_heads_with_mrp", "evaluate_hybrid_multiflow_energy", "apply_baseline_lifecycle")
            if mode is ExperimentMode.MRP_NETWORK_WIDE else
            ("select_ac_aco_cluster_heads", "route_cluster_heads_with_mrp", "evaluate_hybrid_multiflow_energy", "apply_baseline_lifecycle")
        )
        calls = {name: 0 for name in names}
        originals = {name: getattr(module, name) for name in names}
        for name in names:
            def counted(*args, _name=name, **kwargs):
                calls[_name] += 1
                return originals[_name](*args, **kwargs)
            monkeypatch.setattr(module, name, counted)
        config = replace(
            frozen, max_rounds=1, trial_count=1, modes=(mode,),
            evaluation_view=EvaluationView.NATIVE_ALGORITHM_SEMANTICS,
            stop_condition=StopCondition.MAX_ROUNDS,
        )
        trial = run_trial(existing.network, (scenario,), config, 0, 33001)
        result = trial.mode_results[mode]
        row, raw = result.rounds[0], result.raw_round_results[0]
        assert len(result.rounds) == 1 and row.round_index == 0 and row.success
        assert len(row.selected_cluster_heads) == 10
        assert row.delivered_payload_count == 100
        assert raw.state_after != raw.state_before
        assert all(count == 1 for count in calls.values())
        if mode is not ExperimentMode.BASELINE:
            assert len(raw.ch_routing_results) == 10
            assert len(raw.selected_routes) == 10
            assert raw.final_plan is not None
        observed[mode] = calls
        monkeypatch.undo()

    assert set(observed) == {
        ExperimentMode.BASELINE, ExperimentMode.MRP_NETWORK_WIDE, ExperimentMode.HYBRID,
    }
