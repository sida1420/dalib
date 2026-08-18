from dataclasses import dataclass
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest
import csv


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments import ExperimentMode
from ac_aco_mrp import DirectMRPTopKDiagnostics
from mrp.phase2.control_energy import AntControlEnergy, zero_ant_control_energy
from experiments.manual_runner import (
    _EXPERIMENT_MODES, ManualRunnerError, _failure_details, execute_route_exhaustion,
    normalize_manual_mode, run_manual_route_exhaustion,
)
from experiments.native_case import NATIVE_CASE_ID, build_native_round_context, native_case_identity
from experiments.process_metrics import process_memory_bytes
from experiments.real_calibration import primary_calibration_config
from experiments.scenario_generation import load_existing_map_network


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def cli_module():
    spec = importlib.util.spec_from_file_location("manual_cli_under_test", ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _minimal_result(mode):
    return {
        "mode": mode, "seed": 33001, "scenario_id": "calibration-v1-c1", "status": "SAFETY_CAP_REACHED",
        "successful_rounds": 0, "fnd_round_zero_based": None, "fnd_round_one_based": None,
        "hnd_round_zero_based": None, "hnd_round_one_based": None,
        "lnd_round_zero_based": None, "lnd_round_one_based": None,
        "last_successful_round_zero_based": None, "last_successful_round_one_based": None,
        "routing_failure_round_zero_based": None, "routing_failure_round_one_based": None,
        "failure_reason": None, "final_live_nodes": 100, "final_dead_nodes": 0,
        "physical_energy": 0.0, "delivered_payloads": 0, "energy_per_delivered_payload": None,
        "last_success_details": None, "failure_details": None, "peak_rss_bytes": 123,
    }


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("direct", "direct"), ("mrp_pure", "mrp_pure"),
        ("mrp_hybrid", "mrp_hybrid"),
        ("direct_mrp_topk", "direct_mrp_topk"),
        ("mrp_hybrid_lifetime", "mrp_hybrid_lifetime"),
    ],
)
def test_cli_dispatches_exactly_one_requested_mode(cli_module, monkeypatch, tmp_path, requested, expected):
    observed = []

    def fake_runner(root, **kwargs):
        observed.append(kwargs["mode"])
        return _minimal_result(kwargs["mode"])

    monkeypatch.setattr(cli_module, "ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "run_manual_route_exhaustion", fake_runner)

    assert cli_module.main(["--mode", requested]) == 0
    assert observed == [expected]
    assert not (tmp_path / "final_evaluation_outputs").exists()


def test_cli_native_case_skips_event_scenario_and_uses_requested_output_root(
    cli_module, monkeypatch, tmp_path,
):
    observed = []

    def fake_runner(root, **kwargs):
        observed.append(kwargs)
        result = _minimal_result(kwargs["mode"])
        result["scenario_id"] = NATIVE_CASE_ID
        return result

    monkeypatch.setattr(cli_module, "ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "run_manual_route_exhaustion", fake_runner)
    output_root = tmp_path / "diagnostic_original_acaco_case_three_modes"

    assert cli_module.main([
        "--mode", "mrp_pure", "--native-case", "--output-directory", str(output_root),
    ]) == 0
    assert len(observed) == 1
    assert observed[0]["native_case"] is True
    assert (output_root / "mrp_pure" / f"mrp_pure_seed33001_{NATIVE_CASE_ID}.json").exists()


def test_cli_ttl_is_forwarded_as_runtime_only_override(cli_module, monkeypatch, tmp_path):
    observed = []

    def fake_runner(root, **kwargs):
        observed.append(kwargs["ttl_override"])
        result = _minimal_result(kwargs["mode"])
        result["ttl"] = kwargs["ttl_override"]
        return result

    monkeypatch.setattr(cli_module, "ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "run_manual_route_exhaustion", fake_runner)
    assert cli_module.main(["--mode", "mrp_hybrid_lifetime", "--ttl", "5"]) == 0
    assert observed == [5]


def test_cli_phase3_top_k_is_hybrid_only_and_forwarded(cli_module, monkeypatch, tmp_path):
    observed = []

    def fake_runner(root, **kwargs):
        observed.append(kwargs["phase3_top_k"])
        return _minimal_result(kwargs["mode"])

    monkeypatch.setattr(cli_module, "ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "run_manual_route_exhaustion", fake_runner)
    assert cli_module.main(["--mode", "mrp_hybrid", "--phase3-top-k", "3"]) == 0
    assert observed == [3]


def test_cli_phase3_top_k_is_forwarded_to_direct_mrp_topk(cli_module, monkeypatch, tmp_path):
    observed = []

    def fake_runner(root, **kwargs):
        observed.append((kwargs["mode"], kwargs["ttl_override"], kwargs["phase3_top_k"]))
        return _minimal_result(kwargs["mode"])

    monkeypatch.setattr(cli_module, "ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "run_manual_route_exhaustion", fake_runner)
    assert cli_module.main([
        "--mode", "direct_mrp_topk", "--ttl", "3", "--phase3-top-k", "3",
    ]) == 0
    assert observed == [("direct_mrp_topk", 3, 3)]


def test_cli_ant_energy_options_are_forwarded(cli_module, monkeypatch, tmp_path):
    observed = []

    def fake_runner(root, **kwargs):
        observed.append((kwargs["charge_ant_energy"], kwargs["ant_control_packet_bits"]))
        return _minimal_result(kwargs["mode"])

    monkeypatch.setattr(cli_module, "ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "run_manual_route_exhaustion", fake_runner)
    assert cli_module.main([
        "--mode", "mrp_hybrid", "--phase3-top-k", "3",
        "--charge-ant-energy", "--ant-control-packet-bits", "100",
    ]) == 0
    assert observed == [(True, 100)]


@pytest.mark.parametrize(
    ("charge", "bits", "message"),
    [
        (True, None, "required with --charge-ant-energy"),
        (False, 100, "requires --charge-ant-energy"),
        (True, 0, "positive integer"),
    ],
)
def test_ant_energy_request_is_validated_before_loading_artifacts(charge, bits, message):
    with pytest.raises(ManualRunnerError, match=message):
        run_manual_route_exhaustion(
            ROOT,
            mode="mrp_hybrid",
            charge_ant_energy=charge,
            ant_control_packet_bits=bits,
        )


def test_lifetime_selector_rejects_ant_energy_before_loading_artifacts():
    with pytest.raises(ManualRunnerError, match="not supported"):
        run_manual_route_exhaustion(
            ROOT,
            mode="mrp_hybrid_lifetime",
            charge_ant_energy=True,
            ant_control_packet_bits=100,
        )


def test_invalid_cli_ant_energy_does_not_create_round_output(cli_module, monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "ROOT", tmp_path)
    round_path = tmp_path / "must-not-exist.csv"

    with pytest.raises(SystemExit):
        cli_module.main([
            "--mode", "mrp_hybrid", "--charge-ant-energy",
            "--round-state-file", str(round_path),
        ])

    assert not round_path.exists()


@pytest.mark.parametrize("mode", ["direct", "mrp_pure", "mrp_event", "mrp_hybrid_lifetime"])
def test_non_hybrid_modes_reject_phase3_top_k(mode):
    with pytest.raises(ManualRunnerError, match="only to mrp_hybrid"):
        run_manual_route_exhaustion(ROOT, mode=mode, phase3_top_k=3)


def test_cli_prints_and_flushes_every_round_to_round_state_csv(
    cli_module, monkeypatch, tmp_path, capsys,
):
    def fake_runner(root, **kwargs):
        kwargs["progress"]({"round": 1, "alive_nodes": 99, "round_energy": 0.125})
        return _minimal_result(kwargs["mode"])

    monkeypatch.setattr(cli_module, "ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "run_manual_route_exhaustion", fake_runner)

    assert cli_module.main(["--mode", "direct"]) == 0
    rows = list(csv.DictReader((tmp_path / "round_state.csv").open(encoding="utf-8")))
    assert rows == [{"round": "1", "alive_nodes": "99", "round_energy": "0.125"}]
    assert "round,alive_nodes,round_energy\n1,99,0.125" in capsys.readouterr().out


def test_cli_can_rerun_without_overwriting_previous_json(cli_module, monkeypatch, tmp_path):
    monkeypatch.setattr(cli_module, "ROOT", tmp_path)
    monkeypatch.setattr(
        cli_module, "run_manual_route_exhaustion",
        lambda root, **kwargs: _minimal_result(kwargs["mode"]),
    )

    assert cli_module.main(["--mode", "direct"]) == 0
    assert cli_module.main(["--mode", "direct"]) == 0
    output_dir = tmp_path / "run_outputs" / "direct"
    assert (output_dir / "direct_seed33001.json").exists()
    assert (output_dir / "direct_seed33001_run2.json").exists()


def test_direct_rejects_ttl_override_before_loading_any_artifacts():
    with pytest.raises(ManualRunnerError, match="only to MRP-routing modes"):
        run_manual_route_exhaustion(ROOT, mode="direct", ttl_override=5)


def test_native_case_identity_is_canonical_and_uses_no_event_input():
    existing_map = load_existing_map_network(ROOT / "map.pkl")
    config = primary_calibration_config(existing_map)

    identity = native_case_identity(existing_map, config)
    scenario = build_native_round_context(existing_map, config.mrp_parameters.communication_radius)

    assert identity.canonical_sha256 == "fbfc3cb1eb57e2d72b6cf7eb69a05561e25ff641b2e540ce4597f39f56c5b428"
    assert identity.node_count == 100
    assert identity.target_cluster_head_count == 10
    assert scenario.scenario_id == NATIVE_CASE_ID
    assert scenario.event_nodes == ()
    assert dict(scenario.event_signal_strengths) == {}
    assert len(scenario.hop_counts) == 100


def test_process_memory_observation_is_non_mutating_and_available_on_windows():
    current, peak = process_memory_bytes()
    assert current is not None and current > 0
    assert peak is not None and peak >= current


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("mrp pure", "mrp_pure"), ("pure", "mrp_pure"),
        ("mrp hybrid", "mrp_hybrid"), ("hybrid", "mrp_hybrid"),
        ("mrp hybrid lifetime", "mrp_hybrid_lifetime"),
        ("lifetime", "mrp_hybrid_lifetime"),
        ("direct-mrp-topk", "direct_mrp_topk"),
        ("AC-ACO direct MRP TopK", "direct_mrp_topk"),
    ],
)
def test_mode_aliases_normalize(requested, expected):
    assert normalize_manual_mode(requested) == expected


def test_cli_pure_maps_to_network_wide_while_event_reference_remains_explicit():
    assert _EXPERIMENT_MODES["mrp_pure"] is ExperimentMode.MRP_NETWORK_WIDE
    assert _EXPERIMENT_MODES["mrp_event"] is ExperimentMode.PURE_MRP
    assert normalize_manual_mode("paper") == "mrp_event"


def test_experimental_lifetime_mode_reuses_hybrid_runtime_without_changing_frozen_modes():
    assert _EXPERIMENT_MODES["mrp_hybrid_lifetime"] is ExperimentMode.HYBRID
    assert {mode.value for mode in ExperimentMode} == {
        "BASELINE", "PURE_MRP", "MRP_NETWORK_WIDE", "HYBRID",
    }
    assert _EXPERIMENT_MODES["direct_mrp_topk"] is ExperimentMode.HYBRID


@pytest.mark.parametrize("top_k", [None, 0])
def test_direct_mrp_topk_rejects_missing_or_disabled_k_before_loading_artifacts(top_k):
    with pytest.raises(ManualRunnerError, match="requires --phase3-top-k greater than zero"):
        run_manual_route_exhaustion(ROOT, mode="direct_mrp_topk", phase3_top_k=top_k)


def test_direct_mrp_topk_rejects_zero_ttl_before_loading_artifacts():
    with pytest.raises(ManualRunnerError, match="--ttl greater than zero"):
        run_manual_route_exhaustion(
            ROOT, mode="direct_mrp_topk", ttl_override=0, phase3_top_k=3,
        )


def test_invalid_mode_fails_cleanly(cli_module, capsys):
    with pytest.raises(SystemExit) as error:
        cli_module.main(["--mode", "abc"])
    assert error.value.code == 2
    assert "Invalid mode." in capsys.readouterr().err


@dataclass
class _Row:
    round_index: int
    live_after: tuple[int, ...]
    dead_count: int
    cumulative_actual_energy: float
    delivered_payload_count: int
    residual_total: float
    residual_mean: float
    actual_round_energy: float | None = None


def test_fnd_is_recorded_without_terminating_route_exhaustion_runner():
    successful = SimpleNamespace(success=True, phase1_result=SimpleNamespace(selected_cluster_heads=(1,)), selected_routes=None)
    exhausted = SimpleNamespace(
        success=False, failure_stage="final_greedy", failure_reason="legacy final Greedy topology was unavailable",
        phase1_result=None,
    )
    rows = iter((
        (successful, _Row(0, (1, 2), 1, 0.5, 2, 9.5, 3.1666666667)),
        (exhausted, _Row(1, (1, 2), 1, 0.5, 0, 9.5, 3.1666666667)),
    ))
    summary = execute_route_exhaustion(
        canonical_mode="direct", experiment_mode=ExperimentMode.BASELINE,
        network=SimpleNamespace(nodes=(0, 1, 2)), scenario=object(), runtime=object(), config=object(),
        seeds=object(), seed=33001, scenario_id="calibration-v1-c1", anchor=49, safety_max_rounds=10,
        round_executor=lambda *_: next(rows),
    )
    assert summary["fnd_round_zero_based"] == 0
    assert summary["successful_rounds"] == 1
    assert summary["routing_failure_round_zero_based"] == 1
    assert summary["status"] == "GREEDY_ROUTING_FAILURE"


def test_transient_mrp_search_failure_retries_same_round(monkeypatch):
    from experiments import manual_runner as runner_module

    failed = SimpleNamespace(
        success=False, failure_stage="mrp_routing",
        failure_reason="MRP_SEARCH_FAILURE: CH 99: no unique successful routes",
        routing_failure_classification="MRP_SEARCH_FAILURE", ch_routing_results=(),
        failed_phase2_result=None,
    )
    successful = SimpleNamespace(
        success=True,
        phase1_result=SimpleNamespace(selected_cluster_heads=(99,)),
        selected_routes={99: (99, -1)}, ch_routing_results=(),
        failed_phase2_result=None,
    )
    row = _Row(0, (0, 99), 0, 0.25, 2, 1.75, 0.875, 0.25)
    attempts = iter(((failed, row), (successful, row)))
    progress = []
    monkeypatch.setattr(runner_module, "_current_scenario", lambda *_args: object())

    summary = execute_route_exhaustion(
        canonical_mode="mrp_pure", experiment_mode=ExperimentMode.MRP_NETWORK_WIDE,
        network=SimpleNamespace(nodes=(0, 99), base_dists=(1.0, 1.0)),
        scenario=object(), runtime=object(),
        config=SimpleNamespace(mrp_parameters=SimpleNamespace()), seeds=object(),
        seed=33001, scenario_id=NATIVE_CASE_ID, anchor=None, safety_max_rounds=1,
        mrp_search_retries=1, progress=progress.append,
        round_executor=lambda *_args: next(attempts),
    )

    assert summary["successful_rounds"] == 1
    assert summary["mrp_search_retries"] == 1
    assert summary["mrp_search_failures"] == 1
    assert progress[0]["round_energy"] == 0.25


def test_retry_energy_is_retained_when_terminal_attempt_transmits_no_ant(monkeypatch):
    from experiments import manual_runner as runner_module

    charged = AntControlEnergy(
        (0.1, 0.0), sant_tx_count=1, sant_energy=0.1,
    )
    failed_charged = SimpleNamespace(
        success=False, failure_stage="mrp_routing", failure_reason="search miss",
        routing_failure_classification="MRP_SEARCH_FAILURE", ch_routing_results=(),
        failed_phase2_result=None, ant_control_energy=charged,
        phase1_result=SimpleNamespace(selected_cluster_heads=()),
    )
    failed_zero = SimpleNamespace(
        success=False, failure_stage="mrp_routing", failure_reason="search miss",
        routing_failure_classification="MRP_SEARCH_FAILURE", ch_routing_results=(),
        failed_phase2_result=None,
        ant_control_energy=zero_ant_control_energy(2),
        phase1_result=SimpleNamespace(selected_cluster_heads=()),
    )
    attempts = iter((
        (failed_charged, _Row(0, (0, 1), 0, 0.1, 0, 1.9, 0.95, 0.1)),
        (failed_zero, _Row(0, (0, 1), 0, 0.1, 0, 1.9, 0.95, 0.0)),
    ))
    monkeypatch.setattr(runner_module, "_current_scenario", lambda *_args: object())

    summary = execute_route_exhaustion(
        canonical_mode="mrp_pure", experiment_mode=ExperimentMode.MRP_NETWORK_WIDE,
        network=SimpleNamespace(nodes=(0, 1), base_dists=(1.0, 1.0)),
        scenario=object(), runtime=object(),
        config=SimpleNamespace(mrp_parameters=SimpleNamespace()), seeds=object(),
        seed=33001, scenario_id=NATIVE_CASE_ID, anchor=None, safety_max_rounds=1,
        mrp_search_retries=1, round_executor=lambda *_args: next(attempts),
    )

    assert summary["physical_energy"] == pytest.approx(0.1)
    assert summary["total_ant_control_energy"] == pytest.approx(0.1)
    assert summary["data_energy"] == 0.0
    assert summary["total_physical_energy"] == pytest.approx(
        summary["data_energy"] + summary["total_ant_control_energy"]
    )


def test_retry_rebuilds_scenario_after_failed_attempt_changes_state(monkeypatch):
    from experiments import manual_runner as runner_module

    failed = SimpleNamespace(
        success=False, failure_stage="mrp_routing", failure_reason="search miss",
        routing_failure_classification="MRP_SEARCH_FAILURE", ch_routing_results=(),
        failed_phase2_result=None,
    )
    successful = SimpleNamespace(
        success=True, phase1_result=SimpleNamespace(selected_cluster_heads=(1,)),
        selected_routes={1: (1, -1)}, ch_routing_results=(),
        failed_phase2_result=None,
    )
    runtime = SimpleNamespace(state="before", rngs=())
    scenario_states = []
    results = iter((failed, successful))

    def current_scenario(_mode, _scenario, current_runtime, *_args):
        scenario_states.append(current_runtime.state)
        return object()

    def executor(*_args):
        result = next(results)
        if not result.success:
            runtime.state = "after-charged-failure"
        return result, _Row(0, (0, 1), 0, 0.1, 2, 1.9, 0.95, 0.1)

    monkeypatch.setattr(runner_module, "_current_scenario", current_scenario)
    execute_route_exhaustion(
        canonical_mode="mrp_pure", experiment_mode=ExperimentMode.MRP_NETWORK_WIDE,
        network=SimpleNamespace(nodes=(0, 1), base_dists=(1.0, 1.0)),
        scenario=object(), runtime=runtime,
        config=SimpleNamespace(mrp_parameters=SimpleNamespace()), seeds=object(),
        seed=33001, scenario_id=NATIVE_CASE_ID, anchor=None, safety_max_rounds=1,
        mrp_search_retries=1, round_executor=executor,
    )

    assert scenario_states == ["before", "after-charged-failure"]


def test_hybrid_retry_preserves_ac_aco_rng_but_advances_mrp_search_rng(monkeypatch):
    from experiments import manual_runner as runner_module

    failed = SimpleNamespace(
        success=False, failure_stage="mrp_routing", failure_reason="search miss",
        routing_failure_classification="MRP_SEARCH_FAILURE", ch_routing_results=(),
        failed_phase2_result=None,
    )
    successful = SimpleNamespace(
        success=True, phase1_result=SimpleNamespace(selected_cluster_heads=(7,)),
        selected_routes={7: (7, -1)}, ch_routing_results=(), failed_phase2_result=None,
    )
    row = _Row(0, (0, 7), 0, 0.25, 2, 1.75, 0.875, 0.25)
    runtime = SimpleNamespace(rngs=(random.Random(10), random.Random(20)))
    draws = []
    results = iter((failed, successful))

    def executor(*_args):
        draws.append((runtime.rngs[0].random(), runtime.rngs[1].random()))
        return next(results), row

    monkeypatch.setattr(runner_module, "_current_scenario", lambda *_args: object())
    execute_route_exhaustion(
        canonical_mode="mrp_hybrid", experiment_mode=ExperimentMode.HYBRID,
        network=SimpleNamespace(nodes=(0, 7), base_dists=(1.0, 1.0)),
        scenario=object(), runtime=runtime,
        config=SimpleNamespace(mrp_parameters=SimpleNamespace()), seeds=object(),
        seed=33001, scenario_id=NATIVE_CASE_ID, anchor=None, safety_max_rounds=1,
        mrp_search_retries=1, round_executor=executor,
    )

    assert draws[0][0] == draws[1][0]
    assert draws[0][1] != draws[1][1]


def test_hybrid_summary_reports_top_k_pruning_diagnostics(monkeypatch):
    from experiments import manual_runner as runner_module

    phase2 = SimpleNamespace(
        num_sants_requested=20, num_sants_executed=20, num_sants_succeeded=18,
        num_unique_routes=5, ant_results=(),
    )
    phase3 = SimpleNamespace(candidates=(object(), object(), object()), selected_route=(7, -1))
    routed = SimpleNamespace(cluster_head=7, phase2_result=phase2, phase3_result=phase3)
    raw = SimpleNamespace(
        success=True, phase1_result=SimpleNamespace(selected_cluster_heads=(7,)),
        selected_routes={7: (7, -1)}, ch_routing_results=(routed,),
        failed_phase2_result=None, lifetime_selection=None,
    )
    row = _Row(0, (0, 7), 0, 0.25, 2, 1.75, 0.875, 0.25)
    monkeypatch.setattr(runner_module, "_current_scenario", lambda *_args: object())

    summary = execute_route_exhaustion(
        canonical_mode="mrp_hybrid", experiment_mode=ExperimentMode.HYBRID,
        network=SimpleNamespace(nodes=(0, 7), base_dists=(1.0, 1.0)),
        scenario=object(), runtime=SimpleNamespace(rngs=(random.Random(1), random.Random(2))),
        config=SimpleNamespace(mrp_parameters=SimpleNamespace(communication_radius=2.0)),
        seeds=object(), seed=33001, scenario_id=NATIVE_CASE_ID, anchor=None,
        safety_max_rounds=1, phase3_top_k=3,
        round_executor=lambda *_args: (raw, row),
    )

    assert summary["configured_top_k"] == 3
    assert summary["discovered_routes_before_pruning"] == 5
    assert summary["candidate_routes_after_pruning"] == 3
    assert summary["pruned_routes_count"] == 2
    assert summary["mean_routes_before_pruning"] == 5.0
    assert summary["mean_routes_after_pruning"] == 3.0


def test_direct_mrp_topk_summary_proves_direct_chs_skip_discovery(monkeypatch):
    from experiments import manual_runner as runner_module

    phase2 = SimpleNamespace(
        num_sants_requested=4, num_sants_executed=4, num_sants_succeeded=3,
        num_unique_routes=5, ant_results=(), multipath_ready=True,
    )
    phase3 = SimpleNamespace(
        candidates=(object(), object(), object()), selected_route=(7, 8, -1),
        selected_index=0,
    )
    routed = SimpleNamespace(cluster_head=7, phase2_result=phase2, phase3_result=phase3)
    diagnostics = DirectMRPTopKDiagnostics(
        (2, 3), (7,), 1, 0, 5, 3, 2, 4, 3, 1,
    )
    raw = SimpleNamespace(
        success=True, phase1_result=SimpleNamespace(selected_cluster_heads=(2, 7, 3)),
        selected_routes={2: (2, -1), 7: (7, 8, -1), 3: (3, -1)},
        ch_routing_results=(routed,), failed_phase2_result=None,
        direct_mrp_diagnostics=diagnostics, lifetime_selection=None,
    )
    row = _Row(0, (2, 3, 7, 8), 0, 0.25, 4, 3.75, 0.9375, 0.25)
    monkeypatch.setattr(runner_module, "_current_scenario", lambda *_args: object())

    summary = execute_route_exhaustion(
        canonical_mode="direct_mrp_topk", experiment_mode=ExperimentMode.HYBRID,
        network=SimpleNamespace(nodes=(2, 3, 7, 8), base_dists=(1.0,) * 9),
        scenario=object(), runtime=SimpleNamespace(rngs=(random.Random(1), random.Random(2))),
        config=SimpleNamespace(mrp_parameters=SimpleNamespace(communication_radius=2.0, ttl=3)),
        seeds=object(), seed=33001, scenario_id=NATIVE_CASE_ID, anchor=None,
        safety_max_rounds=1, phase3_top_k=3,
        round_executor=lambda *_args: (raw, row),
    )

    assert summary["direct_ch_count"] == 2
    assert summary["fallback_ch_count"] == 1
    assert summary["direct_transmissions"] == 2
    assert summary["fallback_transmissions"] == 1
    assert summary["direct_ratio"] == pytest.approx(2 / 3)
    assert summary["fallback_ratio"] == pytest.approx(1 / 3)
    assert summary["mrp_discoveries"] == 1
    assert summary["cached_route_uses"] == 0
    assert summary["routes_before_topk"] == 5
    assert summary["routes_after_topk"] == 3
    assert summary["routes_pruned"] == 2
    assert summary["sant_count"] == 4
    assert summary["bant_count"] == 3
    assert summary["aant_count"] == 1
    assert summary["mean_selected_route_hops"] == pytest.approx(4 / 3)


def test_direct_mrp_topk_terminal_fallback_failure_counts_one_drop_after_retries(monkeypatch):
    from experiments import manual_runner as runner_module

    diagnostics = DirectMRPTopKDiagnostics(
        (2,), (7,), 1, 0, 0, 0, 0, 4, 0, 1,
    )
    raw = SimpleNamespace(
        success=False, failure_stage="mrp_routing",
        failure_reason="MRP_SEARCH_FAILURE: CH 7: no unique successful routes",
        routing_failure_classification="MRP_SEARCH_FAILURE",
        phase1_result=SimpleNamespace(selected_cluster_heads=(2, 7)),
        selected_routes=None, ch_routing_results=(), failed_phase2_result=None,
        failed_cluster_head=7, direct_mrp_diagnostics=diagnostics,
    )
    row = _Row(0, (2, 7), 0, 0.0, 0, 2.0, 1.0, None)
    monkeypatch.setattr(runner_module, "_current_scenario", lambda *_args: object())

    summary = execute_route_exhaustion(
        canonical_mode="direct_mrp_topk", experiment_mode=ExperimentMode.HYBRID,
        network=SimpleNamespace(nodes=(2, 7), base_dists=(1.0,) * 8),
        scenario=object(), runtime=SimpleNamespace(rngs=(random.Random(1), random.Random(2))),
        config=SimpleNamespace(mrp_parameters=SimpleNamespace(communication_radius=2.0, ttl=3)),
        seeds=object(), seed=33001, scenario_id=NATIVE_CASE_ID, anchor=None,
        safety_max_rounds=1, phase3_top_k=3, mrp_search_retries=0,
        round_executor=lambda *_args: (raw, row),
    )

    assert summary["fallback_failures"] == 1
    assert summary["routing_failures"] == 1
    assert summary["dropped_packets"] == 1
    assert summary["direct_transmissions"] == 0
    assert summary["fallback_transmissions"] == 0


def test_direct_mrp_topk_transient_search_miss_is_not_a_fallback_outcome_failure(monkeypatch):
    from experiments import manual_runner as runner_module

    diagnostics = DirectMRPTopKDiagnostics(
        (2,), (7,), 1, 0, 0, 0, 0, 4, 0, 1,
    )
    failed = SimpleNamespace(
        success=False, failure_stage="mrp_routing", failure_reason="search miss",
        routing_failure_classification="MRP_SEARCH_FAILURE",
        phase1_result=SimpleNamespace(selected_cluster_heads=(2, 7)),
        selected_routes=None, ch_routing_results=(), failed_phase2_result=None,
        failed_cluster_head=7, direct_mrp_diagnostics=diagnostics,
    )
    succeeded = SimpleNamespace(
        success=True, phase1_result=SimpleNamespace(selected_cluster_heads=(2, 7)),
        selected_routes={2: (2, -1), 7: (7, 6, -1)}, ch_routing_results=(),
        failed_phase2_result=None, failed_cluster_head=None,
        direct_mrp_diagnostics=diagnostics, lifetime_selection=None,
    )
    row = _Row(0, (2, 7), 0, 0.25, 2, 1.75, 0.875, 0.25)
    attempts = iter(((failed, row), (succeeded, row)))
    monkeypatch.setattr(runner_module, "_current_scenario", lambda *_args: object())

    summary = execute_route_exhaustion(
        canonical_mode="direct_mrp_topk", experiment_mode=ExperimentMode.HYBRID,
        network=SimpleNamespace(nodes=(2, 7), base_dists=(1.0,) * 8),
        scenario=object(), runtime=SimpleNamespace(rngs=(random.Random(1), random.Random(2))),
        config=SimpleNamespace(mrp_parameters=SimpleNamespace(communication_radius=2.0, ttl=3)),
        seeds=object(), seed=33001, scenario_id=NATIVE_CASE_ID, anchor=None,
        safety_max_rounds=1, phase3_top_k=3, mrp_search_retries=1,
        round_executor=lambda *_args: next(attempts),
    )

    assert summary["mrp_search_failures"] == 1
    assert summary["mrp_search_retries"] == 1
    assert summary["fallback_failures"] == 0
    assert summary["fallback_successes"] == 1
    assert summary["routing_failures"] == 0
    assert summary["dropped_packets"] == 0


def test_event_reason_prefix_is_retryable_without_classification_field():
    from experiments.manual_runner import _retryable_mrp_search_failure

    raw = SimpleNamespace(
        failure_reason="MRP_SEARCH_FAILURE: Phase II produced no unique routes"
    )
    assert _retryable_mrp_search_failure(raw)


def test_multiflow_failure_details_retain_failed_ch_phase2_counts():
    raw = SimpleNamespace(
        phase1_result=SimpleNamespace(selected_cluster_heads=(10, 20)),
        ch_routing_results=(), failed_cluster_head=10,
        failed_phase2_result=SimpleNamespace(
            status="completed", num_sants_requested=20, num_sants_succeeded=0,
            num_unique_routes=0, multipath_ready=False,
        ),
        routing_failure_classification="MRP_SEARCH_FAILURE",
        failure_reason="no unique successful routes",
    )

    details = _failure_details("mrp_pure", raw)

    assert details["failed_ch_ids"] == [10]
    assert details["failed_ch_phase2"] == {
        "status": "completed", "sants_requested": 20, "sants_successful": 0,
        "unique_routes": 0, "multipath_ready": False, "ttl_exhausted_sants": 0,
    }
