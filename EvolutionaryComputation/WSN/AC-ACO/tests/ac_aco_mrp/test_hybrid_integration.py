import inspect
import importlib
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ac_aco_mrp import HybridState, run_hybrid, run_hybrid_round
from mrp import PheromoneLifecycle


def test_actual_hybrid_uses_greedy_only_for_phase1_and_charges_final_multiflow_ledger_once(hybrid_case, monkeypatch):
    import ac_aco_mrp.phase1 as phase1
    runner = importlib.import_module("ac_aco_mrp.run_hybrid")
    legacy_calls, final_calls = [], []
    legacy = phase1.network_config
    final = runner.evaluate_hybrid_multiflow_energy
    monkeypatch.setattr(phase1, "network_config", lambda *args: (legacy_calls.append(1) or legacy(*args)))
    monkeypatch.setattr(runner, "evaluate_hybrid_multiflow_energy", lambda *args: (final_calls.append(1) or final(*args)))
    before = hybrid_case["state"]
    result = run_hybrid_round(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"], before,
        hybrid_case["context"], hybrid_case["parameters"], random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )
    assert result.success and len(legacy_calls) == len(result.phase1_result.candidates)
    assert len(final_calls) == 1 and result.final_plan is not None
    assert all(route.phase3_result.selected_route == result.selected_routes[route.cluster_head] for route in result.ch_routing_results)
    assert result.state_after.physical_state.residual_e == tuple(max(0, energy - spent) for energy, spent in zip(before.physical_state.residual_e, result.e_m_list))
    assert result.state_after.physical_state.cumulative_actual_energy == result.e_sum
    assert result.members_by_head and sum(map(len, result.members_by_head.values())) == 2


def test_mrp_is_per_ch_shared_link_state_and_ac_aco_pheromone_stays_separate(hybrid_case, monkeypatch):
    runner = importlib.import_module("ac_aco_mrp.run_hybrid")
    received = []
    def phase2(ch, *args):
        received.append((ch, args[-1]))
        state = {0: {1: 0.2}, 1: {0: 0.2}, 2: {0: 0.2}, 3: {0: 0.2}}
        state[ch][2] = 0.3 + ch
        return SimpleNamespace(status="completed", final_pheromone_state=state, start_cluster_head=ch)
    monkeypatch.setattr(runner, "discover_mrp_phase2_routes", phase2)
    monkeypatch.setattr(runner, "select_phase3_route", lambda phase2, *args: SimpleNamespace(selected_route=(phase2.start_cluster_head, -1)))
    result = run_hybrid_round(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"], hybrid_case["state"],
        hybrid_case["context"], hybrid_case["parameters"], random.Random(23), random.Random(31),
        PheromoneLifecycle.PERSIST_ACROSS_ROUNDS, 0,
    )
    assert result.success and len(result.ch_routing_results) == 2
    assert received[1][1][received[0][0]][2] == 0.3 + received[0][0]
    assert isinstance(result.state_after.physical_state.pheromone_state, dict)
    assert result.state_after.ac_aco_state.pheromone_matrix is not result.state_after.physical_state.pheromone_state


def test_different_parent_shared_relay_is_a_valid_multiflow_plan(hybrid_case, monkeypatch):
    runner = importlib.import_module("ac_aco_mrp.run_hybrid")
    phase1 = SimpleNamespace(selected_cluster_heads=(0, 1), state_after=hybrid_case["state"].ac_aco_state)
    routes = {0: (0, 2, -1), 1: (1, 2, 3, -1)}
    monkeypatch.setattr(runner, "select_ac_aco_cluster_heads", lambda *args: phase1)
    monkeypatch.setattr(runner, "discover_mrp_phase2_routes", lambda ch, *args: SimpleNamespace(status="completed", final_pheromone_state={i: {} for i in range(4)}, start_cluster_head=ch))
    monkeypatch.setattr(runner, "select_phase3_route", lambda phase2, *args: SimpleNamespace(selected_route=routes[phase2.start_cluster_head]))
    result = run_hybrid_round(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"], hybrid_case["state"],
        hybrid_case["context"], hybrid_case["parameters"], random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )
    assert result.success
    assert result.selected_routes == {0: (0, 2, -1), 1: (1, 2, 3, -1)}
    assert result.final_plan is not None
    assert result.e_m_list[2] > 0


def test_converging_routes_share_one_physical_member_relay(hybrid_case, monkeypatch):
    runner = importlib.import_module("ac_aco_mrp.run_hybrid")
    phase1 = SimpleNamespace(selected_cluster_heads=(0, 2), state_after=hybrid_case["state"].ac_aco_state)
    routes = {0: (0, 1, -1), 2: (2, 1, -1)}
    monkeypatch.setattr(runner, "select_ac_aco_cluster_heads", lambda *args: phase1)
    monkeypatch.setattr(runner, "discover_mrp_phase2_routes", lambda ch, *args: SimpleNamespace(status="completed", final_pheromone_state={i: {} for i in range(4)}, start_cluster_head=ch))
    monkeypatch.setattr(runner, "select_phase3_route", lambda phase2, *args: SimpleNamespace(selected_route=routes[phase2.start_cluster_head]))
    result = run_hybrid_round(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"], hybrid_case["state"],
        hybrid_case["context"], hybrid_case["parameters"], random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )
    assert result.success and result.final_topology is None
    assert result.final_plan is not None and result.final_plan.physical_sensor_ids.count(1) == 1


def test_phase2_fail_fast_is_atomic_and_rng_streams_must_be_separate(hybrid_case, monkeypatch):
    runner = importlib.import_module("ac_aco_mrp.run_hybrid")
    monkeypatch.setattr(runner, "discover_mrp_phase2_routes", lambda *args: SimpleNamespace(status="invalid_negative_pheromone_state", final_pheromone_state={}))
    before = hybrid_case["state"]
    result = run_hybrid_round(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"], before,
        hybrid_case["context"], hybrid_case["parameters"], random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )
    assert not result.success and result.failure_stage == "mrp_phase2" and result.state_after == before
    stream = random.Random(1)
    with pytest.raises(ValueError, match="separate"):
        run_hybrid_round(
            hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"], before,
            hybrid_case["context"], hybrid_case["parameters"], stream, stream,
            PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
        )


def test_multi_round_uses_updated_physical_state_and_only_allowed_boundaries(hybrid_case):
    result = run_hybrid(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"], hybrid_case["state"],
        (hybrid_case["context"], hybrid_case["context"]), hybrid_case["parameters"],
        random.Random(23), random.Random(31), PheromoneLifecycle.RESET_PER_DISCOVERY,
    )
    first, second = result.rounds
    source = inspect.getsource(__import__("ac_aco_mrp.run_hybrid", fromlist=["*"]))
    assert len(result.rounds) == 2 and second.state_before.physical_state.residual_e == first.state_after.physical_state.residual_e
    assert "evaluate_hybrid_multiflow_energy" in source
    assert all(token not in source for token in ("materialize_topology", "_route_parent_conflict", "make_path", "E_transmitting", "E_data_receiving"))


def test_hybrid_mrp_pheromone_policy_is_explicit_across_rounds(hybrid_case):
    contexts = (hybrid_case["context"], hybrid_case["context"])
    persisted = run_hybrid(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"], hybrid_case["state"],
        contexts, hybrid_case["parameters"], random.Random(23), random.Random(31),
        PheromoneLifecycle.PERSIST_ACROSS_ROUNDS,
    )
    first, second = persisted.rounds
    assert first.state_after.physical_state.pheromone_state is not None
    assert second.ch_routing_results[0].phase2_result.ant_results[0].sant_result.pheromone_state[0][1] == first.state_after.physical_state.pheromone_state[0][1]
    reset = run_hybrid(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"], hybrid_case["state"],
        contexts, hybrid_case["parameters"], random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY,
    )
    assert reset.rounds[0].state_after.physical_state.pheromone_state is None


def test_hybrid_only_forwards_configured_top_k_to_phase3(hybrid_case, monkeypatch):
    runner = importlib.import_module("ac_aco_mrp.run_hybrid")
    original, observed = runner.select_phase3_route, []

    def tracked(*args, **kwargs):
        observed.append(kwargs.get("top_k"))
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "select_phase3_route", tracked)
    parameters = replace(hybrid_case["parameters"], phase3_top_k=3)
    result = run_hybrid_round(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"],
        hybrid_case["state"], hybrid_case["context"], parameters,
        random.Random(23), random.Random(31), PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )
    assert result.success and observed and set(observed) == {3}


def test_hybrid_top_k_zero_is_exactly_disabled(hybrid_case):
    args = (
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"],
        hybrid_case["state"], hybrid_case["context"],
    )
    original = run_hybrid_round(
        *args, hybrid_case["parameters"], random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )
    disabled = run_hybrid_round(
        *args, replace(hybrid_case["parameters"], phase3_top_k=0),
        random.Random(23), random.Random(31), PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )
    assert original.selected_routes == disabled.selected_routes
    assert original.ch_routing_results == disabled.ch_routing_results
    assert original.e_sum == disabled.e_sum
    assert original.e_m_list == disabled.e_m_list
    assert original.state_after.physical_state == disabled.state_after.physical_state
    assert original.phase1_result.selected_cluster_heads == disabled.phase1_result.selected_cluster_heads
