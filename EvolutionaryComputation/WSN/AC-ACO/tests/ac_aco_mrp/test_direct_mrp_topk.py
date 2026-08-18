import importlib
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ac_aco_mrp import (
    DirectMRPTopKParameters,
    run_direct_mrp_topk_round,
)
from experiments.baseline_adapter import BaselineState, run_baseline_round
from mrp import MRPRoutingFailure, PheromoneLifecycle
from mrp.multi_ch_routing import MRPMultiCHRoutingResult
from mrp.runner_types import MRPCHRoutingResult


def _parameters(hybrid_case, top_k=3):
    return DirectMRPTopKParameters(
        hybrid_case["ac_aco"], hybrid_case["parameters"].mrp, top_k,
    )


def _phase1(hybrid_case, heads):
    return SimpleNamespace(
        selected_cluster_heads=tuple(heads),
        state_after=hybrid_case["state"].ac_aco_state,
    )


def _phase2(head, unique_routes=5, ants=2):
    ant_results = tuple(
        SimpleNamespace(
            bant_preparation=object(),
            sant_result=SimpleNamespace(
                trace=(
                    (SimpleNamespace(mode="AANT"), SimpleNamespace(mode="AANT"))
                    if index == 0 else (SimpleNamespace(mode="NORMAL_SANT"),)
                ),
            ),
        )
        for index in range(ants)
    )
    return SimpleNamespace(
        start_cluster_head=head,
        num_unique_routes=unique_routes,
        multipath_ready=unique_routes >= 2,
        num_sants_executed=ants,
        ant_results=ant_results,
    )


def _routing_result(routes, before=5, after=3):
    routed = []
    for head, route in routes.items():
        candidates = tuple(object() for _ in range(after))
        phase3 = SimpleNamespace(
            selected_route=route,
            candidates=candidates,
            selected_index=0,
        )
        routed.append(MRPCHRoutingResult(head, _phase2(head, before), phase3))
    return MRPMultiCHRoutingResult(tuple(routed), routes, None, None)


def test_all_direct_skips_phase2_and_all_ant_work(hybrid_case, monkeypatch):
    runner = importlib.import_module("ac_aco_mrp.run_direct_mrp_topk")
    monkeypatch.setattr(
        runner,
        "select_ac_aco_cluster_heads",
        lambda *args: _phase1(hybrid_case, (0, 1, 2)),
    )
    monkeypatch.setattr(
        runner,
        "route_cluster_heads_with_mrp",
        lambda *args, **kwargs: pytest.fail("all-direct CHs must not call MRP"),
    )

    result = run_direct_mrp_topk_round(
        hybrid_case["nodes"],
        hybrid_case["distances"],
        hybrid_case["base_dists"],
        hybrid_case["state"],
        hybrid_case["context"],
        _parameters(hybrid_case),
        random.Random(23),
        random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY,
        0,
    )

    diagnostics = result.direct_mrp_diagnostics
    assert result.success
    assert result.selected_routes == {0: (0, -1), 1: (1, -1), 2: (2, -1)}
    assert diagnostics.direct_cluster_heads == (0, 1, 2)
    assert diagnostics.fallback_cluster_heads == ()
    assert diagnostics.mrp_discoveries == diagnostics.sant_count == 0
    assert diagnostics.bant_count == diagnostics.aant_count == 0
    assert diagnostics.routes_before_top_k == diagnostics.routes_after_top_k == 0


def test_exact_radius_uses_the_shared_mrp_less_than_or_equal_rule(hybrid_case, monkeypatch):
    runner = importlib.import_module("ac_aco_mrp.run_direct_mrp_topk")
    monkeypatch.setattr(
        runner,
        "select_ac_aco_cluster_heads",
        lambda *args: _phase1(hybrid_case, (0,)),
    )
    monkeypatch.setattr(
        runner,
        "route_cluster_heads_with_mrp",
        lambda *args, **kwargs: pytest.fail("d == R is direct under MRP reachability"),
    )
    result = run_direct_mrp_topk_round(
        hybrid_case["nodes"], hybrid_case["distances"], (50.0, 1.0, 1.0, 1.0),
        hybrid_case["state"], hybrid_case["context"], _parameters(hybrid_case),
        random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )

    assert result.success
    assert result.selected_routes == {0: (0, -1)}
    assert result.direct_mrp_diagnostics.direct_cluster_heads == (0,)


def test_mixed_mode_calls_mrp_only_for_fallback_and_forwards_existing_top_k(
    hybrid_case, monkeypatch,
):
    runner = importlib.import_module("ac_aco_mrp.run_direct_mrp_topk")
    monkeypatch.setattr(
        runner,
        "select_ac_aco_cluster_heads",
        lambda *args: _phase1(hybrid_case, (0, 1, 3)),
    )
    observed = []

    def route_fallback(heads, *args, **kwargs):
        observed.append((tuple(heads), kwargs["phase3_selector"].keywords["top_k"]))
        return _routing_result({1: (1, 2, -1)})

    monkeypatch.setattr(runner, "route_cluster_heads_with_mrp", route_fallback)
    base_dists = (1.0, 100.0, 1.0, 1.0)
    result = run_direct_mrp_topk_round(
        hybrid_case["nodes"], hybrid_case["distances"], base_dists,
        hybrid_case["state"], hybrid_case["context"], _parameters(hybrid_case, 3),
        random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )

    diagnostics = result.direct_mrp_diagnostics
    assert result.success and observed == [((1,), 3)]
    assert result.selected_routes == {0: (0, -1), 1: (1, 2, -1), 3: (3, -1)}
    assert diagnostics.direct_cluster_heads == (0, 3)
    assert diagnostics.fallback_cluster_heads == (1,)
    assert diagnostics.mrp_discoveries == 1
    assert diagnostics.routes_before_top_k == 5
    assert diagnostics.routes_after_top_k == 3
    assert diagnostics.routes_pruned == 2
    assert diagnostics.sant_count == 2
    assert diagnostics.bant_count == 2
    assert diagnostics.aant_count == 2


def test_all_fallback_routes_every_head_through_mrp(hybrid_case, monkeypatch):
    runner = importlib.import_module("ac_aco_mrp.run_direct_mrp_topk")
    monkeypatch.setattr(
        runner,
        "select_ac_aco_cluster_heads",
        lambda *args: _phase1(hybrid_case, (0, 1)),
    )
    observed = []

    def route_fallback(heads, *args, **kwargs):
        observed.append(tuple(heads))
        return _routing_result({0: (0, 2, -1), 1: (1, 3, -1)}, before=2, after=2)

    monkeypatch.setattr(runner, "route_cluster_heads_with_mrp", route_fallback)
    result = run_direct_mrp_topk_round(
        hybrid_case["nodes"], hybrid_case["distances"], (100.0, 100.0, 1.0, 1.0),
        hybrid_case["state"], hybrid_case["context"], _parameters(hybrid_case, 5),
        random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )

    assert result.success and observed == [(0, 1)]
    assert result.direct_mrp_diagnostics.direct_cluster_heads == ()
    assert result.direct_mrp_diagnostics.fallback_cluster_heads == (0, 1)
    assert result.direct_mrp_diagnostics.mrp_discoveries == 2
    assert result.direct_mrp_diagnostics.routes_pruned == 0


def test_fallback_failure_is_atomic_and_never_uses_long_range_direct(
    hybrid_case, monkeypatch,
):
    runner = importlib.import_module("ac_aco_mrp.run_direct_mrp_topk")
    monkeypatch.setattr(
        runner,
        "select_ac_aco_cluster_heads",
        lambda *args: _phase1(hybrid_case, (0, 1)),
    )
    failed_phase2 = _phase2(1, unique_routes=0, ants=2)

    def fail_fallback(*args, **kwargs):
        raise MRPRoutingFailure(
            "phase3", 1, "MRP_SEARCH_FAILURE", "no unique successful routes",
            phase2_result=failed_phase2,
        )

    monkeypatch.setattr(runner, "route_cluster_heads_with_mrp", fail_fallback)
    monkeypatch.setattr(
        runner,
        "evaluate_hybrid_multiflow_energy",
        lambda *args: pytest.fail("failed fallback must not commit physical energy"),
    )
    before = hybrid_case["state"]
    result = run_direct_mrp_topk_round(
        hybrid_case["nodes"], hybrid_case["distances"], (1.0, 100.0, 1.0, 1.0),
        before, hybrid_case["context"], _parameters(hybrid_case),
        random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )

    assert not result.success and result.state_after is before
    assert result.failed_cluster_head == 1
    assert result.selected_routes is None
    assert result.direct_mrp_diagnostics.direct_cluster_heads == (0,)
    assert result.direct_mrp_diagnostics.fallback_cluster_heads == (1,)
    assert result.direct_mrp_diagnostics.mrp_discoveries == 1


def test_all_direct_energy_matches_ac_aco_direct_baseline(hybrid_case):
    physical = hybrid_case["state"].physical_state
    phase1_state = hybrid_case["state"].ac_aco_state
    baseline = run_baseline_round(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"],
        BaselineState(physical, phase1_state), hybrid_case["ac_aco"],
        hybrid_case["parameters"].mrp, random.Random(23), 0,
    )
    adapted = run_direct_mrp_topk_round(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"],
        hybrid_case["state"], hybrid_case["context"], _parameters(hybrid_case),
        random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )

    assert baseline.success and adapted.success
    assert baseline.phase1_result.selected_cluster_heads == adapted.phase1_result.selected_cluster_heads
    assert adapted.e_m_list == pytest.approx(baseline.e_m_list)
    assert adapted.e_sum == pytest.approx(baseline.e_sum)


@pytest.mark.parametrize("top_k", [None, 0, -1, True, 1.5])
def test_direct_mrp_topk_requires_positive_integer_k(hybrid_case, top_k):
    parameters = DirectMRPTopKParameters(
        hybrid_case["ac_aco"], hybrid_case["parameters"].mrp, top_k,
    )
    with pytest.raises(ValueError, match="positive integer"):
        run_direct_mrp_topk_round(
            hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"],
            hybrid_case["state"], hybrid_case["context"], parameters,
            random.Random(23), random.Random(31),
            PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
        )


def test_direct_mrp_topk_requires_positive_ttl_at_core_boundary(hybrid_case):
    parameters = DirectMRPTopKParameters(
        hybrid_case["ac_aco"], replace(hybrid_case["parameters"].mrp, ttl=0), 3,
    )
    with pytest.raises(ValueError, match="TTL must be a positive integer"):
        run_direct_mrp_topk_round(
            hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"],
            hybrid_case["state"], hybrid_case["context"], parameters,
            random.Random(23), random.Random(31),
            PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
        )
