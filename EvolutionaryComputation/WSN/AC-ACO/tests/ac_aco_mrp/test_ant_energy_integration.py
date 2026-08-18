import importlib
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ac_aco_mrp import (
    DirectMRPTopKParameters, HybridParameters, run_direct_mrp_topk_round,
    run_hybrid_round,
)
from experiments.baseline_adapter import BaselineState, run_baseline_round
from mrp import PheromoneLifecycle


def _charged_mrp(hybrid_case):
    return replace(
        hybrid_case["parameters"].mrp,
        charge_ant_energy=True,
        ant_control_packet_bits=1,
    )


def test_direct_baseline_has_no_ant_energy(hybrid_case):
    result = run_baseline_round(
        hybrid_case["nodes"], hybrid_case["distances"],
        hybrid_case["base_dists"],
        BaselineState(
            hybrid_case["state"].physical_state,
            hybrid_case["state"].ac_aco_state,
        ),
        hybrid_case["ac_aco"], _charged_mrp(hybrid_case),
        random.Random(23), 0,
    )

    assert result.success
    assert not hasattr(result, "ant_control_energy")


def test_direct_first_all_direct_has_zero_ant_energy(hybrid_case, monkeypatch):
    runner = importlib.import_module("ac_aco_mrp.run_direct_mrp_topk")
    monkeypatch.setattr(
        runner, "select_ac_aco_cluster_heads",
        lambda *args: SimpleNamespace(
            selected_cluster_heads=(0, 1),
            state_after=hybrid_case["state"].ac_aco_state,
        ),
    )
    monkeypatch.setattr(
        runner, "route_cluster_heads_with_mrp",
        lambda *args, **kwargs: pytest.fail("all-direct must skip MRP"),
    )
    parameters = DirectMRPTopKParameters(
        hybrid_case["ac_aco"], _charged_mrp(hybrid_case), 3,
    )

    result = run_direct_mrp_topk_round(
        hybrid_case["nodes"], hybrid_case["distances"],
        hybrid_case["base_dists"], hybrid_case["state"],
        hybrid_case["context"], parameters, random.Random(23),
        random.Random(31), PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )

    assert result.success
    assert result.ant_control_energy.is_zero
    assert result.e_sum == pytest.approx(result.data_energy)


def test_hybrid_fallback_charges_ant_energy_and_total_identity(hybrid_case):
    parameters = HybridParameters(
        hybrid_case["ac_aco"], _charged_mrp(hybrid_case), phase3_top_k=3,
    )
    result = run_hybrid_round(
        hybrid_case["nodes"], hybrid_case["distances"],
        hybrid_case["base_dists"], hybrid_case["state"],
        hybrid_case["context"], parameters, random.Random(23),
        random.Random(31), PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )

    assert result.success and result.ant_control_energy.total_energy > 0
    assert result.e_sum == pytest.approx(
        result.data_energy + result.ant_control_energy.total_energy
    )
    assert result.state_after.physical_state.cumulative_actual_energy == pytest.approx(
        result.e_sum
    )


def test_hybrid_search_failure_keeps_transmitted_ant_energy(hybrid_case, monkeypatch):
    runner = importlib.import_module("ac_aco_mrp.run_hybrid")
    monkeypatch.setattr(
        runner, "select_ac_aco_cluster_heads",
        lambda *args: SimpleNamespace(
            selected_cluster_heads=(0,),
            state_after=hybrid_case["state"].ac_aco_state,
        ),
    )
    charged = replace(_charged_mrp(hybrid_case), ttl=1)
    result = run_hybrid_round(
        hybrid_case["nodes"], hybrid_case["distances"],
        (100.0,) * len(hybrid_case["nodes"]), hybrid_case["state"],
        hybrid_case["context"],
        HybridParameters(hybrid_case["ac_aco"], charged, phase3_top_k=3),
        random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )

    assert not result.success
    assert result.ant_control_energy.total_energy > 0
    assert result.state_after.physical_state.cumulative_actual_energy == pytest.approx(
        result.ant_control_energy.total_energy
    )


def test_direct_first_fallback_charges_only_fallback_ant_work(
    hybrid_case, monkeypatch,
):
    runner = importlib.import_module("ac_aco_mrp.run_direct_mrp_topk")
    monkeypatch.setattr(
        runner, "select_ac_aco_cluster_heads",
        lambda *args: SimpleNamespace(
            selected_cluster_heads=(0,),
            state_after=hybrid_case["state"].ac_aco_state,
        ),
    )
    result = run_direct_mrp_topk_round(
        hybrid_case["nodes"], hybrid_case["distances"],
        (100.0, 3.0, 3.0, 3.0), hybrid_case["state"],
        hybrid_case["context"],
        DirectMRPTopKParameters(
            hybrid_case["ac_aco"], _charged_mrp(hybrid_case), 3,
        ),
        random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )

    assert result.success
    assert result.direct_mrp_diagnostics.direct_cluster_heads == ()
    assert result.direct_mrp_diagnostics.fallback_cluster_heads == (0,)
    assert result.ant_control_energy.total_energy > 0


def test_direct_first_fallback_failure_keeps_transmitted_ant_energy(
    hybrid_case, monkeypatch,
):
    runner = importlib.import_module("ac_aco_mrp.run_direct_mrp_topk")
    monkeypatch.setattr(
        runner, "select_ac_aco_cluster_heads",
        lambda *args: SimpleNamespace(
            selected_cluster_heads=(0,),
            state_after=hybrid_case["state"].ac_aco_state,
        ),
    )
    charged = replace(_charged_mrp(hybrid_case), ttl=1)
    result = run_direct_mrp_topk_round(
        hybrid_case["nodes"], hybrid_case["distances"],
        (100.0,) * len(hybrid_case["nodes"]), hybrid_case["state"],
        hybrid_case["context"],
        DirectMRPTopKParameters(hybrid_case["ac_aco"], charged, 3),
        random.Random(23), random.Random(31),
        PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )

    assert not result.success
    assert result.ant_control_energy.total_energy > 0
    assert result.state_after.physical_state.cumulative_actual_energy == pytest.approx(
        result.ant_control_energy.total_energy
    )
