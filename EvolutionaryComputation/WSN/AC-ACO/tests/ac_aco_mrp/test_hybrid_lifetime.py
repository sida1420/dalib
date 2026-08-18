import importlib
from itertools import product
import math
import random
from types import SimpleNamespace

import pytest

from ac_aco_mrp.lifetime_selector import OBJECTIVE, select_lifetime_aware_routes
from ac_aco_mrp.multiflow import build_hybrid_multiflow_plan
from ac_aco_mrp.multiflow_energy import evaluate_hybrid_multiflow_energy
from mrp import PheromoneLifecycle


def _brute_force(heads, members, candidates, live, residual, case):
    best = None
    for routes in product(*(tuple(sorted(candidates[head])) for head in heads)):
        selected = dict(zip(heads, routes))
        plan = build_hybrid_multiflow_plan(heads, members, selected, live, len(residual))
        energy = evaluate_hybrid_multiflow_energy(
            plan, case["distances"], case["base_dists"], case["parameters"].mrp,
        )
        minimum = min(max(0.0, residual[i] - energy.e_m_list[i]) for i in live)
        hops = sum(len(route) - 1 for route in routes)
        score = (minimum, -energy.e_sum, -hops)
        value = (score, routes, energy)
        if best is None or score > best[0] or (score == best[0] and routes < best[1]):
            best = value
    return best


@pytest.mark.parametrize(
    ("heads", "members", "candidates"),
    [
        (
            (0, 1), {0: (2,), 1: (3,)},
            {0: ((0, -1), (0, 2, -1)), 1: ((1, -1), (1, 2, 3, -1))},
        ),
        (
            (0, 2), {0: (1,), 2: (3,)},
            {0: ((0, -1), (0, 1, -1)), 2: ((2, -1), (2, 1, -1))},
        ),
    ],
)
def test_branch_and_bound_is_exactly_equivalent_to_test_only_brute_force(
    hybrid_case, heads, members, candidates,
):
    live, residual = {0, 1, 2, 3}, (100.0,) * 4
    expected = _brute_force(heads, members, candidates, live, residual, hybrid_case)
    actual = select_lifetime_aware_routes(
        heads, members, candidates, live, residual, hybrid_case["distances"],
        hybrid_case["base_dists"], hybrid_case["parameters"].mrp,
    )

    assert OBJECTIVE == "MAXIMIZE_MINIMUM_POST_ROUND_RESIDUAL_ENERGY"
    assert tuple(actual.selected_routes[head] for head in heads) == expected[1]
    assert actual.minimum_residual_after_round == pytest.approx(expected[0][0])
    assert actual.energy.e_sum == pytest.approx(-expected[0][1])
    assert actual.total_hop_count == -expected[0][2]
    assert actual.theoretical_combinations == math.prod(len(candidates[head]) for head in heads)
    assert actual.candidate_sets_evaluated + actual.branches_pruned == actual.theoretical_combinations


def test_lifetime_objective_excludes_nodes_dead_before_round(hybrid_case):
    result = select_lifetime_aware_routes(
        (0,), {0: (1, 3)}, {0: ((0, -1),)}, {0, 1, 3},
        (100.0, 100.0, 0.0, 100.0), hybrid_case["distances"],
        hybrid_case["base_dists"], hybrid_case["parameters"].mrp,
    )

    assert result.minimum_residual_after_round > 0


def test_experimental_round_discovers_all_routes_then_commits_once(hybrid_case, monkeypatch):
    runner = importlib.import_module("ac_aco_mrp.run_hybrid_lifetime")
    phase1 = SimpleNamespace(
        selected_cluster_heads=(0, 1), state_after=hybrid_case["state"].ac_aco_state,
    )
    discoveries, selections, commits = [], [], []
    real_select, real_commit = runner.select_lifetime_aware_routes, runner.apply_baseline_lifecycle
    monkeypatch.setattr(runner, "select_ac_aco_cluster_heads", lambda *args: phase1)

    def discover(ch, *args):
        discoveries.append(ch)
        return SimpleNamespace(
            status="completed", final_pheromone_state=args[-1], unique_routes=((ch, -1),),
        )

    def select(*args, **kwargs):
        selections.append(tuple(args[0]))
        return real_select(*args, **kwargs)

    def commit(*args, **kwargs):
        commits.append(1)
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(runner, "discover_mrp_phase2_routes", discover)
    monkeypatch.setattr(runner, "select_lifetime_aware_routes", select)
    monkeypatch.setattr(runner, "apply_baseline_lifecycle", commit)
    result = runner.run_hybrid_lifetime_round(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"],
        hybrid_case["state"], hybrid_case["context"], hybrid_case["parameters"],
        random.Random(23), random.Random(31), PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )

    assert result.success
    assert discoveries == [0, 1]
    assert selections == [(0, 1)]
    assert len(result.selected_routes) == 2
    assert commits == [1]
    assert result.state_after.physical_state.cumulative_actual_energy == result.e_sum
