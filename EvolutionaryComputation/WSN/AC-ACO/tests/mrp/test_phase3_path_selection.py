from copy import deepcopy
import inspect
from pathlib import Path
import random
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrp import (
    HeuristicBounds, MRPConfig, MRPPhase2AntResult, MRPPhase2Result, MRPPhase3InputError,
    RouteQuality, SANTRouteResult, calculate_phase3_fitness, select_phase3_route,
    discover_mrp_phase2_routes,
)
from mrp.phase3 import path_selection as selection_module


class FixedRNG:
    def __init__(self, draw=0.0): self.draw = draw
    def random(self): return self.draw


def _quality(path=(0, 1, -1), f1=4.0, f2=9.0, f3=16.0):
    return RouteQuality(path, f1, f2, f3, 1.0)


def _phase2(qualities, *, unique_routes=None, status="completed", pheromone=None):
    ants = tuple(
        MRPPhase2AntResult(
            index,
            SANTRouteResult(quality.path, True, quality.path[:-1], 0, {}, ()),
            quality, None, None, None, quality.quality, "first_route_best_initialized",
        )
        for index, quality in enumerate(qualities)
    )
    routes = tuple(unique_routes if unique_routes is not None else (quality.path for quality in qualities))
    return MRPPhase2Result(
        0, ants, ants, (), routes, pheromone if pheromone is not None else {0: {1: 0.01}},
        max((quality.quality for quality in qualities), default=None), len(ants), len(ants), len(ants),
        len(routes), len(routes) >= 2, status, None,
    )


def test_exact_equation_38_fitness_and_single_route_probability():
    quality = _quality()
    result = select_phase3_route(_phase2([quality]), MRPConfig(), FixedRNG())
    expected = 4.0 ** 0.5 + 1 / (9.0 ** 0.3) + 1 / (16.0 ** 0.2)
    assert result.candidates[0].phase3_fitness == pytest.approx(expected)
    assert result.candidates[0].probability == result.probability_sum == 1.0
    assert result.selected_route == quality.path and result.selected_index == 0 and result.random_draw is None
    assert not result.source_phase2_multipath_ready


def test_phase3_fitness_increases_with_minimum_energy_and_decreases_with_cost_or_length():
    baseline = calculate_phase3_fitness(_quality(), MRPConfig())
    assert calculate_phase3_fitness(_quality(f1=9.0), MRPConfig()) > baseline
    assert calculate_phase3_fitness(_quality(f2=16.0), MRPConfig()) < baseline
    assert calculate_phase3_fitness(_quality(f3=25.0), MRPConfig()) < baseline


def test_phase3_fitness_does_not_reuse_phase2_equation_33_quality():
    first = _quality()
    second = RouteQuality(first.path, first.f1, first.f2, first.f3, 999.0)
    assert calculate_phase3_fitness(first, MRPConfig()) == calculate_phase3_fitness(second, MRPConfig())


@pytest.mark.parametrize("config", [MRPConfig(k10=0.5, k11=0.3, k12=0.3), MRPConfig(k10=-0.1, k11=0.6, k12=0.5), MRPConfig(k10=True)])
def test_k10_k11_k12_are_validated_without_normalization(config):
    with pytest.raises(MRPPhase3InputError, match="k10"):
        select_phase3_route(_phase2([_quality()]), config, FixedRNG())


def test_exact_probabilities_are_positive_and_sum_to_one():
    first, second = _quality(), _quality((0, 2, -1), f1=1.0)
    result = select_phase3_route(_phase2([first, second]), MRPConfig(), FixedRNG())
    first_fitness = calculate_phase3_fitness(first, MRPConfig())
    second_fitness = calculate_phase3_fitness(second, MRPConfig())
    assert [candidate.probability for candidate in result.candidates] == pytest.approx([
        first_fitness / (first_fitness + second_fitness), second_fitness / (first_fitness + second_fitness),
    ])
    assert result.probability_sum == pytest.approx(1.0)
    assert all(candidate.probability > 0 for candidate in result.candidates)


def test_sampling_is_not_argmax_when_high_draw_selects_lower_fitness_route():
    first, second = _quality(), _quality((0, 2, -1), f1=1.0)
    result = select_phase3_route(_phase2([first, second]), MRPConfig(), FixedRNG(0.999))
    assert result.candidates[0].phase3_fitness > result.candidates[1].phase3_fitness
    assert result.selected_index == 1 and result.selected_route == second.path


def test_same_seed_is_reproducible_and_controlled_draws_select_different_routes():
    phase2 = _phase2([_quality(), _quality((0, 2, -1), f1=1.0)])
    assert select_phase3_route(phase2, MRPConfig(), random.Random(3)) == select_phase3_route(phase2, MRPConfig(), random.Random(3))
    assert select_phase3_route(phase2, MRPConfig(), FixedRNG(0.0)).selected_index == 0
    assert select_phase3_route(phase2, MRPConfig(), FixedRNG(0.999)).selected_index == 1


def test_top_k_retains_highest_fitness_and_renormalizes_probabilities(monkeypatch):
    qualities = tuple(
        RouteQuality((0, node, -1), 1.0, 1.0, 1.0, score)
        for node, score in zip(range(1, 6), (10.0, 8.0, 2.0, 1.0, 0.5))
    )
    monkeypatch.setattr(
        selection_module, "calculate_phase3_fitness",
        lambda quality, _config: quality.quality,
    )

    result = select_phase3_route(_phase2(qualities), MRPConfig(), FixedRNG(0.999), top_k=3)

    assert [candidate.route for candidate in result.candidates] == [
        (0, 1, -1), (0, 2, -1), (0, 3, -1),
    ]
    assert [candidate.probability for candidate in result.candidates] == pytest.approx(
        [0.5, 0.4, 0.1]
    )
    assert result.selected_route == (0, 3, -1)
    assert all(candidate.route not in {(0, 4, -1), (0, 5, -1)} for candidate in result.candidates)


def test_top_k_larger_than_route_count_keeps_every_route(monkeypatch):
    qualities = (
        RouteQuality((0, 1, -1), 1.0, 1.0, 1.0, 2.0),
        RouteQuality((0, 2, -1), 1.0, 1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(
        selection_module, "calculate_phase3_fitness",
        lambda quality, _config: quality.quality,
    )
    result = select_phase3_route(_phase2(qualities), MRPConfig(), FixedRNG(), top_k=5)
    assert len(result.candidates) == 2


def test_top_k_equal_fitness_boundary_preserves_phase2_discovery_order(monkeypatch):
    qualities = tuple(
        RouteQuality((0, node, -1), 1.0, 1.0, 1.0, 5.0)
        for node in (3, 1, 2)
    )
    monkeypatch.setattr(
        selection_module, "calculate_phase3_fitness", lambda quality, _config: quality.quality,
    )
    result = select_phase3_route(_phase2(qualities), MRPConfig(), FixedRNG(), top_k=2)
    assert [candidate.route for candidate in result.candidates] == [(0, 3, -1), (0, 1, -1)]


def test_disabled_top_k_is_exactly_the_original_all_route_behavior():
    phase2 = _phase2([_quality(), _quality((0, 2, -1), f1=1.0)])
    implicit = select_phase3_route(phase2, MRPConfig(), FixedRNG(0.999))
    explicit_none = select_phase3_route(phase2, MRPConfig(), FixedRNG(0.999), top_k=None)
    explicit_zero = select_phase3_route(phase2, MRPConfig(), FixedRNG(0.999), top_k=0)
    assert implicit == explicit_none == explicit_zero


@pytest.mark.parametrize("top_k", [-1, True, 1.5])
def test_invalid_top_k_is_rejected(top_k):
    with pytest.raises(MRPPhase3InputError, match="top_k"):
        select_phase3_route(_phase2([_quality()]), MRPConfig(), FixedRNG(), top_k=top_k)


def test_duplicate_raw_ant_records_produce_one_candidate_with_its_first_quality_object():
    first, duplicate = _quality(), _quality(f1=1.0)
    result = select_phase3_route(_phase2([first, duplicate], unique_routes=((0, 1, -1),)), MRPConfig(), FixedRNG())
    assert len(result.candidates) == 1 and result.candidates[0].route_quality is first
    assert result.candidates[0].route == result.candidates[0].route_quality.path


def test_selector_consumes_a_real_phase2_unique_route_result():
    phase2 = discover_mrp_phase2_routes(
        0, {0, 1, 2}, [10.0, 8.0, 8.0], [[0.0, 3.0, 3.0], [3.0, 0.0, 6.0], [3.0, 6.0, 0.0]],
        [7.0, 3.0, 3.0], {0: 2, 1: 1, 2: 1}, 5.0, HeuristicBounds(0.1, 1000.0, 0.1, 1000.0),
        MRPConfig(aant_probability=0.0), 0.1, 2, 2, FixedRNG(0.5), 1.0, 2.0, 1.0, 1.0, 10.0, 1.0, 0.0, 0.01,
    )
    result = select_phase3_route(phase2, MRPConfig(), FixedRNG(0.0))
    assert result.selected_route == phase2.unique_routes[0]
    assert result.candidates[0].route_quality is phase2.successful_ant_results[0].route_quality


def test_empty_failed_or_unmatched_phase2_results_fail_explicitly():
    with pytest.raises(MRPPhase3InputError, match="no unique"):
        select_phase3_route(_phase2([]), MRPConfig(), FixedRNG())
    with pytest.raises(MRPPhase3InputError, match="did not complete"):
        select_phase3_route(_phase2([_quality()], status="invalid_negative_pheromone_state"), MRPConfig(), FixedRNG())
    with pytest.raises(MRPPhase3InputError, match="lacks a successful"):
        select_phase3_route(_phase2([_quality()], unique_routes=((0, 2, -1),)), MRPConfig(), FixedRNG())


@pytest.mark.parametrize("field,value", [("f1", 0.0), ("f2", float("nan")), ("f3", float("inf")), ("f1", True)])
def test_invalid_phase2_route_metrics_are_rejected_without_epsilon(field, value):
    values = {"f1": 4.0, "f2": 9.0, "f3": 16.0}
    values[field] = value
    with pytest.raises(MRPPhase3InputError):
        select_phase3_route(_phase2([_quality(**values)]), MRPConfig(), FixedRNG())


def test_rng_and_probability_domain_errors_are_explicit():
    phase2 = _phase2([_quality(), _quality((0, 2, -1))])
    with pytest.raises(MRPPhase3InputError, match="RNG"):
        select_phase3_route(phase2, MRPConfig(), object())
    with pytest.raises(MRPPhase3InputError, match=r"in \[0, 1\)"):
        select_phase3_route(phase2, MRPConfig(), FixedRNG(1.0))


def test_selection_does_not_mutate_phase2_pheromone_or_use_simulator_lifecycle_code():
    phase2 = _phase2([_quality(), _quality((0, 2, -1))], pheromone={0: {1: 0.01, 2: 0.01}})
    before = deepcopy(phase2.final_pheromone_state)
    select_phase3_route(phase2, MRPConfig(), FixedRNG())
    assert phase2.final_pheromone_state == before
    source = inspect.getsource(selection_module)
    assert "sant_result.trace" not in source
    assert all(name not in source for name in (
        "energy_consumption", "materialize_topology", "E_transmitting", "AC_ACO",
        "network_config", "construct_normal_sant_route", "apply_bant_global_feedback", "residual_e",
    ))
