from copy import deepcopy
import inspect
from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrp import (
    HeuristicBounds, MRPConfig, RouteQuality, apply_bant_global_feedback,
    calculate_global_pheromone_delta, construct_normal_sant_route, evaluate_route_quality,
    prepare_bant,
)
from mrp.phase2 import bant_feedback as bant_module
from mrp.phase2.bant_feedback import BANTFeedbackInputError
from mrp.phase2 import pheromone as pheromone_module


class FixedRNG:
    def __init__(self, *draws): self.draws = iter(draws)
    def random(self): return next(self.draws)


def _preparation(quality=12.0):
    return prepare_bant(RouteQuality((0, 1, 2, -1), 5.0, 6.0, 7.0, quality))


def _state(value0=0.5, value1=0.25):
    return {0: {1: value0}, 1: {2: value1}, 2: {0: 0.75}}


def _feedback(**overrides):
    values = {
        "preparation": _preparation(), "previous_best_quality": 10.0,
        "pheromone_state": _state(), "config": MRPConfig(rho=0.2), "c": 2.0, "c1": 0.25,
    }
    values.update(overrides)
    return apply_bant_global_feedback(**values)


@pytest.mark.parametrize(
    ("current", "previous", "c", "c1", "expected"),
    [(12.0, 10.0, 2.0, 0.25, 3.4), (8.0, 10.0, 2.0, 0.0, -0.4), (10.0, 10.0, 2.0, 0.25, 2.5)],
)
def test_global_delta_matches_raw_equation_32(current, previous, c, c1, expected):
    assert calculate_global_pheromone_delta(current, previous, c, c1) == pytest.approx(expected)


def test_bant_updates_directed_forward_links_in_reverse_traversal_order():
    result = _feedback()
    assert result.global_delta_tau == pytest.approx(3.4)
    assert [(step.reverse_link, step.forward_link) for step in result.trace] == [
        ((2, 1), (1, 2)), ((1, 0), (0, 1)),
    ]
    assert [step.tau_after for step in result.trace] == pytest.approx([0.88, 1.08])
    assert result.pheromone_state[1][2] == pytest.approx(0.88)
    assert result.pheromone_state[0][1] == pytest.approx(1.08)
    assert result.pheromone_state[2][0] == 0.75
    assert result.transition_compatible


def test_bant_does_not_update_the_opposite_directed_link():
    state = {0: {1: 0.5}, 1: {2: 0.25, 0: 0.9}, 2: {1: 0.8}}
    result = _feedback(pheromone_state=state)
    assert result.pheromone_state[2][1] == 0.8
    assert result.pheromone_state[1][0] == 0.9


def test_eq32_uses_previous_best_before_eq37_updates_it():
    result = _feedback(c=10.0, c1=0.0)
    assert result.global_delta_tau == pytest.approx(2.0)
    assert result.previous_best_quality == 10.0
    assert result.updated_best_quality == 12.0


@pytest.mark.parametrize(("quality", "expected_best"), [(12.0, 12.0), (8.0, 10.0)])
def test_eq37_updates_best_only_after_feedback(quality, expected_best):
    result = _feedback(preparation=_preparation(quality), c=1.0, c1=0.0)
    assert result.updated_best_quality == expected_best


@pytest.mark.parametrize("previous_best", [0.0, -1.0, float("nan"), True])
def test_first_route_bootstrap_and_invalid_previous_best_fail_explicitly(previous_best):
    with pytest.raises(BANTFeedbackInputError, match="previous best"):
        _feedback(previous_best_quality=previous_best)


@pytest.mark.parametrize(("name", "value"), [("c", float("nan")), ("c", True), ("c1", float("inf")), ("c1", True)])
def test_c_and_c1_are_required_finite_caller_coefficients(name, value):
    with pytest.raises(BANTFeedbackInputError, match=name):
        _feedback(**{name: value})


def test_bant_reuses_shared_eq30_mixer(monkeypatch):
    calls = []
    def mix(old_tau, delta_tau, config):
        calls.append((old_tau, delta_tau, config.rho))
        return old_tau + delta_tau
    monkeypatch.setattr(bant_module, "update_pheromone_with_deposit", mix)
    result = _feedback()
    assert calls == [(0.25, 3.4, 0.2), (0.5, 3.4, 0.2)]
    assert [step.tau_after for step in result.trace] == pytest.approx([3.65, 3.9])


def test_local_sant_and_bant_share_one_eq30_implementation():
    source = inspect.getsource(pheromone_module.update_local_pheromone)
    assert "update_pheromone_with_deposit" in source


def test_negative_feedback_and_negative_tau_are_returned_without_clamping():
    result = _feedback(preparation=_preparation(1.0), c=2.0, c1=0.0, pheromone_state=_state(0.1, 0.1))
    assert result.global_delta_tau == pytest.approx(-1.8)
    assert [step.tau_after for step in result.trace] == pytest.approx([-0.28, -0.28])
    assert result.negative_pheromone_links == ((1, 2), (0, 1))
    assert not result.transition_compatible
    with pytest.raises(BANTFeedbackInputError, match="pheromone state values"):
        _feedback(pheromone_state=result.pheromone_state)


@pytest.mark.parametrize("invalid_tau", [-0.75, float("nan"), True])
def test_off_route_invalid_pheromone_rejects_the_whole_returned_snapshot(invalid_tau):
    state = _state()
    state[2][0] = invalid_tau
    with pytest.raises(BANTFeedbackInputError, match="pheromone state values"):
        _feedback(pheromone_state=state)


def test_terminal_sink_link_is_skipped_without_terminal_pheromone_state():
    result = _feedback()
    assert result.skipped_terminal_link == (2, -1)
    assert all(-1 not in outgoing for outgoing in result.pheromone_state.values())
    with pytest.raises(BANTFeedbackInputError, match="Sink"):
        _feedback(pheromone_state={0: {1: 0.5}, 1: {2: 0.25}, 2: {-1: 0.01}})


def test_direct_sink_route_validates_eq30_config_without_terminal_pheromone_update():
    preparation = prepare_bant(RouteQuality((0, -1), 1.0, 1.0, 1.0, 1.0))
    with pytest.raises(BANTFeedbackInputError, match="rho"):
        apply_bant_global_feedback(preparation, 1.0, {0: {}}, MRPConfig(rho=1.1), 1.0, 0.0)


def test_bant_does_not_mutate_input_pheromone_or_simulation_state():
    state = _state()
    before = deepcopy(state)
    _feedback(pheromone_state=state)
    assert state == before
    source = inspect.getsource(bant_module)
    assert all(name not in source for name in ("residual_e", "energy_consumption", "E_transmitting", "AC_ACO"))


def _discovered_route(aant_probability, draws):
    return construct_normal_sant_route(
        0, {0, 1}, [10.0, 8.0], [[0.0, 3.0], [3.0, 0.0]], [7.0, 3.0], {0: 2, 1: 1},
        5.0, {0: {1: 0.01}, 1: {0: 0.01}}, HeuristicBounds(0.1, 1000.0, 0.1, 1000.0),
        MRPConfig(aant_probability=aant_probability), 0.1, 2, FixedRNG(*draws),
    )


def test_normal_and_aant_completed_routes_use_the_same_bant_feedback_contract():
    for route in (_discovered_route(0.0, (0.5, 0.0)), _discovered_route(1.0, (0.0, 0.0))):
        quality = evaluate_route_quality(
            route.path, {0, 1}, [10.0, 8.0], [[0.0, 3.0], [3.0, 0.0]], [7.0, 3.0],
            MRPConfig(), 1.0, 2.0, 1.0, 1.0, 10.0, 1.0,
        )
        result = apply_bant_global_feedback(prepare_bant(quality), 1.0, {0: {1: 0.01}, 1: {0: 0.01}}, MRPConfig(), 1.0, 0.0)
        assert route.success and result.trace[0].forward_link == (0, 1)
