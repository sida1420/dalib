from copy import deepcopy
import inspect
from pathlib import Path
import random
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrp import HeuristicBounds, MRPConfig, construct_normal_sant_route
from mrp.phase2 import aant as aant_module
from mrp.phase2 import sant as sant_module


BOUNDS = HeuristicBounds(mu_min=0.0, mu_max=100.0, eta_min=0.1, eta_max=100.0)


class FixedRNG:
    def __init__(self, *draws):
        self._draws = iter(draws)

    def random(self):
        return next(self._draws)


def _inputs(base_dists, dist_matrix, live_nodes=None, ttl=2, probability=0.001):
    node_count = len(base_dists)
    return {
        "start_ch": 0,
        "live_nodes": set(range(node_count)) if live_nodes is None else live_nodes,
        "residual_e": [5.0] * node_count,
        "dist_matrix": dist_matrix,
        "base_dists": base_dists,
        "hop_counts": {node_id: node_count - node_id for node_id in range(node_count)},
        "communication_radius": 5.0,
        "pheromone_state": {},
        "bounds": BOUNDS,
        "config": MRPConfig(aant_probability=probability),
        "lambda_coefficient": 0.5,
        "ttl": ttl,
    }


def test_aant_trigger_boundary_uses_strict_paper_comparison():
    result = construct_normal_sant_route(
        **_inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]]), rng=FixedRNG(0.0009, 0.0)
    )
    assert result.success and result.path == (0, 1, -1)
    assert result.trace[0].mode == "AANT"


def test_non_trigger_at_exact_boundary_uses_normal_sant_branch():
    values = _inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]])
    values["pheromone_state"] = {0: {1: 0.01}}
    result = construct_normal_sant_route(**values, rng=FixedRNG(0.001, 0.0))
    assert result.success and result.trace[0].mode == "NORMAL_SANT"
    assert result.trace[0].local_tau_after is not None


@pytest.mark.parametrize("probability", (0.0, 1.0))
def test_aant_boundary_probabilities_still_consume_the_trigger_draw(probability):
    matrix = [[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]]
    values = _inputs([10.0, 4.0, 4.0], matrix, probability=probability)
    if probability == 0.0:
        values["pheromone_state"] = {0: {1: 1.0, 2: 1.0}}
    result = construct_normal_sant_route(**values, rng=FixedRNG(0.5, 0.1))
    assert result.trace[0].sampled_next_hop == 1
    assert result.trace[0].mode == ("NORMAL_SANT" if probability == 0.0 else "AANT")


def test_aant_selection_uses_no_eq26_to_31_calculation(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("normal SANT calculation called by AANT")

    for name in ("calculate_mu", "calculate_eta", "transition_probabilities", "update_local_pheromone"):
        monkeypatch.setattr(sant_module, name, fail_if_called)
    result = construct_normal_sant_route(
        **_inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]]), rng=FixedRNG(0.0, 0.0)
    )
    hop = result.trace[0]
    assert hop.mode == "AANT" and not hop.tau and not hop.mu and not hop.probabilities
    assert hop.local_tau_before is None and hop.local_tau_after is None


def test_aant_uniform_random_selection_uses_candidate_order_not_argmax():
    matrix = [[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]]
    result = construct_normal_sant_route(
        **_inputs([10.0, 4.0, 4.0], matrix), rng=FixedRNG(0.0, 0.75)
    )
    assert result.trace[0].valid_candidates == (1, 2)
    assert result.trace[0].sampled_next_hop == 2


def test_same_rng_seed_produces_same_aant_selection_and_trace():
    matrix = [[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]]
    values = _inputs([10.0, 4.0, 4.0], matrix, probability=1.0)
    first = construct_normal_sant_route(**values, rng=random.Random(9))
    second = construct_normal_sant_route(**values, rng=random.Random(9))
    assert first == second and first.trace[0].mode == "AANT"


def test_aant_uses_only_live_radio_neighbors_and_excludes_self_and_visited():
    matrix = [[0.0, 3.0, 7.0], [3.0, 0.0, 3.0], [7.0, 3.0, 0.0]]
    result = construct_normal_sant_route(
        **_inputs([20.0, 15.0, 4.0], matrix, ttl=3, probability=1.0),
        rng=FixedRNG(0.0, 0.0, 0.0, 0.0),
    )
    assert result.success and result.path == (0, 1, 2, -1)
    assert result.trace[0].valid_candidates == (1,)
    assert result.trace[1].valid_candidates == (2,)
    assert len(result.visited_nodes) == len(set(result.visited_nodes))


def test_dead_and_outside_radius_sensors_are_not_aant_candidates():
    matrix = [[0.0, 3.0, 7.0], [3.0, 0.0, 3.0], [7.0, 3.0, 0.0]]
    result = construct_normal_sant_route(
        **_inputs([10.0, 4.0, 4.0], matrix, live_nodes={0, 1}), rng=FixedRNG(0.0, 0.0)
    )
    assert result.trace[0].valid_candidates == (1,)


def test_aant_forward_consumes_one_ttl_and_no_candidate_fails_explicitly():
    forwarded = construct_normal_sant_route(
        **_inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]], ttl=1), rng=FixedRNG(0.0, 0.0)
    )
    assert not forwarded.success and forwarded.path == (0, 1)
    assert forwarded.remaining_ttl == 0 and forwarded.trace[0].ttl_before == 1
    failed = construct_normal_sant_route(
        **_inputs([10.0, 10.0], [[0.0, 7.0], [7.0, 0.0]]), rng=FixedRNG(0.0)
    )
    assert not failed.success and failed.failure_reason == "no_valid_aant_sensor_neighbor"


def test_terminal_sink_rule_precedes_aant_trigger(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("AANT trigger evaluated after direct Sink terminal condition")

    monkeypatch.setattr(sant_module, "should_create_aant", fail_if_called)
    result = construct_normal_sant_route(
        **_inputs([4.0], [[0.0]], ttl=1), rng=FixedRNG()
    )
    assert result.success and result.path == (0, -1)
    assert result.trace[0].mode == "TERMINAL_SINK"


def test_aant_keeps_residual_and_input_state_unchanged():
    values = _inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]])
    before = deepcopy(values)
    result = construct_normal_sant_route(**values, rng=FixedRNG(0.0, 0.0))
    assert values == before and result.pheromone_state == {}


def test_aant_module_has_no_common_energy_or_ac_aco_dependencies():
    source = inspect.getsource(aant_module) + inspect.getsource(sant_module)
    forbidden = (
        "energy_consumption", "E_transmitting", "E_data_receiving", "E_m(",
        "AC_ACO", "make_path", "network_config", "pheromone_update", "Node(",
    )
    assert all(reference not in source for reference in forbidden)
