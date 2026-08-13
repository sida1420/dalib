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
from mrp.phase2 import sant as sant_module
from mrp.phase2.sant_validation import SANTInputError


BOUNDS = HeuristicBounds(mu_min=0.0, mu_max=100.0, eta_min=0.1, eta_max=100.0)


class FixedRNG:
    def __init__(self, *draws):
        self._draws = iter(draws)

    def random(self):
        return next(self._draws)


def _route_inputs(base_dists, dist_matrix, pheromone_state, ttl=3):
    node_count = len(base_dists)
    return {
        "start_ch": 0,
        "live_nodes": set(range(node_count)),
        "residual_e": [5.0] * node_count,
        "dist_matrix": dist_matrix,
        "base_dists": base_dists,
        "hop_counts": {node_id: node_count - node_id for node_id in range(node_count)},
        "communication_radius": 5.0,
        "pheromone_state": pheromone_state,
        "bounds": BOUNDS,
        "config": MRPConfig(rho=0.2, aant_probability=0.0),
        "lambda_coefficient": 0.5,
        "ttl": ttl,
    }


def test_direct_sink_termination_skips_equations_and_local_pheromone(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("sensor-only calculation called for terminal Sink hop")

    for name in ("calculate_mu", "transition_probabilities", "update_local_pheromone"):
        monkeypatch.setattr(sant_module, name, fail_if_called)
    result = construct_normal_sant_route(
        **_route_inputs([3.0], [[0.0]], {}, ttl=1), rng=FixedRNG()
    )
    assert result.success and result.path == (0, -1)
    assert result.remaining_ttl == 0 and result.trace[0].local_tau_after is None


def test_one_relay_path_updates_only_sensor_link_pheromone():
    result = construct_normal_sant_route(
        **_route_inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]], {0: {1: 0.01}}, ttl=2),
        rng=FixedRNG(0.0, 0.0),
    )
    assert result.success and result.path == (0, 1, -1)
    assert result.trace[0].local_tau_before == 0.01
    assert result.trace[0].local_tau_after == pytest.approx(0.11911111111111111)
    assert result.trace[1].sampled_next_hop == -1
    assert result.trace[1].local_tau_before is None


def test_multihop_path_excludes_visited_sensor_and_never_loops():
    matrix = [[0.0, 3.0, 9.0], [3.0, 0.0, 3.0], [9.0, 3.0, 0.0]]
    result = construct_normal_sant_route(
        **_route_inputs([15.0, 10.0, 4.0], matrix, {0: {1: 0.01}, 1: {2: 0.01}}, ttl=3),
        rng=FixedRNG(0.0, 0.0, 0.0, 0.0),
    )
    assert result.success and result.path == (0, 1, 2, -1)
    assert result.trace[1].valid_candidates == (2,)
    assert len(result.visited_nodes) == len(set(result.visited_nodes))


def test_dead_sensor_is_excluded_from_normal_candidate_set():
    matrix = [[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]]
    values = _route_inputs([10.0, 4.0, 4.0], matrix, {0: {1: 0.01}}, ttl=2)
    values["live_nodes"] = {0, 1}
    result = construct_normal_sant_route(**values, rng=FixedRNG(0.0, 0.0))
    assert result.success and result.path == (0, 1, -1)
    assert result.trace[0].valid_candidates == (1,)


def test_eq26_trace_and_sampling_are_probabilistic_not_argmax():
    matrix = [[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 2.0, 0.0]]
    result = construct_normal_sant_route(
        **_route_inputs([10.0, 4.0, 4.0], matrix, {0: {1: 1.0, 2: 2.0}}, ttl=2),
        rng=FixedRNG(0.0, 0.1),
    )
    hop = result.trace[0]
    assert hop.probabilities == pytest.approx({1: 0.2, 2: 0.8})
    assert hop.sampled_next_hop == 1
    assert set(hop.tau) == set(hop.mu) == set(hop.eta) == set(hop.probabilities) == {1, 2}


def test_same_rng_seed_produces_same_result_and_trace():
    matrix = [[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 2.0, 0.0]]
    values = _route_inputs([10.0, 4.0, 4.0], matrix, {0: {1: 1.0, 2: 2.0}}, ttl=2)
    first = construct_normal_sant_route(**values, rng=random.Random(7))
    second = construct_normal_sant_route(**values, rng=random.Random(7))
    assert first == second


def test_no_neighbor_fails_without_invented_backtracking():
    result = construct_normal_sant_route(
        **_route_inputs([10.0, 10.0], [[0.0, 7.0], [7.0, 0.0]], {}, ttl=3), rng=FixedRNG(0.0)
    )
    assert not result.success and result.path == (0,)
    assert result.failure_reason == "no_valid_sensor_neighbor"
    assert not result.trace


def test_zero_sensor_transition_weight_is_explicit_failure():
    result = construct_normal_sant_route(
        **_route_inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]], {0: {1: 0.0}}, ttl=2),
        rng=FixedRNG(0.0),
    )
    assert not result.success and result.failure_reason == "no_positive_transition_weight"


def test_missing_current_sensor_link_pheromone_fails_explicitly():
    with pytest.raises(SANTInputError, match="missing pheromone value"):
        construct_normal_sant_route(
            **_route_inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]], {0: {}}, ttl=2),
            rng=FixedRNG(0.0),
        )


@pytest.mark.parametrize("state", [{0: {1: 0.01, -1: 0.01}}, {0: {1: 0.01}, -1: {0: 0.01}}])
def test_sant_rejects_sink_pheromone_edges_in_caller_state(state):
    with pytest.raises(SANTInputError, match="Sink, dead, or invalid"):
        construct_normal_sant_route(
            **_route_inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]], state, ttl=2),
            rng=FixedRNG(),
        )


def test_ttl_counts_sensor_forwards_and_expires_before_later_sink_hop():
    result = construct_normal_sant_route(
        **_route_inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]], {0: {1: 0.01}}, ttl=1),
        rng=FixedRNG(0.0, 0.0),
    )
    assert not result.success and result.path == (0, 1)
    assert result.remaining_ttl == 0 and result.failure_reason == "ttl_exhausted"


def test_sant_reads_but_does_not_mutate_simulation_or_input_pheromone_state():
    values = _route_inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]], {0: {1: 0.01}}, ttl=2)
    before = deepcopy(values)
    construct_normal_sant_route(**values, rng=FixedRNG(0.0, 0.0))
    assert values == before


def test_sensor_hop_reuses_checkpoint_4b_local_update(monkeypatch):
    calls = []
    original = sant_module.update_local_pheromone

    def tracked_update(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(sant_module, "update_local_pheromone", tracked_update)
    construct_normal_sant_route(
        **_route_inputs([10.0, 4.0], [[0.0, 3.0], [3.0, 0.0]], {0: {1: 0.01}}, ttl=2),
        rng=FixedRNG(0.0, 0.0),
    )
    assert len(calls) == 1 and calls[0][0][:3] == (0, 1, 0.01)


def test_normal_sant_imports_no_common_energy_or_ac_aco_code():
    source = inspect.getsource(sant_module)
    forbidden = (
        "energy_consumption", "E_transmitting", "E_data_receiving", "E_m(",
        "AC_ACO", "make_path", "network_config", "pheromone_update", "Node(",
    )
    assert all(reference not in source for reference in forbidden)
