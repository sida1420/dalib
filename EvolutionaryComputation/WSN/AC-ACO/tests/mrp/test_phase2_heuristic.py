from copy import deepcopy
from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrp.config import MRPConfig
from mrp.phase2.heuristic import (
    HeuristicBounds,
    HeuristicInputError,
    calculate_eta,
    calculate_mu,
    calculate_theta,
)
from mrp.phase2.transition import NoValidTransitionError, transition_probabilities


BOUNDS = HeuristicBounds(mu_min=1.0, mu_max=10.0, eta_min=0.5, eta_max=20.0)


def _heuristic_inputs():
    return {
        "residual_e": [4.0, 5.0, 2.0],
        "dist_matrix": [[0.0, 4.0, 8.0], [4.0, 0.0, 4.0], [8.0, 4.0, 0.0]],
        "base_dists": [10.0, 5.0, 2.0],
        "hop_counts": {0: 3, 1: 2, 2: 1},
        "config": MRPConfig(),
    }


def test_theta_is_active_when_candidate_hop_count_is_not_greater():
    assert calculate_theta(0, 1, **_heuristic_inputs()) == pytest.approx(12.5)


def test_theta_is_zero_when_candidate_hop_count_is_greater():
    values = _heuristic_inputs()
    values["hop_counts"] = {0: 1, 1: 2, 2: 3}
    assert calculate_theta(0, 1, **values) == 0.0


def test_mu_matches_equations_28_and_29_exactly():
    # base = 5^2 / 4 = 6.25; theta = 6.25 * (10 / 5) = 12.5.
    assert calculate_mu(0, 1, **_heuristic_inputs()) == pytest.approx(18.75)


def test_higher_candidate_energy_increases_mu():
    lower = _heuristic_inputs()
    higher = _heuristic_inputs()
    higher["residual_e"][1] = 10.0
    assert calculate_mu(0, 1, **higher) > calculate_mu(0, 1, **lower)


def test_shorter_inter_node_distance_increases_mu_base_component():
    short = _heuristic_inputs()
    long = _heuristic_inputs()
    short["hop_counts"] = long["hop_counts"] = {0: 1, 1: 2, 2: 3}
    long["dist_matrix"][0][1] = long["dist_matrix"][1][0] = 8.0
    assert calculate_mu(0, 1, **short) > calculate_mu(0, 1, **long)


@pytest.mark.parametrize(
    ("mu", "expected"),
    [(0.25, 0.5), (1.0, 1.0), (5.0, 5.0), (10.0, 10.0), (10.1, 20.0)],
)
def test_eta_matches_equation_27_including_bounds(mu, expected):
    assert calculate_eta(mu, BOUNDS) == expected


def test_transition_distribution_matches_controlled_two_candidate_equation_26():
    probabilities = transition_probabilities(
        0, [1, 2], set(), {1: 1.0, 2: 2.0}, {1: 1.0, 2: 1.0},
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], 2.0, MRPConfig()
    )
    assert probabilities == pytest.approx({1: 0.2, 2: 0.8})
    assert all(probability >= 0 for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_transition_excludes_visited_and_non_neighbors():
    probabilities = transition_probabilities(
        0, [1, 2, 3], {2}, {1: 1.0}, {1: 1.0},
        [[0.0, 1.0, 1.0, 5.0], [1.0, 0.0, 1.0, 5.0], [1.0, 1.0, 0.0, 5.0], [5.0, 5.0, 5.0, 0.0]],
        2.0, MRPConfig()
    )
    assert probabilities == {1: 1.0}
    assert 2 not in probabilities and 3 not in probabilities


def test_transition_excludes_the_source_from_its_own_neighbor_set():
    probabilities = transition_probabilities(
        0, [0, 1], set(), {1: 1.0}, {1: 1.0},
        [[0.0, 1.0], [1.0, 0.0]], 2.0, MRPConfig()
    )
    assert probabilities == {1: 1.0}


def test_transition_uses_generator_inputs_as_one_consistent_snapshot():
    probabilities = transition_probabilities(
        0, (node_id for node_id in [1, 2]), (node_id for node_id in [2]),
        {1: 1.0}, {1: 1.0},
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], 2.0, MRPConfig()
    )
    assert probabilities == {1: 1.0}


def test_higher_tau_increases_transition_probability():
    probabilities = transition_probabilities(
        0, [1, 2], set(), {1: 1.0, 2: 3.0}, {1: 1.0, 2: 1.0},
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], 2.0, MRPConfig()
    )
    assert probabilities[2] > probabilities[1]


def test_higher_eta_increases_transition_probability():
    probabilities = transition_probabilities(
        0, [1, 2], set(), {1: 1.0, 2: 1.0}, {1: 1.0, 2: 3.0},
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], 2.0, MRPConfig()
    )
    assert probabilities[2] > probabilities[1]


def test_zero_transition_denominator_fails_without_uniform_fallback():
    with pytest.raises(NoValidTransitionError, match="denominator"):
        transition_probabilities(
            0, [1], set(), {1: 0.0}, {1: 1.0}, [[0.0, 1.0], [1.0, 0.0]], 2.0, MRPConfig()
        )


def test_heuristic_rejects_sink_candidate_and_missing_hop_count():
    with pytest.raises(HeuristicInputError, match="Sink candidate"):
        calculate_mu(0, -1, **_heuristic_inputs())
    values = _heuristic_inputs()
    values["hop_counts"] = {0: 3}
    with pytest.raises(HeuristicInputError, match="missing hop count"):
        calculate_mu(0, 1, **values)


def test_heuristic_and_transition_reject_boolean_sensor_ids():
    with pytest.raises(HeuristicInputError, match="valid sensor IDs"):
        calculate_mu(True, 1, **_heuristic_inputs())
    with pytest.raises(HeuristicInputError, match="valid sensor ID"):
        transition_probabilities(
            True, [1], set(), {1: 1.0}, {1: 1.0}, [[0.0, 1.0], [1.0, 0.0]], 2.0, MRPConfig()
        )


def test_invalid_bounds_and_transition_parameters_fail_explicitly():
    with pytest.raises(HeuristicInputError, match="min bound"):
        calculate_eta(1.0, HeuristicBounds(2.0, 1.0, 0.5, 1.0))
    with pytest.raises(HeuristicInputError, match="alpha"):
        transition_probabilities(
            0, [1], set(), {1: 1.0}, {1: 1.0}, [[0.0, 1.0], [1.0, 0.0]], 2.0, MRPConfig(alpha=0.0)
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("residual_e", -1.0, "residual energy"),
        ("inter_node_distance", 0.0, "d_ij"),
        ("candidate_sink_distance", 0.0, "d_js"),
        ("source_sink_distance", float("nan"), "d_is"),
    ],
)
def test_heuristic_rejects_invalid_equation_28_and_29_values(field, value, match):
    values = _heuristic_inputs()
    if field == "residual_e":
        values["residual_e"][1] = value
    elif field == "inter_node_distance":
        values["dist_matrix"][0][1] = values["dist_matrix"][1][0] = value
    elif field == "candidate_sink_distance":
        values["base_dists"][1] = value
    else:
        values["base_dists"][0] = value
    with pytest.raises(HeuristicInputError, match=match):
        calculate_mu(0, 1, **values)


def test_heuristic_rejects_negative_residual_energy_anywhere_in_input_state():
    values = _heuristic_inputs()
    values["residual_e"][0] = -1.0
    with pytest.raises(HeuristicInputError, match="residual energy"):
        calculate_mu(0, 1, **values)


def test_heuristic_rejects_finite_input_that_overflows_theta_and_mu():
    values = _heuristic_inputs()
    values["base_dists"][1] = 5e-324
    with pytest.raises(HeuristicInputError, match=r"Eq. \(29\).*not finite"):
        calculate_theta(0, 1, **values)
    with pytest.raises(HeuristicInputError, match=r"Eq. \(29\).*not finite"):
        calculate_mu(0, 1, **values)


@pytest.mark.parametrize("bad_distance", [float("nan"), float("inf"), -1.0, True, "1.0"])
def test_transition_rejects_invalid_distinct_node_distance(bad_distance):
    with pytest.raises(HeuristicInputError, match="d_ij"):
        transition_probabilities(
            0, [1], set(), {1: 1.0}, {1: 1.0},
            [[0.0, bad_distance], [bad_distance, 0.0]], 2.0, MRPConfig()
        )


def test_transition_rejects_negative_pheromone_and_sink_candidate():
    with pytest.raises(HeuristicInputError, match="pheromone"):
        transition_probabilities(
            0, [1], set(), {1: -1.0}, {1: 1.0}, [[0.0, 1.0], [1.0, 0.0]], 2.0, MRPConfig()
        )
    with pytest.raises(HeuristicInputError, match="Sink transition"):
        transition_probabilities(
            0, [-1], set(), {}, {}, [[0.0]], 2.0, MRPConfig()
        )


def test_heuristic_inputs_are_not_mutated():
    values = _heuristic_inputs()
    before = deepcopy(values)
    calculate_mu(0, 1, **values)
    assert values == before


def test_transition_inputs_are_not_mutated():
    pheromones = {1: 1.0, 2: 2.0}
    heuristic_values = {1: 1.0, 2: 1.0}
    dist_matrix = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    before = deepcopy((pheromones, heuristic_values, dist_matrix))
    transition_probabilities(
        0, [1, 2], {0}, pheromones, heuristic_values, dist_matrix, 2.0, MRPConfig()
    )
    assert (pheromones, heuristic_values, dist_matrix) == before
