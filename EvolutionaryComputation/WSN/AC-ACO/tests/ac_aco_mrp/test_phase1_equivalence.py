import random

import numpy as np

from adaptive import adapt
from evaluate import E_m, energy_consumption, network_config
from path_constructing import make_path
from updater import pheromone_update

from ac_aco_mrp import select_ac_aco_cluster_heads
from conftest import make_mrp_parameters


def _baseline_equivalent(nodes, distances, base_dists, state, ac, common, rng):
    """Safe lower-level reproduction of AC_ACO.py lines 100--185 for comparison."""
    pheromone, chaos = np.array(state.pheromone_matrix, copy=True), list(state.chaos)
    p, beta, alpha, hopping = adapt(
        state.p, state.beta, state.alpha, 0, ac.t_max, ac.p_min, ac.p_max,
        ac.beta_min, ac.beta_max, ac.sigmoid_k, ac.alpha_min, ac.alpha_max,
        state.legacy_fitness_total, 0.0, ac.adaptive_energy_upper_bound,
        state.hopping_factor, ac.hopping_factor_min, ac.hopping_factor_max,
    )
    heuristic = [[0.0] * len(nodes) for _ in nodes]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                cost = E_m(common.e_elec, common.free_space_coeff, common.e_agg, common.multipath_coeff, common.bit_count, 0, 0, common.ctrl_bit, 0, distances[i][j], common.d0)
                heuristic[i][j] = (1 / cost) ** ac.gamma
    original = random.getstate()
    random.setstate(rng.getstate())
    candidates = []
    try:
        for ant_index, start in enumerate(random.sample([0, 1, 2, 3], ac.candidate_count)):
            timeout = 0
            while timeout < ac.path_making_timeout:
                path = make_path(ac.ch_proportion * len(nodes), start, [0, 1, 2, 3], pheromone, distances, heuristic, [100.0] * 4, ac.pheromone_weight, beta, alpha, chaos)
                if path is None:
                    timeout += 1
                    continue
                net = network_config(nodes, path, common.communication_radius, common.d0, ac.base_pos, hopping, base_dists, distances, [100.0] * 4)
                if net is None:
                    timeout += 1
                    continue
                _, e_sum = energy_consumption(nodes, net, common.d0, common.bit_count, common.ctrl_bit, base_dists, distances, common.e_elec, common.e_agg, common.free_space_coeff, common.multipath_coeff)
                pheromone_update(pheromone, ac.pheromone_intensity, path, chaos, alpha, distances)
                candidates.append((ant_index, start, tuple(path), e_sum))
                break
    finally:
        rng.setstate(random.getstate())
        random.setstate(original)
    winner = min(range(len(candidates)), key=lambda index: candidates[index][3])
    return candidates, winner


def test_hybrid_phase1_matches_safe_baseline_equivalent_candidate_loop(hybrid_case):
    nodes, distances = hybrid_case["nodes"], hybrid_case["distances"]
    common = make_mrp_parameters()
    reference_rng, hybrid_rng = random.Random(23), random.Random(23)
    expected, winner = _baseline_equivalent(nodes, distances, hybrid_case["base_dists"], hybrid_case["state"].ac_aco_state, hybrid_case["ac_aco"], common, reference_rng)
    actual = select_ac_aco_cluster_heads(
        nodes, distances, hybrid_case["base_dists"], (100.0,) * 4, (0, 1, 2, 3),
        hybrid_case["state"].ac_aco_state, hybrid_case["ac_aco"], common, hybrid_rng, 0,
    )
    assert [(item.ant_index, item.start_node, item.cluster_heads, item.legacy_energy_sum) for item in actual.candidates] == expected
    assert actual.selected_candidate_index == winner
    assert actual.selected_cluster_heads == expected[winner][2]


def test_phase1_preserves_first_minimum_tie_and_keeps_global_random_untouched(hybrid_case, monkeypatch):
    import ac_aco_mrp.phase1 as phase1
    calls = []
    global_before = random.getstate()
    original = phase1.energy_consumption
    monkeypatch.setattr(phase1, "energy_consumption", lambda *args: (calls.append(1) or ([0.0] * 4, 7.0)))
    result = select_ac_aco_cluster_heads(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"], (100.0,) * 4,
        (0, 1, 2, 3), hybrid_case["state"].ac_aco_state, hybrid_case["ac_aco"],
        hybrid_case["parameters"].mrp, random.Random(23), 0,
    )
    assert len(calls) == len(result.candidates) == 2 and result.selected_candidate_index == 0
    assert random.getstate() == global_before
    monkeypatch.setattr(phase1, "energy_consumption", original)


def test_phase1_uses_the_baseline_global_best_sentinel(hybrid_case, monkeypatch):
    import ac_aco_mrp.phase1 as phase1
    monkeypatch.setattr(phase1, "energy_consumption", lambda *args: ([0.0] * 4, 1e9))
    result = select_ac_aco_cluster_heads(
        hybrid_case["nodes"], hybrid_case["distances"], hybrid_case["base_dists"], (100.0,) * 4,
        (0, 1, 2, 3), hybrid_case["state"].ac_aco_state, hybrid_case["ac_aco"],
        hybrid_case["parameters"].mrp, random.Random(23), 0,
    )
    assert result.state_after.global_best_energy == 1e9
    assert result.state_after.global_best_path == ()
