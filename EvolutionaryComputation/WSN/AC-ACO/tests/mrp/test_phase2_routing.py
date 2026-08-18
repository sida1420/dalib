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
    HeuristicBounds,
    MRPConfig,
    PheromoneLifecycle,
    PureMRPParameters,
    PureMRPState,
    discover_mrp_phase2_routes,
    initialize_sensor_pheromone_state,
    route_cluster_heads_with_mrp,
)
from mrp.phase2 import routing as routing_module
from mrp.phase2.routing import MRPPhase2InputError


class FixedRNG:
    def __init__(self, *draws): self.draws = iter(draws)
    def random(self): return next(self.draws)


def _inputs(**overrides):
    values = {
        "start_cluster_head": 0, "live_nodes": {0, 1, 2}, "residual_e": [10.0, 8.0, 8.0],
        "dist_matrix": [[0.0, 3.0, 3.0], [3.0, 0.0, 6.0], [3.0, 6.0, 0.0]],
        "base_dists": [7.0, 3.0, 3.0], "hop_counts": {0: 2, 1: 1, 2: 1},
        "communication_radius": 5.0, "bounds": HeuristicBounds(0.1, 1000.0, 0.1, 1000.0),
        "config": MRPConfig(aant_probability=0.0), "lambda_coefficient": 0.1, "ttl": 2,
        "num_sants": 2, "rng": FixedRNG(0.5, 0.0, 0.5, 0.0), "c0": 1.0,
        "e_elec": 2.0, "free_space_coeff": 1.0, "multipath_coeff": 1.0,
        "d0": 10.0, "bit_count": 1.0, "c": 0.0, "c1": 0.01,
    }
    values.update(overrides)
    return values


def _run(**overrides):
    return discover_mrp_phase2_routes(**_inputs(**overrides))


def test_requested_ant_count_bootstraps_then_feedbacks_with_shared_pheromone():
    result = _run(num_sants=2)
    first, second = result.ant_results
    assert result.num_sants_executed == result.num_sants_requested == 2
    assert first.feedback_status == "first_route_best_initialized"
    assert first.bant_feedback is None and first.f_best_before is None
    assert second.feedback_status == "bant_feedback_applied"
    assert second.bant_feedback.previous_best_quality == first.f_best_after
    assert second.sant_result.trace[0].local_tau_before == first.sant_result.pheromone_state[0][1]
    assert result.f_best == max(first.route_quality.quality, second.route_quality.quality)


def test_third_ant_observes_the_second_ants_bant_updated_pheromone():
    result = _run(num_sants=3, rng=FixedRNG(0.5, 0.0, 0.5, 0.0, 0.5, 0.0))
    second, third = result.ant_results[1:]
    assert second.bant_feedback is not None and third.bant_feedback is not None
    assert third.sant_result.trace[0].local_tau_before == second.bant_feedback.pheromone_state[0][1]
    assert third.bant_feedback.previous_best_quality == second.f_best_after


def test_rng_seed_makes_the_complete_sequential_result_reproducible():
    one = _run(num_sants=3, rng=random.Random(7))
    two = _run(num_sants=3, rng=random.Random(7))
    assert one == two


def test_each_ant_owns_a_fresh_visited_route_and_ttl_budget():
    values = _inputs(
        live_nodes={0, 1}, residual_e=[10.0, 8.0],
        dist_matrix=[[0.0, 3.0], [3.0, 0.0]], base_dists=[7.0, 3.0],
        hop_counts={0: 2, 1: 1}, ttl=1, num_sants=2, rng=FixedRNG(0.5, 0.0, 0.5, 0.0),
    )
    result = discover_mrp_phase2_routes(**values)
    assert result.num_sants_executed == 2 and not result.successful_ant_results
    assert [ant.sant_result.path for ant in result.failed_ant_results] == [(0, 1), (0, 1)]
    assert [ant.sant_result.visited_nodes for ant in result.failed_ant_results] == [(0, 1), (0, 1)]
    assert [ant.sant_result.remaining_ttl for ant in result.failed_ant_results] == [0, 0]


def test_failed_sant_creates_no_route_quality_or_bant():
    result = _run(
        live_nodes={0, 1}, residual_e=[10.0, 8.0], dist_matrix=[[0.0, 9.0], [9.0, 0.0]],
        base_dists=[7.0, 3.0], hop_counts={0: 2, 1: 1}, num_sants=1, rng=FixedRNG(0.5),
    )
    ant = result.failed_ant_results[0]
    assert ant.feedback_status == "sant_failed" and ant.route_quality is ant.bant_preparation is ant.bant_feedback is None


def test_normal_and_aant_routes_are_collected_with_the_same_quality_contract():
    normal = _run(num_sants=1)
    aant = _run(num_sants=1, config=MRPConfig(aant_probability=1.0), rng=FixedRNG(0.0, 0.0))
    assert normal.successful_ant_results[0].sant_result.trace[0].mode == "NORMAL_SANT"
    assert aant.successful_ant_results[0].sant_result.trace[0].mode == "AANT"
    assert normal.successful_ant_results[0].route_quality.quality > 0
    assert aant.successful_ant_results[0].route_quality.quality > 0


def test_raw_duplicates_are_preserved_and_unique_paths_use_exact_id_equality():
    result = _run(num_sants=2)
    assert [ant.sant_result.path for ant in result.successful_ant_results] == [(0, 1, -1), (0, 1, -1)]
    assert result.unique_routes == ((0, 1, -1),)
    assert result.num_unique_routes == 1 and not result.multipath_ready


def test_two_distinct_paths_make_multipath_ready_without_fabricating_routes():
    result = _run(num_sants=2, rng=FixedRNG(0.5, 0.0, 0.5, 0.999))
    assert [ant.sant_result.path for ant in result.successful_ant_results] == [(0, 1, -1), (0, 2, -1)]
    assert result.unique_routes == ((0, 1, -1), (0, 2, -1))
    assert result.multipath_ready and result.num_unique_routes == 2


def test_direct_sink_route_skips_terminal_pheromone_for_bootstrap_and_feedback():
    result = _run(
        live_nodes={0}, residual_e=[10.0], dist_matrix=[[0.0]], base_dists=[3.0],
        hop_counts={0: 1}, num_sants=2, rng=FixedRNG(),
    )
    first, second = result.successful_ant_results
    assert first.sant_result.path == second.sant_result.path == (0, -1)
    assert first.bant_feedback is None
    assert second.bant_feedback.trace == ()
    assert second.bant_feedback.skipped_terminal_link == (0, -1)
    assert result.final_pheromone_state == {0: {}}


def test_final_pheromone_snapshot_is_detached_from_per_ant_diagnostics():
    result = _run(num_sants=2)
    before = result.ant_results[-1].bant_feedback.pheromone_state[0][1]
    result.final_pheromone_state[0][1] = 123.0
    assert result.ant_results[-1].bant_feedback.pheromone_state[0][1] == before


def test_negative_bant_pheromone_fails_fast_before_a_third_ant_can_transition():
    result = _run(num_sants=3, rng=FixedRNG(0.5, 0.0, 0.5, 0.0), c=0.0, c1=-1.0)
    assert result.status == "invalid_negative_pheromone_state"
    assert result.halted_ant_index == 1 and result.num_sants_executed == 2
    feedback = result.successful_ant_results[-1].bant_feedback
    assert feedback.global_delta_tau < 0 and feedback.negative_pheromone_links
    assert not feedback.transition_compatible


@pytest.mark.parametrize("num_sants", [0, -1, True])
def test_num_sants_is_required_positive_and_c_c1_must_be_finite(num_sants):
    with pytest.raises(MRPPhase2InputError, match="num_sants"):
        _run(num_sants=num_sants)
    with pytest.raises(MRPPhase2InputError, match="c"):
        _run(num_sants=1, c=float("nan"))
    with pytest.raises(MRPPhase2InputError, match="c1"):
        _run(num_sants=1, c1=True)


def test_orchestration_is_read_only_for_physical_state_and_uses_no_common_topology_energy_code():
    values = _inputs(num_sants=1)
    before = deepcopy((values["live_nodes"], values["residual_e"], values["dist_matrix"], values["base_dists"]))
    _run(**values)
    assert (values["live_nodes"], values["residual_e"], values["dist_matrix"], values["base_dists"]) == before
    source = inspect.getsource(routing_module)
    assert all(name not in source for name in ("energy_consumption", "materialize_topology", "AC_ACO", "network_config", "E_transmitting"))


def test_discovery_rejects_poisoned_off_route_snapshot_and_pheromone_once_at_boundary():
    distances = deepcopy(_inputs()["dist_matrix"])
    distances[1][2] = float("nan")
    with pytest.raises(MRPPhase2InputError, match="distance matrix value"):
        _run(dist_matrix=distances)

    state = initialize_sensor_pheromone_state(
        _inputs()["live_nodes"], _inputs()["dist_matrix"],
        _inputs()["communication_radius"], _inputs()["config"],
    )
    state[1][2] = float("nan")
    with pytest.raises(MRPPhase2InputError, match="pheromone state values"):
        _run(initial_pheromone_state=state)


def test_public_quality_and_feedback_apis_expose_no_validation_bypass_switch():
    from mrp.phase2.bant_feedback import apply_bant_global_feedback
    from mrp.phase2.route_quality import evaluate_route_quality

    assert "_snapshot_validated" not in inspect.signature(evaluate_route_quality).parameters
    assert "_state_validated" not in inspect.signature(apply_bant_global_feedback).parameters


def test_highest_valid_sensor_id_is_an_ordinary_phase2_source_in_a_100_sensor_network():
    node_count, radius = 100, 5.0
    distances = [
        [0.0 if source == candidate else 10.0 for candidate in range(node_count)]
        for source in range(node_count)
    ]
    distances[98][99] = distances[99][98] = 3.0
    base_dists = [10.0] * node_count
    base_dists[98] = 3.0
    residual = (10.0,) * node_count
    live = tuple(range(node_count))
    config = MRPConfig(aant_probability=0.0)
    pheromone = initialize_sensor_pheromone_state(live, distances, radius, config)

    assert 99 in pheromone and len(pheromone[99]) == 99
    assert pheromone[99][98] == config.initial_pheromone
    assert pheromone[99][0] == 0.0

    parameters = PureMRPParameters(
        radius, HeuristicBounds(0.1, 1000.0, 0.1, 1000.0), config,
        0.1, 2, 1, 1.0, 0.0, 0.01, 10.0, 1.0, 0.0,
        2.0, 0.0, 1.0, 1.0,
    )
    routing = route_cluster_heads_with_mrp(
        (99,), live, PureMRPState(residual, live, frozenset()),
        distances, base_dists, {sensor_id: 1 for sensor_id in live},
        parameters, FixedRNG(0.5, 0.0), PheromoneLifecycle.RESET_PER_DISCOVERY,
    )

    phase2 = routing.ch_routing_results[0].phase2_result
    assert phase2.start_cluster_head == 99
    assert phase2.num_sants_executed == 1
    assert phase2.unique_routes == ((99, 98, -1),)
    assert phase2.ant_results[0].sant_result.visited_nodes == (99, 98)
    assert routing.selected_routes[99] == (99, 98, -1)
