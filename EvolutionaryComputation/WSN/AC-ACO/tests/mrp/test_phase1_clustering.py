from copy import deepcopy
from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrp.config import MRPConfig
from mrp.phase1.clustering import (
    ClusterFormationInputError,
    _count_event_neighbors,
    calculate_cluster_head_score,
    calculate_cluster_head_timer,
    select_cluster_head,
)
from point import Point


def _nodes(count: int = 4):
    return [Point(index, 0) for index in range(count)]


def _distances(values):
    return [[abs(left - right) for right in values] for left in values]


def _select(*, residual_e=(4.0, 9.0, 16.0), signals=None, distances=None, radius=10):
    nodes = _nodes(3)
    return select_cluster_head(
        nodes=nodes,
        live_nodes={0, 1, 2},
        residual_e=list(residual_e),
        event_nodes=[0, 1, 2],
        event_signal_strengths=signals or {0: 1.0, 1: 2.0, 2: 3.0},
        dist_matrix=distances or _distances([0, 4, 8]),
        communication_radius=radius,
        config=MRPConfig(),
    )


def test_q_i_implements_equation_24_exactly():
    config = MRPConfig(k1=0.5, k2=0.1, k3=0.4)
    score = calculate_cluster_head_score(4.0, 9, 16.0, config)
    assert score == pytest.approx(4.0**0.5 * 9**0.1 * 16.0**0.4)


def test_higher_residual_energy_increases_q_i():
    config = MRPConfig()
    assert calculate_cluster_head_score(9, 2, 3, config) > calculate_cluster_head_score(4, 2, 3, config)


def test_higher_event_neighbor_count_increases_q_i():
    config = MRPConfig()
    assert calculate_cluster_head_score(4, 3, 3, config) > calculate_cluster_head_score(4, 1, 3, config)


def test_higher_signal_strength_increases_q_i():
    config = MRPConfig()
    assert calculate_cluster_head_score(4, 2, 9, config) > calculate_cluster_head_score(4, 2, 3, config)


def test_timer_and_score_ranking_have_inverse_order_for_common_q():
    assert calculate_cluster_head_timer(9, 18) < calculate_cluster_head_timer(3, 18)
    result = _select(residual_e=(1.0, 4.0, 9.0), signals={0: 1.0, 1: 1.0, 2: 2.0})
    assert result.cluster_head == max(result.scores, key=result.scores.get)


def test_dead_event_node_cannot_be_cluster_head():
    result = select_cluster_head(
        _nodes(2), {0, 1}, [0.0, 4.0], [0, 1], {1: 2.0}, _distances([0, 1]), 10, MRPConfig()
    )
    assert result.cluster_head == 1
    assert result.members == ()


def test_node_outside_event_area_cannot_be_cluster_head():
    result = select_cluster_head(
        _nodes(3), {0, 1, 2}, [1.0, 4.0, 10_000.0], [0, 1], {0: 1.0, 1: 2.0}, _distances([0, 1, 2]), 10, MRPConfig()
    )
    assert result.cluster_head == 1
    assert 2 not in result.scores


def test_k_i_counts_only_other_event_neighbors_in_range():
    matrix = _distances([0, 5, 15])
    assert _count_event_neighbors(0, [0, 1, 2], matrix, 10) == 1
    assert _count_event_neighbors(1, [0, 1, 2], matrix, 10) == 2
    assert _count_event_neighbors(2, [0, 1, 2], matrix, 10) == 1


def test_k_i_excludes_nearby_node_outside_event_area():
    nodes = _nodes(3)
    result = select_cluster_head(
        nodes,
        {0, 1, 2},
        [1.0, 4.0, 100.0],
        [0, 1],
        {0: 1.0, 1: 2.0},
        _distances([0, 5, 1]),
        10,
        MRPConfig(),
    )
    assert dict(result.neighbor_counts) == {0: 1, 1: 1}
    assert 2 not in result.scores


def test_one_event_area_with_unique_earliest_timer_has_one_cluster_head():
    result = _select()
    assert isinstance(result.cluster_head, int)
    assert len(result.members) == 2
    assert {result.cluster_head, *result.members} == {0, 1, 2}


def test_non_clique_event_area_fails_instead_of_false_global_argmax():
    with pytest.raises(ClusterFormationInputError, match="single CH advertisement"):
        _select(
            distances=_distances([0, 5, 15]),
            radius=10,
        )


def test_non_clique_event_area_is_valid_when_winner_reaches_all_candidates():
    result = _select(
        residual_e=(1.0, 100.0, 1.0),
        signals={0: 1.0, 1: 1.0, 2: 1.0},
        distances=_distances([0, 5, 15]),
        radius=10,
    )
    assert result.cluster_head == 1


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -1.0, True, "unknown"])
def test_invalid_event_distance_fails_closed(distance):
    matrix = _distances([0, 5, 8])
    matrix[0][2] = matrix[2][0] = distance
    with pytest.raises(ClusterFormationInputError, match="event distance"):
        _select(distances=matrix)


def test_missing_signal_strength_fails_without_rssi_fallback():
    with pytest.raises(ClusterFormationInputError, match="missing event signal"):
        _select(signals={0: 1.0, 1: 2.0})


def test_phase_i_does_not_mutate_inputs():
    residual_e = [1.0, 4.0, 9.0]
    signals = {0: 1.0, 1: 1.0, 2: 2.0}
    distances = _distances([0, 4, 8])
    before = deepcopy((residual_e, signals, distances))
    select_cluster_head(
        _nodes(3), {0, 1, 2}, residual_e, [0, 1, 2], signals, distances, 10, MRPConfig()
    )
    assert (residual_e, signals, distances) == before


def test_timer_tie_is_rejected_without_invented_node_id_rule():
    with pytest.raises(ClusterFormationInputError, match="timer tie"):
        _select(residual_e=(4.0, 4.0, 4.0), signals={0: 1.0, 1: 1.0, 2: 1.0})


@pytest.mark.parametrize("signal", [0.0, -1.0, float("nan"), float("inf"), True, "strong"])
def test_invalid_event_signal_fails_without_fallback(signal):
    with pytest.raises(ClusterFormationInputError, match="event signal strength"):
        _select(signals={0: signal, 1: 1.0, 2: 2.0})


def test_all_zero_scores_fail_without_invented_cluster_head():
    with pytest.raises(ClusterFormationInputError, match="finite positive q_i"):
        select_cluster_head(
            _nodes(1), {0}, [1.0], [0], {0: 1.0}, [[0.0]], 10, MRPConfig()
        )
