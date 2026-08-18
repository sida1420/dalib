import ast
import inspect
from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrp import HeuristicBounds, MRPConfig, RouteQuality, RouteQualityInputError, construct_normal_sant_route
from mrp.phase2 import route_quality as route_quality_module
from mrp.phase2.route_quality import evaluate_route_quality, prepare_bant


class FixedRNG:
    def __init__(self, *draws):
        self._draws = iter(draws)

    def random(self):
        return next(self._draws)


def _sant_route(aant_probability):
    return construct_normal_sant_route(
        start_ch=0,
        live_nodes={0, 1},
        residual_e=[10.0, 8.0],
        dist_matrix=[[0.0, 3.0], [3.0, 0.0]],
        base_dists=[7.0, 3.0],
        hop_counts={0: 2, 1: 1},
        communication_radius=5.0,
        pheromone_state={0: {1: 0.01}, 1: {0: 0.01}},
        bounds=HeuristicBounds(0.1, 1000.0, 0.1, 1000.0),
        config=MRPConfig(aant_probability=aant_probability),
        lambda_coefficient=0.1,
        ttl=2,
        rng=FixedRNG(0.5, 0.0) if aant_probability == 0.0 else FixedRNG(0.0, 0.0),
    )


def _quality(path):
    return evaluate_route_quality(
        path=path,
        live_nodes={0, 1},
        residual_e=[10.0, 8.0],
        dist_matrix=[[0.0, 3.0], [3.0, 0.0]],
        base_dists=[7.0, 3.0],
        config=MRPConfig(),
        c0=1.5,
        e_elec=2.0,
        free_space_coeff=1.0,
        multipath_coeff=1.0,
        d0=10.0,
        bit_count=1.0,
    )


def test_normal_and_aant_discovered_routes_share_the_same_quality_evaluation():
    normal = _sant_route(0.0)
    aant = _sant_route(1.0)
    assert normal.success and aant.success
    assert normal.path == aant.path == (0, 1, -1)
    assert normal.trace[0].mode == "NORMAL_SANT"
    assert aant.trace[0].mode == "AANT"
    assert _quality(normal.path) == _quality(aant.path)


def test_bant_preparation_reverses_completed_route_without_pheromone_feedback():
    normal = _sant_route(0.0)
    route_quality = _quality(normal.path)
    bant = prepare_bant(route_quality)
    assert bant.forward_path == (0, 1, -1)
    assert bant.reverse_path == (-1, 1, 0)
    assert bant.route_quality is route_quality
    calls = [
        node.func.id for node in ast.walk(ast.parse(inspect.getsource(prepare_bant)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "update_local_pheromone" not in calls
    assert "global_pheromone" not in inspect.getsource(route_quality_module)


@pytest.mark.parametrize(
    "route_quality",
    [
        RouteQuality((0, 1), 1.0, 1.0, 1.0, 1.0),
        RouteQuality((0, 0, -1), 1.0, 1.0, 1.0, 1.0),
        RouteQuality((0, -1), 0.0, 1.0, 1.0, 1.0),
    ],
)
def test_bant_preparation_rejects_forged_incomplete_or_invalid_quality(route_quality):
    with pytest.raises(RouteQualityInputError, match="BANT preparation|f1"):
        prepare_bant(route_quality)


def test_bant_preparation_rejects_non_route_quality_payload():
    with pytest.raises(RouteQualityInputError, match="RouteQuality"):
        prepare_bant(object())
