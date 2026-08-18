import ast
from copy import deepcopy
import inspect
from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import E_data_receiving, E_transmitting
from mrp.config import MRPConfig
from mrp.phase2 import route_quality as route_quality_module
from mrp.phase2.route_quality import RouteQualityInputError, evaluate_route_quality


def _inputs(**overrides):
    values = {
        "path": [0, 1, -1],
        "live_nodes": {0, 1},
        "residual_e": [10.0, 4.0],
        "dist_matrix": [[0.0, 3.0], [3.0, 0.0]],
        "base_dists": [20.0, 5.0],
        "config": MRPConfig(),
        "c0": 2.0,
        "e_elec": 2.0,
        "free_space_coeff": 1.0,
        "multipath_coeff": 1.0,
        "d0": 10.0,
        "bit_count": 1.0,
    }
    values.update(overrides)
    return values


def test_route_quality_matches_equations_33_to_36_exactly():
    result = evaluate_route_quality(**_inputs())
    expected_f2 = sum(
        E_transmitting(2.0, 1.0, 1.0, distance, 10.0, 1.0) + E_data_receiving(2.0, 1.0)
        for distance in (3.0, 5.0)
    )
    expected_quality = 2.0 * 4.0**0.4 / (expected_f2**0.2 * 8.0**0.4)
    assert result.f1 == 4.0
    assert result.f2 == pytest.approx(expected_f2)
    assert result.f3 == 8.0
    assert result.quality == pytest.approx(expected_quality)


def test_f2_reuses_existing_transmit_and_receive_primitives(monkeypatch):
    calls = []

    def transmit(*args):
        calls.append(("tx", args[3]))
        return 7.0

    def receive(*args):
        calls.append(("rx", args[1]))
        return 3.0

    monkeypatch.setattr(route_quality_module, "E_transmitting", transmit)
    monkeypatch.setattr(route_quality_module, "E_data_receiving", receive)
    assert evaluate_route_quality(**_inputs()).f2 == 20.0
    assert calls == [("tx", 3.0), ("rx", 1.0), ("tx", 5.0), ("rx", 1.0)]


def test_terminal_sink_link_uses_base_distance_without_sink_energy():
    values = _inputs(
        path=[0, -1], live_nodes={0}, residual_e=[7.0], dist_matrix=[[0.0]], base_dists=[5.0]
    )
    result = evaluate_route_quality(**values)
    expected = E_transmitting(2.0, 1.0, 1.0, 5.0, 10.0, 1.0) + E_data_receiving(2.0, 1.0)
    assert (result.f1, result.f2, result.f3) == pytest.approx((7.0, expected, 5.0))


def test_higher_minimum_energy_increases_route_quality():
    lower = evaluate_route_quality(**_inputs(residual_e=[10.0, 4.0]))
    higher = evaluate_route_quality(**_inputs(residual_e=[10.0, 5.0]))
    assert higher.quality > lower.quality


def test_higher_communication_energy_and_longer_path_reduce_quality():
    baseline = evaluate_route_quality(**_inputs())
    costly = evaluate_route_quality(**_inputs(free_space_coeff=2.0))
    longer = evaluate_route_quality(**_inputs(base_dists=[20.0, 8.0]))
    assert costly.f2 > baseline.f2 and costly.quality < baseline.quality
    assert longer.f3 > baseline.f3 and longer.quality < baseline.quality


def test_route_quality_is_read_only():
    values = _inputs()
    before = deepcopy(values)
    evaluate_route_quality(**values)
    assert values == before


@pytest.mark.parametrize(
    ("path", "match"),
    [([], "route"), ([-1], "route"), ([0, 1], "route"), ([0, 1, 0, -1], "duplicate")],
)
def test_invalid_route_shapes_fail_closed(path, match):
    with pytest.raises(RouteQualityInputError, match=match):
        evaluate_route_quality(**_inputs(path=path))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("residual_e", [10.0, float("nan")], "residual energy"),
        ("base_dists", [20.0, float("inf")], "base distance"),
        ("c0", 0.0, "c0"),
    ],
)
def test_invalid_route_metric_values_fail_closed(field, value, match):
    with pytest.raises(RouteQualityInputError, match=match):
        evaluate_route_quality(**_inputs(**{field: value}))


def test_dead_sensor_and_non_positive_metrics_fail_closed():
    with pytest.raises(RouteQualityInputError, match="dead sensor"):
        evaluate_route_quality(**_inputs(live_nodes={0}))
    with pytest.raises(RouteQualityInputError, match="f2"):
        evaluate_route_quality(**_inputs(e_elec=0.0, free_space_coeff=0.0, multipath_coeff=0.0))
    with pytest.raises(RouteQualityInputError, match="f3"):
        evaluate_route_quality(**_inputs(path=[0, -1], live_nodes={0}, residual_e=[1.0], dist_matrix=[[0.0]], base_dists=[0.0]))


def test_overflow_and_non_finite_quality_fail_closed():
    with pytest.raises(RouteQualityInputError, match="overflowed"):
        evaluate_route_quality(
            **_inputs(path=[0, -1], live_nodes={0}, residual_e=[1e308], dist_matrix=[[0.0]],
                      base_dists=[1.0], config=MRPConfig(k7=2.0))
        )
    with pytest.raises(RouteQualityInputError, match="link communication energy overflowed"):
        evaluate_route_quality(**_inputs(path=[0, -1], live_nodes={0}, residual_e=[1.0],
                                          dist_matrix=[[0.0]], base_dists=[1e308], d0=1.0))
    with pytest.raises(RouteQualityInputError, match="denominator"):
        evaluate_route_quality(**_inputs(path=[0, -1], live_nodes={0}, residual_e=[1.0],
                                          dist_matrix=[[0.0]], base_dists=[1e-300], config=MRPConfig(k9=1000.0)))


def test_malformed_distance_matrix_row_fails_with_route_quality_error():
    with pytest.raises(RouteQualityInputError, match="distance inputs"):
        evaluate_route_quality(**_inputs(dist_matrix=[None, None]))


def test_route_quality_has_no_private_radio_equations_or_physical_state_mutation():
    source = inspect.getsource(route_quality_module)
    assert all(name not in source for name in ("def mrp_tx_energy", "def mrp_rx_energy", "energy_consumption", "AC_ACO"))
    imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    assert any(node.module == "evaluate" and {name.name for name in node.names} == {"E_data_receiving", "E_transmitting"} for node in imports)
    mutations = [
        node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        and any(isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "residual_e" for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))
    ]
    assert mutations == []
