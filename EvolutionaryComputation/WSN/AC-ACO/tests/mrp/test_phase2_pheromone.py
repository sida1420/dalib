from copy import deepcopy
import inspect
from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrp import MRPConfig
from mrp.phase2 import pheromone as pheromone_module
from mrp.phase2.pheromone import (
    PheromoneInputError,
    calculate_local_pheromone_deposit,
    initialize_sensor_pheromone_state,
    update_local_pheromone,
)


def _link_inputs():
    return {
        "source_id": 0,
        "candidate_id": 1,
        "residual_e": [4.0, 5.0, 2.0],
        "dist_matrix": [[0.0, 3.0, 7.0], [3.0, 0.0, 2.0], [7.0, 2.0, 0.0]],
        "lambda_coefficient": 0.5,
    }


def test_local_deposit_matches_equation_31_exactly():
    assert calculate_local_pheromone_deposit(**_link_inputs()) == pytest.approx(0.5)


def test_local_update_matches_equation_30_exactly():
    assert update_local_pheromone(
        old_pheromone=0.2, config=MRPConfig(rho=0.2), **_link_inputs()
    ) == pytest.approx(0.26)


def test_higher_source_or_candidate_energy_increases_local_deposit():
    lower = _link_inputs()
    higher_source = _link_inputs()
    higher_candidate = _link_inputs()
    higher_source["residual_e"][0] = 8.0
    higher_candidate["residual_e"][1] = 10.0
    baseline = calculate_local_pheromone_deposit(**lower)
    assert calculate_local_pheromone_deposit(**higher_source) > baseline
    assert calculate_local_pheromone_deposit(**higher_candidate) > baseline


def test_larger_distance_reduces_local_deposit():
    near = _link_inputs()
    far = _link_inputs()
    far["dist_matrix"][0][1] = far["dist_matrix"][1][0] = 6.0
    assert calculate_local_pheromone_deposit(**far) < calculate_local_pheromone_deposit(**near)


def test_rho_zero_preserves_old_pheromone():
    assert update_local_pheromone(
        old_pheromone=0.7, config=MRPConfig(rho=0.0), **_link_inputs()
    ) == 0.7


def test_controlled_rho_mixes_old_value_and_deposit_exactly():
    assert update_local_pheromone(
        old_pheromone=0.2, config=MRPConfig(rho=0.5), **_link_inputs()
    ) == pytest.approx(0.35)


def test_initial_pheromone_remains_config_owned_not_local_update_state():
    config = MRPConfig(initial_pheromone=99.0, rho=0.2)
    assert update_local_pheromone(old_pheromone=0.2, config=config, **_link_inputs()) == pytest.approx(0.26)


def test_table2_initializes_neighbor_and_non_neighbor_sensor_links():
    state = initialize_sensor_pheromone_state(
        {0, 1, 2}, [[0.0, 3.0, 7.0], [3.0, 0.0, 3.0], [7.0, 3.0, 0.0]], 5.0, MRPConfig()
    )
    assert state[0] == {1: 0.01, 2: 0.0}
    assert state[1] == {0: 0.01, 2: 0.01}


def test_initializer_rejects_mixed_invalid_ids_before_sorting():
    with pytest.raises(PheromoneInputError, match="live_nodes"):
        initialize_sensor_pheromone_state({0, "bad"}, [[0.0]], 5.0, MRPConfig())


@pytest.mark.parametrize(
    ("source_id", "candidate_id", "match"),
    [(0, -1, "temporary source-boundary"), (-1, 1, "temporary source-boundary"), (0, 0, "distinct")],
)
def test_sink_and_self_links_are_rejected(source_id, candidate_id, match):
    values = _link_inputs()
    values["source_id"] = source_id
    values["candidate_id"] = candidate_id
    with pytest.raises(PheromoneInputError, match=match):
        calculate_local_pheromone_deposit(**values)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_energy", 0.0, "source residual energy"),
        ("source_energy", float("inf"), "source residual energy"),
        ("candidate_energy", -1.0, "candidate residual energy"),
        ("candidate_energy", float("nan"), "candidate residual energy"),
        ("distance", 0.0, "d_ij"),
        ("distance", float("nan"), "d_ij"),
        ("lambda", -1.0, "lambda coefficient"),
        ("lambda", float("nan"), "lambda coefficient"),
        ("lambda", True, "lambda coefficient"),
        ("lambda", float("inf"), "lambda coefficient"),
    ],
)
def test_invalid_equation_31_inputs_fail_explicitly(field, value, match):
    values = _link_inputs()
    if field == "source_energy":
        values["residual_e"][0] = value
    elif field == "candidate_energy":
        values["residual_e"][1] = value
    elif field == "distance":
        values["dist_matrix"][0][1] = values["dist_matrix"][1][0] = value
    else:
        values["lambda_coefficient"] = value
    with pytest.raises(PheromoneInputError, match=match):
        calculate_local_pheromone_deposit(**values)


def test_finite_distance_that_overflows_dij_squared_fails_explicitly():
    values = _link_inputs()
    values["dist_matrix"][0][1] = values["dist_matrix"][1][0] = 1e308
    with pytest.raises(PheromoneInputError, match=r"Eq. \(31\) intermediate"):
        calculate_local_pheromone_deposit(**values)


@pytest.mark.parametrize("rho", [-0.1, 1.1, float("nan"), True])
def test_invalid_rho_fails_explicitly(rho):
    with pytest.raises(PheromoneInputError, match="rho"):
        update_local_pheromone(old_pheromone=0.2, config=MRPConfig(rho=rho), **_link_inputs())


def test_missing_mrp_config_fails_with_local_input_error():
    with pytest.raises(PheromoneInputError, match="MRPConfig"):
        update_local_pheromone(old_pheromone=0.2, config=None, **_link_inputs())


@pytest.mark.parametrize("old_pheromone", [-1.0, float("nan"), float("inf"), True])
def test_invalid_old_pheromone_fails_explicitly(old_pheromone):
    with pytest.raises(PheromoneInputError, match="old pheromone"):
        update_local_pheromone(old_pheromone=old_pheromone, config=MRPConfig(), **_link_inputs())


def test_invalid_sensor_id_fails_explicitly():
    values = _link_inputs()
    values["candidate_id"] = True
    with pytest.raises(PheromoneInputError, match="valid sensor IDs"):
        calculate_local_pheromone_deposit(**values)


def test_local_pheromone_inputs_are_not_mutated():
    values = _link_inputs()
    before = deepcopy(values)
    update_local_pheromone(old_pheromone=0.2, config=MRPConfig(), **values)
    assert values == before


def test_local_primitive_does_not_import_common_energy_or_ac_aco_pheromone_code():
    source = inspect.getsource(pheromone_module)
    forbidden_references = (
        "E_data_receiving",
        "E_data_aggregating",
        "E_transmitting",
        "energy_consumption",
        "src.updater",
        "pheromone_update",
    )
    assert all(reference not in source for reference in forbidden_references)
