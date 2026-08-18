from pathlib import Path
import random
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ac_aco_mrp import ACACOParameters, HybridParameters, HybridRoundContext, HybridState, initialize_ac_aco_phase1_state
from mrp import HeuristicBounds, MRPConfig, PureMRPParameters, PureMRPState
from point import Point


def make_nodes():
    return [Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)]


def make_distances(nodes):
    return [[abs(left - right) for right in nodes] for left in nodes]


def make_mrp_parameters(**overrides):
    values = dict(
        communication_radius=50.0, heuristic_bounds=HeuristicBounds(0.1, 1000.0, 0.1, 1000.0),
        config=MRPConfig(aant_probability=0.0), lambda_coefficient=0.1, ttl=2, num_sants=1,
        c0=1.0, c=0.0, c1=0.01, d0=10.0, bit_count=1.0, ctrl_bit=0.0,
        e_elec=2.0, e_agg=0.0, free_space_coeff=1.0, multipath_coeff=1.0,
    )
    values.update(overrides)
    return PureMRPParameters(**values)


def make_ac_aco_parameters(**overrides):
    values = dict(
        base_pos=Point(0, 3), adaptive_energy_upper_bound=400.0,
        candidate_count=2, ch_proportion=0.5,
    )
    values.update(overrides)
    return ACACOParameters(**values)


@pytest.fixture
def hybrid_case():
    nodes = make_nodes()
    ac_aco = make_ac_aco_parameters()
    physical = PureMRPState((100.0,) * 4, (0, 1, 2, 3), frozenset())
    return dict(
        nodes=nodes,
        distances=make_distances(nodes),
        base_dists=(3.0, 3.0, 3.0, 3.0),
        ac_aco=ac_aco,
        parameters=HybridParameters(ac_aco, make_mrp_parameters()),
        state=HybridState(physical, initialize_ac_aco_phase1_state(4, ac_aco, random.Random(11))),
        context=HybridRoundContext({0: 1, 1: 1, 2: 1, 3: 1}),
    )
