from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrp import (
    HeuristicBounds, MRPConfig, PheromoneLifecycle, PureMRPIntegrationError,
    PureMRPParameters, PureMRPRoundContext, PureMRPState, run_pure_mrp_round,
)
from point import Point


def _parameters():
    return PureMRPParameters(
        5.0, HeuristicBounds(0.1, 1000.0, 0.1, 1000.0), MRPConfig(),
        0.1, 2, 1, 1.0, 0.0, 0.01, 10.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0,
    )


class FixedRNG:
    def random(self):
        return 0.0


def _run(state, policy=PheromoneLifecycle.RESET_PER_DISCOVERY):
    nodes = [Point(0, 0), Point(3, 0)]
    return run_pure_mrp_round(
        nodes, [[0.0, 3.0], [3.0, 0.0]], [3.0, 3.0], state,
        PureMRPRoundContext((0, 1), {0: 2.0, 1: 1.0}, {0: 1, 1: 1}),
        _parameters(), FixedRNG(), policy, 0,
    )


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (PureMRPState((1.0, 1.0), (0, 0), frozenset({1})), "unique"),
        (PureMRPState((1.0, 1.0), (0,), frozenset()), "partition"),
        (PureMRPState((1.0, float("nan")), (0, 1), frozenset()), "residual"),
        (PureMRPState((1.0, 1.0), (0,), frozenset({1})), "dead nodes"),
    ],
)
def test_runner_rejects_inconsistent_public_state_snapshots(state, message):
    with pytest.raises(PureMRPIntegrationError, match=message):
        _run(state)


@pytest.mark.parametrize("pheromone", [
    {0: {1: -0.1}, 1: {0: 0.01}}, {0: {}, 1: {0: 0.01}},
    {False: {True: 0.01}, True: {False: 0.01}},
])
def test_runner_rejects_malformed_persisted_pheromone_before_direct_sink_routing(pheromone):
    state = PureMRPState((1.0, 1.0), (0, 1), frozenset(), pheromone_state=pheromone)
    with pytest.raises(PureMRPIntegrationError, match="pheromone"):
        _run(state)


def test_pheromone_snapshots_are_detached_and_deeply_immutable():
    pheromone = {0: {1: 0.01}, 1: {0: 0.01}}
    state = PureMRPState((1.0, 1.0), (0, 1), frozenset(), pheromone_state=pheromone)
    pheromone[0][1] = 9.0
    assert state.pheromone_state[0][1] == 0.01
    with pytest.raises(TypeError):
        state.pheromone_state[0][1] = 8.0
    result = _run(PureMRPState((100.0, 100.0), (0, 1), frozenset()), PheromoneLifecycle.PERSIST_ACROSS_ROUNDS)
    with pytest.raises(TypeError):
        result.state_after.pheromone_state[0][1] = 8.0
