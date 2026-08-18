from copy import deepcopy
from dataclasses import fields
import inspect
from pathlib import Path
import random
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrp import (
    HeuristicBounds, MRPConfig, PheromoneLifecycle, PureMRPParameters,
    PureMRPRoundContext, PureMRPState, run_pure_mrp, run_pure_mrp_round,
)
from mrp import run_mrp as run_module
from point import Point


class FixedRNG:
    def __init__(self, *draws): self.draws = iter(draws)
    def random(self): return next(self.draws)


def _nodes(): return [Point(0, 0), Point(3, 0), Point(-3, 0), Point(20, 0)]
def _distances(nodes): return [[abs(left - right) for right in nodes] for left in nodes]


def _parameters(**overrides):
    values = dict(
        communication_radius=5.0, heuristic_bounds=HeuristicBounds(0.1, 1000.0, 0.1, 1000.0),
        config=MRPConfig(aant_probability=0.0), lambda_coefficient=0.1, ttl=2, num_sants=2,
        c0=1.0, c=0.0, c1=0.01, d0=10.0, bit_count=1.0, ctrl_bit=0.0,
        e_elec=2.0, e_agg=0.0, free_space_coeff=1.0, multipath_coeff=1.0,
    )
    values.update(overrides)
    return PureMRPParameters(**values)


def _state(residual=(100.0, 100.0, 100.0, 100.0), pheromone=None):
    return PureMRPState(tuple(residual), (0, 1, 2, 3), frozenset(), 0.0, pheromone)


def _context():
    return PureMRPRoundContext((0, 1), {0: 10.0, 1: 1.0}, {0: 2, 1: 1, 2: 1, 3: 2})


def _round(*, base_dists=(7.0, 3.0, 3.0, 10.0), state=None, rng=None, policy=PheromoneLifecycle.RESET_PER_DISCOVERY, **parameter_overrides):
    nodes = _nodes()
    return run_pure_mrp_round(
        nodes, _distances(nodes), base_dists, state or _state(), _context(), _parameters(**parameter_overrides),
        rng or FixedRNG(0.5, 0.0, 0.5, 0.999, 0.999), policy, 0,
    )


def _topology_ids(root):
    stack, result = [root], {}
    while stack:
        node = stack.pop()
        result[node.idx] = node
        stack.extend(node.branches)
    return result


def test_search_failure_commits_transmitted_ant_energy_when_enabled():
    before = _state()
    result = _round(
        state=before, ttl=1, num_sants=1,
        charge_ant_energy=True, ant_control_packet_bits=1,
        rng=FixedRNG(0.5, 0.0),
    )

    assert not result.success and result.failure_stage == "phase3"
    assert result.ant_control_energy.total_energy > 0
    assert result.state_after.cumulative_actual_energy == pytest.approx(
        result.ant_control_energy.total_energy
    )
    assert result.state_after.residual_e != before.residual_e


def test_search_failure_remains_atomic_when_ant_charging_is_disabled():
    before = _state()
    result = _round(
        state=before, ttl=1, num_sants=1,
        charge_ant_energy=False, ant_control_packet_bits=None,
        rng=FixedRNG(0.5, 0.0),
    )

    assert not result.success
    assert result.state_after is before
    assert result.ant_control_energy.is_zero


def test_end_to_end_direct_sink_uses_common_energy_and_baseline_residual_commit(monkeypatch):
    calls = []
    original = run_module.energy_consumption
    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(run_module, "energy_consumption", tracked)
    before = _state()
    result = _round(base_dists=(3.0, 3.0, 3.0, 10.0), state=before, rng=FixedRNG())
    assert result.success and result.selected_route == (0, -1) and len(calls) == 1
    assert result.e_sum == pytest.approx(sum(result.e_m_list))
    assert result.state_after.residual_e == tuple(max(0, energy - spent) for energy, spent in zip(before.residual_e, result.e_m_list))
    assert result.state_after.cumulative_actual_energy == result.e_sum
    assert result.state_after.live_nodes == before.live_nodes and not result.newly_dead


def test_multihop_member_relay_has_one_node_and_event_scope_excludes_unrelated_sensor():
    result = _round(rng=FixedRNG(0.5, 0.0), num_sants=1)
    nodes = _topology_ids(result.final_topology)
    assert result.success and result.selected_route == (0, 1, -1)
    assert set(nodes) == {-1, 0, 1} and nodes[1].isCH and nodes[1].p_idx == -1
    assert result.e_m_list[3] == 0  # live but neither event member nor selected relay
    assert result.e_m_list[1] > 0


def test_multiple_phase2_routes_phase3_selects_non_argmax_route_and_materializes_it():
    result = _round(
        state=_state((100.0, 100.0, 1.0, 100.0)), config=MRPConfig(aant_probability=1.0),
        rng=FixedRNG(0.0, 0.0, 0.0, 0.999, 0.999),
    )
    assert result.phase2_result.unique_routes == ((0, 1, -1), (0, 2, -1))
    assert result.phase3_result.candidates[0].phase3_fitness > result.phase3_result.candidates[1].phase3_fitness
    assert result.selected_route == (0, 2, -1)
    topology_nodes = _topology_ids(result.final_topology)
    assert set(topology_nodes) == {-1, 0, 1, 2}
    assert topology_nodes[2].relay_data == 2 * _parameters().bit_count
    assert result.e_m_list[2] > 0 and result.e_m_list[3] == 0


def test_pure_mrp_phase3_uses_all_discovered_routes_and_has_no_top_k_parameter():
    result = _round(
        state=_state((100.0, 100.0, 1.0, 100.0)),
        config=MRPConfig(aant_probability=1.0),
        rng=FixedRNG(0.0, 0.0, 0.0, 0.999, 0.999),
    )
    assert len(result.phase3_result.candidates) == result.phase2_result.num_unique_routes == 2
    assert "phase3_top_k" not in {field.name for field in fields(PureMRPParameters)}


def test_phase2_negative_pheromone_phase3_and_topology_failures_are_atomic(monkeypatch):
    before = _state()
    negative = _round(state=before, config=MRPConfig(aant_probability=1.0), rng=FixedRNG(0.0, 0.0, 0.0, 0.0), c1=-1.0)
    assert not negative.success and negative.failure_stage == "phase2" and negative.state_after == before
    monkeypatch.setattr(run_module, "select_phase3_route", lambda *args: (_ for _ in ()).throw(ValueError("phase3 fail")))
    phase3 = _round(state=before, rng=FixedRNG(0.5, 0.0), num_sants=1)
    assert not phase3.success and phase3.failure_stage == "phase3" and phase3.state_after == before
    monkeypatch.undo()
    monkeypatch.setattr(run_module, "materialize_topology", lambda *args: None)
    topology = _round(state=before, rng=FixedRNG(0.5, 0.0), num_sants=1)
    assert not topology.success and topology.failure_stage == "topology" and topology.state_after == before


def test_energy_failure_is_atomic(monkeypatch):
    before = _state()
    monkeypatch.setattr(run_module, "energy_consumption", lambda *args: (_ for _ in ()).throw(ValueError("energy fail")))
    result = _round(state=before, rng=FixedRNG(0.5, 0.0), num_sants=1)
    assert not result.success and result.failure_stage == "energy" and result.state_after == before


def test_phase1_failure_and_missing_lifecycle_policy_leave_state_unchanged():
    before = _state()
    missing_signal = PureMRPRoundContext((0, 1), {0: 10.0}, {0: 2, 1: 1, 2: 1, 3: 2})
    nodes = _nodes()
    result = run_pure_mrp_round(
        nodes, _distances(nodes), (7.0, 3.0, 3.0, 10.0), before, missing_signal,
        _parameters(), FixedRNG(), PheromoneLifecycle.RESET_PER_DISCOVERY, 0,
    )
    assert not result.success and result.failure_stage == "phase1" and result.state_after == before
    with pytest.raises(ValueError, match="policy"):
        run_pure_mrp_round(
            nodes, _distances(nodes), (7.0, 3.0, 3.0, 10.0), before, _context(),
            _parameters(), FixedRNG(), None, 0,
        )


def test_baseline_live_dead_swap_removal_is_applied_after_final_energy_only():
    result = _round(
        base_dists=(3.0, 3.0, 3.0, 10.0), state=_state((100.0, 0.1, 100.0, 100.0)),
        rng=FixedRNG(), policy=PheromoneLifecycle.PERSIST_ACROSS_ROUNDS,
    )
    assert result.success and result.newly_dead == (1,)
    assert result.state_after.live_nodes == (0, 3, 2)
    assert result.state_after.dead_nodes == frozenset({1})
    assert result.state_after.residual_e[1] == 0
    assert 1 not in result.state_after.pheromone_state
    assert all(1 not in outgoing for outgoing in result.state_after.pheromone_state.values())


def test_multi_round_preserves_physical_state_and_requires_explicit_pheromone_policy():
    nodes, matrix = _nodes(), _distances(_nodes())
    contexts = (_context(), _context())
    seed = FixedRNG(0.5, 0.0, 0.5, 0.0)
    result = run_pure_mrp(nodes, matrix, (7.0, 3.0, 3.0, 10.0), _state(), contexts, _parameters(num_sants=1), seed, PheromoneLifecycle.PERSIST_ACROSS_ROUNDS)
    first, second = result.rounds
    assert len(result.rounds) == 2 and second.state_before.residual_e == first.state_after.residual_e
    assert second.state_after.cumulative_actual_energy == pytest.approx(first.e_sum + second.e_sum)
    assert second.pheromone_before[0][1] == first.pheromone_after[0][1]


def test_reset_policy_restarts_at_tau_initial_while_persistence_keeps_link_state():
    nodes, matrix, contexts = _nodes(), _distances(_nodes()), (_context(), _context())
    reset = run_pure_mrp(nodes, matrix, (7.0, 3.0, 3.0, 10.0), _state(), contexts, _parameters(num_sants=1), FixedRNG(0.5, 0.0, 0.5, 0.0), PheromoneLifecycle.RESET_PER_DISCOVERY)
    assert reset.rounds[1].pheromone_before[0][1] == 0.01
    assert reset.rounds[1].state_after.pheromone_state is None


def test_round_is_seed_reproducible_and_does_not_mutate_caller_state():
    nodes, matrix, before = _nodes(), _distances(_nodes()), _state()
    one = run_pure_mrp_round(nodes, matrix, (7.0, 3.0, 3.0, 10.0), before, _context(), _parameters(num_sants=1), random.Random(5), PheromoneLifecycle.RESET_PER_DISCOVERY, 0)
    two = run_pure_mrp_round(nodes, matrix, (7.0, 3.0, 3.0, 10.0), before, _context(), _parameters(num_sants=1), random.Random(5), PheromoneLifecycle.RESET_PER_DISCOVERY, 0)
    assert one.selected_route == two.selected_route and one.state_after == two.state_after and before == _state()


def test_runner_reuses_only_allowed_boundaries_and_has_no_import_side_effects():
    source = inspect.getsource(run_module)
    assert all(name not in source for name in ("AC_ACO", "make_path", "network_config", "adapt(", "chaos", "E_transmitting"))
    assert "energy_consumption" in source and "materialize_topology" in source
