"""Import-safe baseline-round adapter; AC_ACO.py is intentionally never imported."""

from collections.abc import Sequence
from dataclasses import dataclass

from ac_aco_mrp.phase1 import ACACOPhase1Result, ACACOPhase1State, select_ac_aco_cluster_heads
from ac_aco_mrp.types import ACACOParameters
from evaluate import energy_consumption, network_config
from mrp.runner_lifecycle import apply_baseline_lifecycle
from mrp.runner_types import PureMRPParameters, PureMRPState


@dataclass(frozen=True)
class BaselineState:
    physical_state: PureMRPState
    ac_aco_state: ACACOPhase1State


@dataclass(frozen=True)
class BaselineRoundResult:
    round_index: int
    phase1_result: ACACOPhase1Result | None
    final_topology: object | None
    e_m_list: tuple[float, ...] | None
    e_sum: float | None
    state_before: BaselineState
    state_after: BaselineState
    newly_dead: tuple[int, ...]
    success: bool
    failure_stage: str | None
    failure_reason: str | None


def run_baseline_round(
    nodes: Sequence[object], dist_matrix: Sequence[Sequence[float]], base_dists: Sequence[float],
    state: BaselineState, ac_aco: ACACOParameters, common: PureMRPParameters,
    ac_aco_rng: object, round_index: int,
) -> BaselineRoundResult:
    """Mirror AC_ACO.py final-Greedy behavior without its import-time side effect."""

    try:
        phase1 = select_ac_aco_cluster_heads(
            nodes, dist_matrix, base_dists, state.physical_state.residual_e,
            state.physical_state.live_nodes, state.ac_aco_state, ac_aco, common,
            ac_aco_rng, round_index,
        )
    except Exception as error:
        return _failure(round_index, state, "ac_aco_phase1", error)
    try:
        topology = network_config(
            nodes, list(phase1.selected_cluster_heads), common.communication_radius, common.d0,
            ac_aco.base_pos, phase1.state_after.hopping_factor, base_dists, dist_matrix,
            state.physical_state.residual_e,
        )
        if topology is None:
            raise ValueError("legacy final Greedy topology was unavailable")
        e_m_list, e_sum = energy_consumption(
            nodes, topology, common.d0, common.bit_count, common.ctrl_bit, base_dists,
            dist_matrix, common.e_elec, common.e_agg, common.free_space_coeff,
            common.multipath_coeff,
        )
        physical_after, newly_dead = apply_baseline_lifecycle(
            state.physical_state, e_m_list, e_sum, None,
        )
    except Exception as error:
        return _failure(round_index, state, "final_greedy", error, phase1)
    return BaselineRoundResult(
        round_index, phase1, topology, tuple(e_m_list), e_sum,
        state, BaselineState(physical_after, phase1.state_after), newly_dead,
        True, None, None,
    )


def _failure(round_index, state, stage, reason, phase1=None) -> BaselineRoundResult:
    return BaselineRoundResult(
        round_index, phase1, None, None, None, state, state, (), False, stage, str(reason),
    )
