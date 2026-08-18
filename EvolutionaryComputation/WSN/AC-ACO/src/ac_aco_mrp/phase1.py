"""Safe reproduction of AC_ACO.py's Phase-I candidate-selection loop."""

from collections.abc import Collection, Sequence
from contextlib import contextmanager
import math
import random
import threading

import numpy as np

from adaptive import adapt
from evaluate import E_m, energy_consumption, network_config
from path_constructing import make_path
from updater import path_length, pheromone_update

from .types import (
    ACACOCandidateFitness,
    ACACOParameters,
    ACACOPhase1Result,
    ACACOPhase1State,
)
from mrp.runner_types import PureMRPParameters


class ACACOPhase1Error(ValueError):
    """Raised when the baseline-equivalent CH-selection loop cannot run."""


_BASELINE_RANDOM_LOCK = threading.RLock()


def initialize_ac_aco_phase1_state(
    node_count: int, parameters: ACACOParameters, ac_aco_rng: object,
) -> ACACOPhase1State:
    """Create the exact baseline initial tau/chaos state without importing AC_ACO."""

    _validate_parameters(node_count, parameters)
    _require_stateful_rng(ac_aco_rng)
    chaos = [ac_aco_rng.random()]
    for _ in range(1, node_count):
        chaos.append(chaos[-1] * (1 - chaos[-1]) * parameters.chaos_r)
    return ACACOPhase1State(
        np.full((node_count, node_count), parameters.pheromone_initial, dtype=np.float64),
        tuple(chaos), parameters.initial_p, parameters.initial_beta,
        parameters.initial_alpha, parameters.initial_hopping_factor,
        parameters.initial_l_best,
    )


def select_ac_aco_cluster_heads(
    nodes: Sequence[object], dist_matrix: Sequence[Sequence[float]], base_dists: Sequence[float],
    residual_e: Sequence[float], live_nodes: Collection[int], state: ACACOPhase1State,
    parameters: ACACOParameters, common: PureMRPParameters, ac_aco_rng: object,
    round_index: int,
) -> ACACOPhase1Result:
    """Use legacy Greedy only as the read-only AC-ACO candidate-fitness oracle."""

    node_count = len(nodes)
    _validate_parameters(node_count, parameters)
    _validate_inputs(node_count, dist_matrix, base_dists, residual_e, live_nodes, state, common, round_index)
    _require_stateful_rng(ac_aco_rng)
    pheromone = np.array(state.pheromone_matrix, dtype=np.float64, copy=True)
    chaos = list(state.chaos)
    p, beta, alpha, hopping = adapt(
        state.p, state.beta, state.alpha, round_index, parameters.t_max,
        parameters.p_min, parameters.p_max, parameters.beta_min, parameters.beta_max,
        parameters.sigmoid_k, parameters.alpha_min, parameters.alpha_max,
        state.legacy_fitness_total, 0.0, parameters.adaptive_energy_upper_bound,
        state.hopping_factor, parameters.hopping_factor_min, parameters.hopping_factor_max,
    )
    heuristic = _heuristic_matrix(node_count, dist_matrix, parameters, common)
    candidates: list[ACACOCandidateFitness] = []
    with _baseline_random_stream(ac_aco_rng):
        starts = random.sample(list(live_nodes), min(parameters.candidate_count, len(live_nodes)))
        for ant_index, start in enumerate(starts):
            candidate = _evaluate_candidate(
                ant_index, start, nodes, dist_matrix, base_dists, residual_e, live_nodes,
                pheromone, chaos, beta, alpha, hopping, heuristic, parameters, common,
            )
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        raise ACACOPhase1Error("legacy AC-ACO candidate loop found no valid cluster-head set")
    best_index = 0
    for index, candidate in enumerate(candidates):
        if candidates[best_index].legacy_energy_sum > candidate.legacy_energy_sum:
            best_index = index
    winner = candidates[best_index]
    global_energy, global_path = state.global_best_energy, state.global_best_path
    if winner.legacy_energy_sum < global_energy:
        global_energy, global_path = winner.legacy_energy_sum, winner.cluster_heads
    pheromone_update(pheromone, parameters.pheromone_intensity, list(winner.cluster_heads), chaos, alpha, dist_matrix)
    next_chaos = tuple(value * (1 - value) * parameters.chaos_r for value in chaos)
    next_state = ACACOPhase1State(
        np.clip(pheromone * (1 - p), 0.1, 10.0), next_chaos, p, beta, alpha, hopping,
        min(state.l_best, path_length(list(winner.cluster_heads), dist_matrix)),
        state.legacy_fitness_total + winner.legacy_energy_sum, global_energy, global_path,
    )
    return ACACOPhase1Result(tuple(candidates), winner.cluster_heads, best_index, winner.legacy_energy_sum, next_state)


def _evaluate_candidate(
    ant_index, start, nodes, dist_matrix, base_dists, residual_e, live_nodes, pheromone,
    chaos, beta, alpha, hopping, heuristic, parameters, common,
) -> ACACOCandidateFitness | None:
    timeout = 0
    while timeout < parameters.path_making_timeout:
        path = make_path(
            parameters.ch_proportion * len(nodes), start, live_nodes, pheromone, dist_matrix,
            heuristic, residual_e, parameters.pheromone_weight, beta, alpha, chaos,
        )
        if path is None:
            timeout += 1
            continue
        legacy_topology = network_config(
            nodes, path, common.communication_radius, common.d0, parameters.base_pos,
            hopping, base_dists, dist_matrix, residual_e,
        )
        if legacy_topology is None:
            timeout += 1
            continue
        _, e_sum = energy_consumption(
            nodes, legacy_topology, common.d0, common.bit_count, common.ctrl_bit,
            base_dists, dist_matrix, common.e_elec, common.e_agg,
            common.free_space_coeff, common.multipath_coeff,
        )
        pheromone_update(pheromone, parameters.pheromone_intensity, path, chaos, alpha, dist_matrix)
        return ACACOCandidateFitness(ant_index, start, tuple(path), e_sum)
    return None


def _heuristic_matrix(node_count, dist_matrix, parameters, common):
    values = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    for source in range(node_count):
        for target in range(node_count):
            if source != target:
                cost = E_m(common.e_elec, common.free_space_coeff, common.e_agg, common.multipath_coeff,
                           common.bit_count, 0, 0, common.ctrl_bit, 0, dist_matrix[source][target], common.d0)
                values[source][target] = (1 / cost) ** parameters.gamma
    return values


@contextmanager
def _baseline_random_stream(rng):
    """Serialize the legacy module-level RNG bridge around one candidate loop."""

    with _BASELINE_RANDOM_LOCK:
        original = random.getstate()
        random.setstate(rng.getstate())
        try:
            yield
        finally:
            rng.setstate(random.getstate())
            random.setstate(original)


def _validate_parameters(node_count, parameters):
    if not isinstance(parameters, ACACOParameters) or node_count <= 0:
        raise ACACOPhase1Error("valid AC-ACO parameters and a non-empty network are required")
    if parameters.t_max <= 0 or parameters.candidate_count <= 0 or parameters.path_making_timeout <= 0:
        raise ACACOPhase1Error("AC-ACO iteration, candidate, and timeout bounds must be positive")
    if not math.isfinite(parameters.adaptive_energy_upper_bound) or parameters.adaptive_energy_upper_bound <= 0:
        raise ACACOPhase1Error("AC-ACO adaptive energy upper bound must be finite and positive")


def _validate_inputs(node_count, dist_matrix, base_dists, residual_e, live_nodes, state, common, round_index):
    if not isinstance(state, ACACOPhase1State) or not isinstance(common, PureMRPParameters):
        raise ACACOPhase1Error("AC-ACO Phase I requires complete state and common parameters")
    if len(dist_matrix) != node_count or any(len(row) != node_count for row in dist_matrix):
        raise ACACOPhase1Error("AC-ACO requires a square distance matrix")
    if len(base_dists) != node_count or len(residual_e) != node_count or not live_nodes:
        raise ACACOPhase1Error("AC-ACO state dimensions or live nodes are invalid")
    if np.shape(state.pheromone_matrix) != (node_count, node_count) or len(state.chaos) != node_count:
        raise ACACOPhase1Error("AC-ACO pheromone or chaos state dimensions are invalid")
    if not isinstance(round_index, int) or round_index < 0:
        raise ACACOPhase1Error("round index must be a non-negative integer")


def _require_stateful_rng(rng):
    if not callable(getattr(rng, "random", None)) or not callable(getattr(rng, "getstate", None)) or not callable(getattr(rng, "setstate", None)):
        raise ACACOPhase1Error("AC-ACO requires a caller-owned stateful random.Random-compatible RNG")
