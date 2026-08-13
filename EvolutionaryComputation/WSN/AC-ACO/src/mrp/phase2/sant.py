"""MRP_PHASE_II SANT construction with the Algorithm-2 AANT branch only."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from mrp.config import MRPConfig
from mrp.phase2.aant import choose_aant_sensor_neighbor, should_create_aant
from mrp.phase2.heuristic import HeuristicBounds, calculate_eta, calculate_mu
from mrp.phase2.pheromone import update_local_pheromone
from mrp.phase2.sant_validation import (
    SANTInputError,
    draw_unit_interval,
    eligible_sensor_neighbors,
    sink_is_reachable,
    validate_sant_route_inputs,
)
from mrp.phase2.transition import NoValidTransitionError, transition_probabilities


@dataclass(frozen=True)
class SANTHopTrace:
    """Read-only audit record for one Algorithm-2 forwarding decision."""

    mode: str
    current_node: int
    valid_candidates: tuple[int, ...]
    tau: dict[int, float]
    mu: dict[int, float]
    eta: dict[int, float]
    probabilities: dict[int, float]
    sampled_next_hop: int
    ttl_before: int
    ttl_after: int
    visited_nodes: tuple[int, ...]
    local_tau_before: float | None
    local_tau_after: float | None


@dataclass(frozen=True)
class SANTRouteResult:
    """One SANT route attempt; terminal ``-1`` denotes the existing Sink."""

    path: tuple[int, ...]
    success: bool
    visited_nodes: tuple[int, ...]
    remaining_ttl: int
    pheromone_state: dict[int, dict[int, float]]
    trace: tuple[SANTHopTrace, ...]
    failure_reason: str | None = None


def construct_normal_sant_route(
    start_ch: int,
    live_nodes: Collection[int],
    residual_e: Sequence[float],
    dist_matrix: Sequence[Sequence[float]],
    base_dists: Sequence[float],
    hop_counts: Mapping[int, int],
    communication_radius: float,
    pheromone_state: Mapping[int, Mapping[int, float]],
    bounds: HeuristicBounds,
    config: MRPConfig,
    lambda_coefficient: float,
    ttl: int,
    rng: object,
) -> SANTRouteResult:
    """Construct one SANT route using normal or Algorithm-2 AANT forwarding.

    The direct-Sink test is a SIMULATOR_COMPATIBILITY_ASSUMPTION, not paper
    Sink mathematics. AANT candidate safety is also a compatibility mapping.
    The paper's visited-node backtracking details remain underspecified; no
    unprovided DFS behavior is added.
    """

    live_set = validate_sant_route_inputs(
        start_ch, live_nodes, residual_e, dist_matrix, base_dists,
        communication_radius, pheromone_state, bounds, config, ttl, rng,
    )
    state = _copy_pheromone_state(pheromone_state, live_set)
    path, trace = [start_ch], []
    current, remaining_ttl = start_ch, ttl
    while remaining_ttl > 0:
        if sink_is_reachable(current, base_dists, communication_radius):
            trace.append(_terminal_trace(current, remaining_ttl, tuple(path)))
            return _result(path + [-1], True, remaining_ttl - 1, state, trace)

        candidates = eligible_sensor_neighbors(
            current, live_set, path, dist_matrix, communication_radius
        )
        if should_create_aant(config, rng):
            if not candidates:
                return _result(path, False, remaining_ttl, state, trace, "no_valid_aant_sensor_neighbor")
            next_hop = choose_aant_sensor_neighbor(candidates, rng)
            remaining_ttl -= 1
            path.append(next_hop)
            trace.append(SANTHopTrace(
                "AANT", current, candidates, {}, {}, {}, {}, next_hop,
                remaining_ttl + 1, remaining_ttl, tuple(path), None, None,
            ))
            current = next_hop
            continue
        if not candidates:
            return _result(path, False, remaining_ttl, state, trace, "no_valid_sensor_neighbor")
        outgoing = _outgoing_values(state, current)
        tau = _candidate_tau_values(outgoing, candidates)
        mu = {
            candidate: calculate_mu(
                current, candidate, residual_e, dist_matrix, base_dists, hop_counts, config
            )
            for candidate in candidates
        }
        eta = {candidate: calculate_eta(value, bounds) for candidate, value in mu.items()}
        try:
            probabilities = transition_probabilities(
                current, candidates, path, tau, eta, dist_matrix, communication_radius, config
            )
        except NoValidTransitionError:
            return _result(path, False, remaining_ttl, state, trace, "no_positive_transition_weight")
        next_hop = _sample(probabilities, rng)
        old_tau = tau[next_hop]
        new_tau = update_local_pheromone(
            current, next_hop, old_tau, residual_e, dist_matrix, lambda_coefficient, config
        )
        state[current][next_hop] = new_tau
        remaining_ttl -= 1
        path.append(next_hop)
        trace.append(SANTHopTrace(
            "NORMAL_SANT", current, candidates, tau, mu, eta, probabilities, next_hop,
            remaining_ttl + 1, remaining_ttl, tuple(path), old_tau, new_tau,
        ))
        current = next_hop
    return _result(path, False, remaining_ttl, state, trace, "ttl_exhausted")


def _copy_pheromone_state(
    pheromone_state: Mapping[int, Mapping[int, float]], live_nodes: Collection[int],
) -> dict[int, dict[int, float]]:
    try:
        state = {source: dict(outgoing) for source, outgoing in pheromone_state.items()}
    except (AttributeError, TypeError, ValueError) as error:
        raise SANTInputError("pheromone state must map source IDs to outgoing link values") from error
    live_set = set(live_nodes)
    if any(source_id not in live_set for source_id in state):
        raise SANTInputError("pheromone state cannot contain Sink, dead, or invalid source IDs")
    if any(candidate_id not in live_set for outgoing in state.values() for candidate_id in outgoing):
        raise SANTInputError("pheromone state cannot contain Sink, dead, or invalid destination IDs")
    return state


def _outgoing_values(state: Mapping[int, Mapping[int, float]], source_id: int) -> Mapping[int, float]:
    if source_id not in state or not isinstance(state[source_id], Mapping):
        raise SANTInputError(f"missing outgoing pheromone state for sensor {source_id}")
    return state[source_id]


def _candidate_tau_values(
    outgoing: Mapping[int, float], candidates: Sequence[int],
) -> dict[int, float]:
    missing = [candidate for candidate in candidates if candidate not in outgoing]
    if missing:
        raise SANTInputError(f"missing pheromone value for sensor link to {missing[0]}")
    return {candidate: outgoing[candidate] for candidate in candidates}


def _sample(probabilities: Mapping[int, float], rng: object) -> int:
    draw = draw_unit_interval(rng)
    cumulative = 0.0
    for candidate, probability in probabilities.items():
        cumulative += probability
        if draw < cumulative:
            return candidate
    return next(reversed(probabilities))


def _terminal_trace(current: int, ttl: int, path: tuple[int, ...]) -> SANTHopTrace:
    return SANTHopTrace("TERMINAL_SINK", current, (), {}, {}, {}, {}, -1, ttl, ttl - 1, path, None, None)


def _result(
    path: list[int], success: bool, ttl: int, state: dict[int, dict[int, float]],
    trace: list[SANTHopTrace], failure_reason: str | None = None,
) -> SANTRouteResult:
    return SANTRouteResult(tuple(path), success, tuple(node for node in path if node != -1), ttl, state, tuple(trace), failure_reason)
