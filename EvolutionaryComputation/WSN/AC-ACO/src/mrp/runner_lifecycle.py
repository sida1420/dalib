"""Runner-owned state commits that mirror the baseline after final evaluation."""

from collections.abc import Collection, Mapping, Sequence
import math

from mrp.config import MRPConfig
from mrp.phase2.pheromone import initialize_sensor_pheromone_state
from mrp.runner_types import PheromoneLifecycle, PureMRPState


class PureMRPIntegrationError(ValueError):
    """Raised when a runner snapshot cannot safely cross an integration boundary."""


def validate_pure_mrp_state(
    state: PureMRPState,
    node_count: int,
    dist_matrix: Sequence[Sequence[float]],
    communication_radius: float,
    config: MRPConfig,
) -> None:
    """Reject public snapshots that cannot represent one complete network state."""

    if not isinstance(state.residual_e, tuple) or len(state.residual_e) != node_count:
        raise PureMRPIntegrationError("residual energy must be a node-sized tuple")
    if not isinstance(state.live_nodes, tuple) or not isinstance(state.dead_nodes, frozenset):
        raise PureMRPIntegrationError("live and dead nodes must be immutable snapshots")
    if not _finite_non_negative(state.cumulative_actual_energy):
        raise PureMRPIntegrationError("cumulative actual energy must be finite and non-negative")
    if any(not _finite_non_negative(energy) for energy in state.residual_e):
        raise PureMRPIntegrationError("residual energy must be finite and non-negative")
    live_nodes, dead_nodes = set(state.live_nodes), set(state.dead_nodes)
    if len(live_nodes) != len(state.live_nodes) or not _valid_sensor_ids(live_nodes, node_count):
        raise PureMRPIntegrationError("live nodes must be unique valid sensor IDs")
    if not _valid_sensor_ids(dead_nodes, node_count):
        raise PureMRPIntegrationError("dead nodes must be valid sensor IDs")
    if live_nodes & dead_nodes or live_nodes | dead_nodes != set(range(node_count)):
        raise PureMRPIntegrationError("live and dead nodes must form one complete partition")
    if any(state.residual_e[sensor_id] <= 0 for sensor_id in live_nodes):
        raise PureMRPIntegrationError("live nodes must have positive residual energy")
    if any(state.residual_e[sensor_id] != 0 for sensor_id in dead_nodes):
        raise PureMRPIntegrationError("dead nodes must have zero residual energy")
    _validate_persisted_pheromone(
        state.pheromone_state, state.live_nodes, dist_matrix, communication_radius, config
    )


def pheromone_for_discovery(
    state: PureMRPState,
    policy: PheromoneLifecycle,
    dist_matrix: Sequence[Sequence[float]],
    communication_radius: float,
    config: MRPConfig,
) -> dict[int, dict[int, float]]:
    """Return source-initialized or caller-approved persisted link pheromone."""

    if policy is PheromoneLifecycle.RESET_PER_DISCOVERY or state.pheromone_state is None:
        return initialize_sensor_pheromone_state(
            state.live_nodes, dist_matrix, communication_radius, config
        )
    if policy is not PheromoneLifecycle.PERSIST_ACROSS_ROUNDS:
        raise PureMRPIntegrationError("pheromone lifecycle policy is required")
    return _live_link_snapshot(state.pheromone_state, state.live_nodes)


def apply_baseline_lifecycle(
    state: PureMRPState,
    e_m_list: Sequence[float],
    e_sum: float,
    persisted_pheromone: Mapping[int, Mapping[int, float]] | None,
) -> tuple[PureMRPState, tuple[int, ...]]:
    """Mirror AC_ACO.py: clamp residuals, then swap-remove depleted live nodes."""

    if len(e_m_list) != len(state.residual_e) or not _finite_non_negative(e_sum):
        raise PureMRPIntegrationError("final energy result is incompatible with current state")
    if any(not _finite_non_negative(energy) for energy in e_m_list):
        raise PureMRPIntegrationError("per-node final energy must be finite and non-negative")
    residual_after = tuple(
        max(0, energy - consumed)
        for energy, consumed in zip(state.residual_e, e_m_list)
    )
    live_after, dead_after, newly_dead = list(state.live_nodes), set(state.dead_nodes), []
    index = 0
    while index < len(live_after):
        sensor_id = live_after[index]
        if residual_after[sensor_id] <= 0:
            dead_after.add(sensor_id)
            newly_dead.append(sensor_id)
            live_after[index], live_after[-1] = live_after[-1], live_after[index]
            live_after.pop()
            index -= 1
        index += 1
    return (
        PureMRPState(
            residual_after, tuple(live_after), frozenset(dead_after),
            state.cumulative_actual_energy + e_sum,
            _live_link_snapshot(persisted_pheromone, live_after),
        ),
        tuple(newly_dead),
    )


def combine_data_and_control_energy(data_e_m, data_sum, control_energy):
    """Combine independent radio ledgers without changing either formula."""

    if control_energy is None:
        return tuple(data_e_m), data_sum
    if len(data_e_m) != len(control_energy.e_m_list):
        raise PureMRPIntegrationError("data and control energy vectors must match")
    combined = tuple(
        math.fsum((data, control))
        for data, control in zip(data_e_m, control_energy.e_m_list)
    )
    return combined, math.fsum((data_sum, control_energy.total_energy))


def commit_control_energy_after_failure(state, control_energy):
    """Keep transmitted-ant cost while rolling back failed routing/pheromone work."""

    if control_energy is None or control_energy.is_zero:
        return state, ()
    return apply_baseline_lifecycle(
        state, control_energy.e_m_list, control_energy.total_energy,
        state.pheromone_state,
    )


def copy_pheromone_state(
    state: Mapping[int, Mapping[int, float]] | None,
) -> dict[int, dict[int, float]] | None:
    return None if state is None else {source_id: dict(outgoing) for source_id, outgoing in state.items()}


def _live_link_snapshot(
    state: Mapping[int, Mapping[int, float]] | None, live_nodes: Collection[int],
) -> dict[int, dict[int, float]] | None:
    if state is None:
        return None
    live_set = set(live_nodes)
    try:
        return {
            source_id: {
                destination_id: value
                for destination_id, value in outgoing.items()
                if destination_id in live_set
            }
            for source_id, outgoing in state.items()
            if source_id in live_set
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise PureMRPIntegrationError("persisted pheromone state must map sensor links") from error


def _finite_non_negative(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _valid_sensor_ids(sensor_ids: Collection[int], node_count: int) -> bool:
    return all(
        isinstance(sensor_id, int) and not isinstance(sensor_id, bool)
        and 0 <= sensor_id < node_count
        for sensor_id in sensor_ids
    )


def _validate_persisted_pheromone(
    pheromone_state: Mapping[int, Mapping[int, float]] | None,
    live_nodes: tuple[int, ...],
    dist_matrix: Sequence[Sequence[float]],
    communication_radius: float,
    config: MRPConfig,
) -> None:
    if pheromone_state is None:
        return
    if not isinstance(pheromone_state, Mapping):
        raise PureMRPIntegrationError("persisted pheromone state must map sensor links")
    try:
        snapshot = {source_id: dict(outgoing) for source_id, outgoing in pheromone_state.items()}
        expected = initialize_sensor_pheromone_state(
            live_nodes, dist_matrix, communication_radius, config
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise PureMRPIntegrationError("persisted pheromone state must map valid live sensor links") from error
    node_count = len(dist_matrix)
    if not _valid_sensor_ids(set(snapshot), node_count) or any(
        not _valid_sensor_ids(set(outgoing), node_count)
        for outgoing in snapshot.values()
    ):
        raise PureMRPIntegrationError("persisted pheromone state must use valid sensor IDs")
    if set(snapshot) != set(expected) or any(
        set(snapshot[source_id]) != set(expected[source_id]) for source_id in expected
    ):
        raise PureMRPIntegrationError("persisted pheromone state must cover every live sensor link")
    if any(
        not _finite_non_negative(value)
        for outgoing in snapshot.values() for value in outgoing.values()
    ):
        raise PureMRPIntegrationError("persisted pheromone values must be finite and non-negative")
