"""Experiment-layer hop-count snapshots for MRP scenario execution."""

from collections import deque
from collections.abc import Collection
from dataclasses import dataclass
import math

from .config import ExperimentNetwork, ExperimentRoundScenario


class HopCountInputError(ValueError):
    """Raised when a hop-count snapshot cannot be built from experiment state."""


@dataclass(frozen=True)
class HopCountSnapshot:
    """Live sensor hop counts and explicitly unreachably sensors.

    ``h_i`` is the number of forwarding hops from sensor ``i`` to Sink.
    A direct sensor-to-Sink edge is therefore ``1``: the current SANT
    implementation consumes one TTL for that terminal edge.
    """

    hop_counts: dict[int, int]
    unreachable_nodes: tuple[int, ...]


def derive_hop_counts(
    network: ExperimentNetwork, live_nodes: Collection[int], communication_radius: float,
) -> HopCountSnapshot:
    """Use deterministic BFS over existing live sensor-neighbor links.

    This is a SIMULATOR_IMPLEMENTATION_MAPPING, deliberately outside MRP
    heuristic code.  It reuses the supplied distance snapshots and creates no
    geometry or routing fallback.
    """

    _validate_inputs(network, live_nodes, communication_radius)
    live = tuple(sorted(set(live_nodes)))
    direct = tuple(node_id for node_id in live if network.base_dists[node_id] <= communication_radius)
    hops = {node_id: 1 for node_id in direct}
    queue = deque(direct)
    while queue:
        current = queue.popleft()
        for candidate in live:
            if candidate in hops or candidate == current:
                continue
            if network.dist_matrix[current][candidate] <= communication_radius:
                hops[candidate] = hops[current] + 1
                queue.append(candidate)
    return HopCountSnapshot(hops, tuple(node_id for node_id in live if node_id not in hops))


def scenario_with_current_hops(
    scenario: ExperimentRoundScenario,
    network: ExperimentNetwork,
    live_nodes: Collection[int],
    communication_radius: float,
) -> tuple[ExperimentRoundScenario, HopCountSnapshot]:
    """Keep frozen event inputs while refreshing only the live hop snapshot."""

    snapshot = derive_hop_counts(network, live_nodes, communication_radius)
    return ExperimentRoundScenario(
        scenario.event_nodes, dict(scenario.event_signal_strengths), snapshot.hop_counts, scenario.scenario_id,
    ), snapshot


def _validate_inputs(network: ExperimentNetwork, live_nodes: Collection[int], communication_radius: float) -> None:
    if not isinstance(network, ExperimentNetwork):
        raise HopCountInputError("hop-count provider requires an ExperimentNetwork")
    if isinstance(communication_radius, bool) or not isinstance(communication_radius, (int, float)):
        raise HopCountInputError("communication radius must be numeric")
    if not math.isfinite(communication_radius) or communication_radius <= 0:
        raise HopCountInputError("communication radius must be finite and positive")
    node_count = len(network.nodes)
    try:
        live = set(live_nodes)
    except TypeError as error:
        raise HopCountInputError("live nodes must be a collection") from error
    if any(not isinstance(node_id, int) or isinstance(node_id, bool) or not 0 <= node_id < node_count for node_id in live):
        raise HopCountInputError("live nodes must be valid sensor IDs")
