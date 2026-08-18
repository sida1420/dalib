"""Pure-MRP integration contracts; these do not replace simulator DTOs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from mrp.config import MRPConfig
from mrp.phase1.clustering import ClusterFormationResult
from mrp.phase2.heuristic import HeuristicBounds
from mrp.phase2.control_energy import AntControlEnergy
from mrp.phase2.routing import MRPPhase2Result
from mrp.phase3.path_selection import MRPPhase3Result


class PheromoneLifecycle(str, Enum):
    """Caller-selected cross-discovery policy; the paper leaves it unresolved."""

    RESET_PER_DISCOVERY = "RESET_PER_DISCOVERY"
    PERSIST_ACROSS_ROUNDS = "PERSIST_ACROSS_ROUNDS"


class _FrozenDict(dict):
    """A ``dict``-compatible immutable snapshot for existing simulator callers."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("Pure-MRP state snapshots are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable


@dataclass(frozen=True)
class PureMRPState:
    residual_e: tuple[float, ...]
    live_nodes: tuple[int, ...]
    dead_nodes: frozenset[int]
    cumulative_actual_energy: float = 0.0
    pheromone_state: Mapping[int, Mapping[int, float]] | None = None

    def __post_init__(self) -> None:
        """Detach and deep-freeze valid caller-owned pheromone snapshots."""

        if not isinstance(self.pheromone_state, Mapping):
            return
        try:
            snapshot = {
                source_id: _FrozenDict(outgoing)
                for source_id, outgoing in self.pheromone_state.items()
            }
        except (AttributeError, TypeError, ValueError):
            return
        object.__setattr__(self, "pheromone_state", _FrozenDict(snapshot))


@dataclass(frozen=True)
class PureMRPRoundContext:
    event_nodes: tuple[int, ...]
    event_signal_strengths: Mapping[int, float]
    hop_counts: Mapping[int, int]


@dataclass(frozen=True)
class MRPCHRoutingResult:
    """One Phase-II/III result produced by the shared per-CH router."""

    cluster_head: int
    phase2_result: MRPPhase2Result
    phase3_result: MRPPhase3Result


@dataclass(frozen=True)
class PureMRPParameters:
    communication_radius: float
    heuristic_bounds: HeuristicBounds
    config: MRPConfig
    lambda_coefficient: float
    ttl: int
    num_sants: int
    c0: float
    c: float
    c1: float
    d0: float
    bit_count: float
    ctrl_bit: float
    e_elec: float
    e_agg: float
    free_space_coeff: float
    multipath_coeff: float
    charge_ant_energy: bool = False
    ant_control_packet_bits: int | None = None


@dataclass(frozen=True)
class PureMRPRoundResult:
    round_index: int
    phase1_result: ClusterFormationResult | None
    phase2_result: MRPPhase2Result | None
    phase3_result: MRPPhase3Result | None
    selected_route: tuple[int, ...] | None
    final_topology: object | None
    e_m_list: tuple[float, ...] | None
    e_sum: float | None
    state_before: PureMRPState
    state_after: PureMRPState
    pheromone_before: Mapping[int, Mapping[int, float]] | None
    pheromone_after: Mapping[int, Mapping[int, float]] | None
    newly_dead: tuple[int, ...]
    success: bool
    failure_stage: str | None
    failure_reason: str | None
    data_energy: float | None = None
    ant_control_energy: AntControlEnergy | None = None


@dataclass(frozen=True)
class PureMRPRunResult:
    rounds: tuple[PureMRPRoundResult, ...]
    final_state: PureMRPState
