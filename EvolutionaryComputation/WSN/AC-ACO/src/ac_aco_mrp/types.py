"""Contracts for the import-safe Hybrid integration boundary."""

from collections.abc import Mapping
from dataclasses import dataclass

from mrp.runner_types import MRPCHRoutingResult, PureMRPParameters, PureMRPState
from mrp.phase2.control_energy import AntControlEnergy


@dataclass(frozen=True)
class ACACOParameters:
    """Baseline AC-ACO constants, retained for CH selection only."""

    base_pos: object
    adaptive_energy_upper_bound: float
    candidate_count: int = 10
    path_making_timeout: int = 100
    ch_proportion: float = 0.1
    pheromone_initial: float = 1.0
    pheromone_intensity: float = 100.0
    pheromone_weight: float = 1.0
    gamma: float = 0.1
    chaos_r: float = 3.61
    t_max: int = 3500
    p_min: float = 0.1
    p_max: float = 0.9
    beta_min: float = 1.0
    beta_max: float = 5.0
    sigmoid_k: float = 5.0
    alpha_min: float = 0.05
    alpha_max: float = 0.3
    hopping_factor_min: float = 0.67
    hopping_factor_max: float = 0.9
    initial_p: float = 0.0
    initial_beta: float = 3.0
    initial_alpha: float = 0.0
    initial_hopping_factor: float = 0.4
    initial_l_best: float = 1000.0


@dataclass(frozen=True)
class ACACOPhase1State:
    """AC-ACO-only mutable algorithm state, stored as an immutable snapshot."""

    pheromone_matrix: object
    chaos: tuple[float, ...]
    p: float
    beta: float
    alpha: float
    hopping_factor: float
    l_best: float
    legacy_fitness_total: float = 0.0
    global_best_energy: float = 1e9
    global_best_path: tuple[int, ...] = ()


@dataclass(frozen=True)
class ACACOCandidateFitness:
    """Read-only legacy-Greedy fitness trace; no topology/vector is retained."""

    ant_index: int
    start_node: int
    cluster_heads: tuple[int, ...]
    legacy_energy_sum: float


@dataclass(frozen=True)
class ACACOPhase1Result:
    candidates: tuple[ACACOCandidateFitness, ...]
    selected_cluster_heads: tuple[int, ...]
    selected_candidate_index: int
    selected_legacy_energy_sum: float
    state_after: ACACOPhase1State


@dataclass(frozen=True)
class HybridParameters:
    ac_aco: ACACOParameters
    mrp: PureMRPParameters
    phase3_top_k: int | None = None


@dataclass(frozen=True)
class DirectMRPTopKParameters:
    """Project-adaptation settings for Direct-first with MRP fallback."""

    ac_aco: ACACOParameters
    mrp: PureMRPParameters
    phase3_top_k: int


@dataclass(frozen=True)
class HybridState:
    """Separate AC-ACO and MRP/physical state; their pheromones never mix."""

    physical_state: PureMRPState
    ac_aco_state: ACACOPhase1State


@dataclass(frozen=True)
class HybridRoundContext:
    """External Phase-II data; hop counts are never synthesized by Hybrid."""

    hop_counts: Mapping[int, int]


HybridCHRoutingResult = MRPCHRoutingResult


@dataclass(frozen=True)
class HybridSelectedFlow:
    """One independently selected CH-to-Sink physical flow."""

    source_ch: int
    route: tuple[int, ...]


@dataclass(frozen=True)
class HybridMultiFlowPlan:
    """Final Hybrid routing without a global single-parent-tree requirement."""

    selected_flows: tuple[HybridSelectedFlow, ...]
    member_assignments: Mapping[int, tuple[int, ...]]
    physical_sensor_ids: tuple[int, ...]
    local_payload_flow_by_sensor: Mapping[int, int]


@dataclass(frozen=True)
class HybridMultiFlowEnergy:
    """Read-only final physical-energy ledger for one multi-flow plan."""

    e_m_list: tuple[float, ...]
    e_sum: float
    edge_payload_bits: Mapping[tuple[int, int], float]


@dataclass(frozen=True)
class HybridRoundResult:
    round_index: int
    phase1_result: ACACOPhase1Result | None
    members_by_head: Mapping[int, tuple[int, ...]] | None
    ch_routing_results: tuple[HybridCHRoutingResult, ...]
    selected_routes: Mapping[int, tuple[int, ...]] | None
    final_topology: object | None
    final_plan: HybridMultiFlowPlan | None
    e_m_list: tuple[float, ...] | None
    e_sum: float | None
    state_before: HybridState
    state_after: HybridState
    newly_dead: tuple[int, ...]
    success: bool
    failure_stage: str | None
    failure_reason: str | None
    failed_cluster_head: int | None = None
    routing_failure_classification: str | None = None
    failed_phase2_result: object | None = None
    data_energy: float | None = None
    ant_control_energy: AntControlEnergy | None = None


@dataclass(frozen=True)
class DirectMRPTopKDiagnostics:
    """Per-round proof that directly reachable CHs skipped MRP discovery."""

    direct_cluster_heads: tuple[int, ...]
    fallback_cluster_heads: tuple[int, ...]
    mrp_discoveries: int
    cached_route_uses: int
    routes_before_top_k: int
    routes_after_top_k: int
    routes_pruned: int
    sant_count: int
    bant_count: int
    aant_count: int


@dataclass(frozen=True)
class DirectMRPTopKRoundResult(HybridRoundResult):
    """Result for the project Direct-first + MRP-fallback + Top-K mode."""

    direct_mrp_diagnostics: DirectMRPTopKDiagnostics | None = None


@dataclass(frozen=True)
class HybridRunResult:
    rounds: tuple[HybridRoundResult, ...]
    final_state: HybridState


@dataclass(frozen=True)
class LifetimeRouteCandidate:
    """Selected-route metric adapter for experiment reporting."""

    path_length: float


@dataclass(frozen=True)
class LifetimeAwareRouteSelection:
    """Experimental non-probabilistic replacement for one CH's Phase III."""

    candidates: tuple[LifetimeRouteCandidate, ...]
    selected_route: tuple[int, ...]
    selected_index: int = 0


@dataclass(frozen=True)
class LifetimeAwareCHRoutingResult:
    """Phase-II candidates plus the route chosen by the joint selector."""

    cluster_head: int
    phase2_result: object
    selection_result: LifetimeAwareRouteSelection

    @property
    def phase3_result(self) -> LifetimeAwareRouteSelection:
        """Compatibility view for existing read-only metrics adapters."""

        return self.selection_result


@dataclass(frozen=True)
class LifetimeSelectionResult:
    """Winning complete route set and its lexicographic objective values."""

    selected_routes: Mapping[int, tuple[int, ...]]
    plan: HybridMultiFlowPlan
    energy: HybridMultiFlowEnergy
    minimum_residual_after_round: float
    total_hop_count: int
    candidate_sets_evaluated: int
    branches_visited: int = 0
    branches_pruned: int = 0
    theoretical_combinations: int = 0


@dataclass(frozen=True)
class HybridLifetimeRoundResult(HybridRoundResult):
    """Hybrid round tagged with the experimental joint lifetime selection."""

    lifetime_selection: LifetimeSelectionResult | None = None
