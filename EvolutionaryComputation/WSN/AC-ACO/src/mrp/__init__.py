"""MRP compatibility foundation.

The package currently provides Phase I dynamic-clustering core, Phase II
Eq. (26)â€“(31) mathematical primitives, and contracts that adapt later
algorithm decisions to the existing simulator's ``Node(-1)`` topology
boundary.  SANT/AANT forwarding and BANT feedback are present; route discovery
orchestration, Phase-III path selection, event-reference integration, and the
explicit network-wide comparison adaptation are present.
"""

from .config import MRPConfig
from .phase1.clustering import (
    ClusterFormationInputError,
    calculate_cluster_head_score,
    calculate_cluster_head_timer,
    select_cluster_head,
)
from .phase2.heuristic import (
    HeuristicBounds,
    HeuristicInputError,
    calculate_eta,
    calculate_mu,
    calculate_theta,
)
from .phase2.pheromone import (
    PheromoneInputError,
    calculate_local_pheromone_deposit,
    initialize_sensor_pheromone_state,
    update_local_pheromone,
    update_pheromone_with_deposit,
    validate_pheromone_config,
)
from .phase2.bant_feedback import (
    BANTFeedbackInputError,
    BANTFeedbackResult,
    BANTLinkFeedbackTrace,
    apply_bant_global_feedback,
    calculate_global_pheromone_delta,
)
from .phase2.aant import choose_aant_sensor_neighbor, should_create_aant
from .phase2.sant import SANTHopTrace, SANTRouteResult, construct_normal_sant_route
from .phase2.sant_validation import SANTInputError
from .phase2.transition import NoValidTransitionError, transition_probabilities
from .phase2.route_quality import (
    BANTPreparation,
    RouteQuality,
    RouteQualityInputError,
    evaluate_route_quality,
    prepare_bant,
)
from .phase2.routing import (
    MRPPhase2AntResult,
    MRPPhase2InputError,
    MRPPhase2Result,
    discover_mrp_phase2_routes,
)
from .phase3.path_selection import (
    MRPPhase3Candidate,
    MRPPhase3InputError,
    MRPPhase3Result,
    calculate_phase3_fitness,
    select_phase3_route,
)
from .multi_ch_routing import (
    MRPMultiCHRoutingResult,
    MRPRoutingFailure,
    route_cluster_heads_with_mrp,
)
from .network_wide_clustering import (
    NETWORK_WIDE_MRP_ADAPTATION,
    NETWORK_WIDE_SELECTION_POLICY,
    NetworkWideClusterFormationResult,
    NetworkWideMRPClusteringError,
    configured_target_cluster_head_count,
    select_network_wide_cluster_heads,
)
from .run_mrp import run_pure_mrp, run_pure_mrp_round
from .runner_lifecycle import PureMRPIntegrationError
from .runner_types import (
    PheromoneLifecycle,
    MRPCHRoutingResult,
    PureMRPParameters,
    PureMRPRoundContext,
    PureMRPRoundResult,
    PureMRPRunResult,
    PureMRPState,
)
from .topology import (
    TopologyValidationError,
    assign_members_to_cluster_heads,
    materialize_topology,
    validate_topology,
)
from .types import ClusterFormationResult

__all__ = [
    "MRPConfig",
    "ClusterFormationInputError",
    "ClusterFormationResult",
    "calculate_cluster_head_score",
    "calculate_cluster_head_timer",
    "select_cluster_head",
    "HeuristicBounds",
    "HeuristicInputError",
    "PheromoneInputError",
    "BANTFeedbackInputError",
    "BANTFeedbackResult",
    "BANTLinkFeedbackTrace",
    "BANTPreparation",
    "RouteQuality",
    "RouteQualityInputError",
    "SANTHopTrace",
    "SANTInputError",
    "SANTRouteResult",
    "MRPPhase2AntResult",
    "MRPPhase2InputError",
    "MRPPhase2Result",
    "MRPPhase3Candidate",
    "MRPPhase3InputError",
    "MRPPhase3Result",
    "PheromoneLifecycle",
    "MRPCHRoutingResult",
    "MRPMultiCHRoutingResult",
    "MRPRoutingFailure",
    "NETWORK_WIDE_MRP_ADAPTATION",
    "NETWORK_WIDE_SELECTION_POLICY",
    "NetworkWideClusterFormationResult",
    "NetworkWideMRPClusteringError",
    "PureMRPIntegrationError",
    "PureMRPParameters",
    "PureMRPRoundContext",
    "PureMRPRoundResult",
    "PureMRPRunResult",
    "PureMRPState",
    "choose_aant_sensor_neighbor",
    "calculate_eta",
    "calculate_local_pheromone_deposit",
    "initialize_sensor_pheromone_state",
    "calculate_mu",
    "calculate_theta",
    "construct_normal_sant_route",
    "apply_bant_global_feedback",
    "calculate_global_pheromone_delta",
    "evaluate_route_quality",
    "prepare_bant",
    "discover_mrp_phase2_routes",
    "calculate_phase3_fitness",
    "select_phase3_route",
    "route_cluster_heads_with_mrp",
    "configured_target_cluster_head_count",
    "select_network_wide_cluster_heads",
    "run_pure_mrp",
    "run_pure_mrp_round",
    "should_create_aant",
    "NoValidTransitionError",
    "transition_probabilities",
    "update_local_pheromone",
    "update_pheromone_with_deposit",
    "validate_pheromone_config",
    "TopologyValidationError",
    "assign_members_to_cluster_heads",
    "materialize_topology",
    "validate_topology",
]
