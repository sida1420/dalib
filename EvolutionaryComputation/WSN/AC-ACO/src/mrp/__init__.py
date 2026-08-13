"""MRP compatibility foundation.

The package currently provides Phase I dynamic-clustering core, Phase II
Eq. (26)â€“(31) mathematical primitives, and contracts that adapt later
algorithm decisions to the existing simulator's ``Node(-1)`` topology
boundary.  SANT/AANT forwarding is present; BANT, route discovery orchestration,
global pheromone feedback, and Phase III remain intentionally absent.
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
)
from .phase2.aant import choose_aant_sensor_neighbor, should_create_aant
from .phase2.sant import SANTHopTrace, SANTRouteResult, construct_normal_sant_route
from .phase2.sant_validation import SANTInputError
from .phase2.transition import NoValidTransitionError, transition_probabilities
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
    "SANTHopTrace",
    "SANTInputError",
    "SANTRouteResult",
    "choose_aant_sensor_neighbor",
    "calculate_eta",
    "calculate_local_pheromone_deposit",
    "initialize_sensor_pheromone_state",
    "calculate_mu",
    "calculate_theta",
    "construct_normal_sant_route",
    "should_create_aant",
    "NoValidTransitionError",
    "transition_probabilities",
    "update_local_pheromone",
    "TopologyValidationError",
    "assign_members_to_cluster_heads",
    "materialize_topology",
    "validate_topology",
]
