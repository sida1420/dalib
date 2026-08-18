"""MRP Phase II primitives and sequential multipath discovery orchestration."""

from .aant import choose_aant_sensor_neighbor, should_create_aant

from .heuristic import (
    HeuristicBounds,
    HeuristicInputError,
    calculate_eta,
    calculate_mu,
    calculate_theta,
)
from .pheromone import (
    PheromoneInputError,
    calculate_local_pheromone_deposit,
    initialize_sensor_pheromone_state,
    update_local_pheromone,
    update_pheromone_with_deposit,
    validate_pheromone_config,
)
from .bant_feedback import (
    BANTFeedbackInputError,
    BANTFeedbackResult,
    BANTLinkFeedbackTrace,
    apply_bant_global_feedback,
    calculate_global_pheromone_delta,
)
from .sant import SANTHopTrace, SANTRouteResult, construct_normal_sant_route
from .sant_validation import SANTInputError
from .transition import NoValidTransitionError, transition_probabilities
from .route_quality import BANTPreparation, RouteQuality, RouteQualityInputError, evaluate_route_quality, prepare_bant
from .routing import (
    MRPPhase2AntResult,
    MRPPhase2InputError,
    MRPPhase2Result,
    discover_mrp_phase2_routes,
)

__all__ = [
    "HeuristicBounds",
    "HeuristicInputError",
    "NoValidTransitionError",
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
    "should_create_aant",
    "transition_probabilities",
    "update_local_pheromone",
    "update_pheromone_with_deposit",
    "validate_pheromone_config",
]
