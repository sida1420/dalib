"""Import-safe Hybrid AC-ACO Phase I plus MRP Phase II/III integration."""

from .phase1 import (
    ACACOPhase1Error,
    initialize_ac_aco_phase1_state,
    select_ac_aco_cluster_heads,
)
from .run_hybrid import run_hybrid, run_hybrid_round
from .run_hybrid_lifetime import run_hybrid_lifetime_round
from .run_direct_mrp_topk import run_direct_mrp_topk_round
from .types import (
    ACACOCandidateFitness,
    ACACOParameters,
    ACACOPhase1Result,
    ACACOPhase1State,
    DirectMRPTopKDiagnostics,
    DirectMRPTopKParameters,
    DirectMRPTopKRoundResult,
    HybridCHRoutingResult,
    HybridMultiFlowEnergy,
    HybridMultiFlowPlan,
    HybridParameters,
    HybridRoundContext,
    HybridRoundResult,
    HybridRunResult,
    HybridState,
    HybridSelectedFlow,
    HybridLifetimeRoundResult,
    LifetimeSelectionResult,
)

__all__ = [
    "ACACOCandidateFitness",
    "ACACOParameters",
    "ACACOPhase1Error",
    "ACACOPhase1Result",
    "ACACOPhase1State",
    "DirectMRPTopKDiagnostics",
    "DirectMRPTopKParameters",
    "DirectMRPTopKRoundResult",
    "HybridCHRoutingResult",
    "HybridLifetimeRoundResult",
    "HybridMultiFlowEnergy",
    "HybridMultiFlowPlan",
    "HybridParameters",
    "HybridRoundContext",
    "HybridRoundResult",
    "HybridRunResult",
    "HybridState",
    "HybridSelectedFlow",
    "LifetimeSelectionResult",
    "initialize_ac_aco_phase1_state",
    "run_hybrid",
    "run_hybrid_round",
    "run_hybrid_lifetime_round",
    "run_direct_mrp_topk_round",
    "select_ac_aco_cluster_heads",
]
