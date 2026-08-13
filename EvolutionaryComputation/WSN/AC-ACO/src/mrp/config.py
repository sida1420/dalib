"""Source-confirmed MRP settings; no algorithm is executed here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MRPConfig:
    """MRP settings whose values are confirmed by Yang et al. (2010).

    All fields below are PAPER_SIMULATION_PARAMETER except
    ``aant_probability``, which is a PAPER_ALGORITHM_RULE from Algorithm 2.
    Unresolved paper quantities (for example RSS threshold, q, TTL, and
    pheromone bounds) are intentionally absent rather than guessed.
    """

    k1: float = 0.5
    k2: float = 0.1
    k3: float = 0.4
    k4: float = 2.0
    k5: float = 1.0
    k6: float = 1.0
    k7: float = 0.4
    k8: float = 0.2
    k9: float = 0.4
    k10: float = 0.5
    k11: float = 0.3
    k12: float = 0.2
    alpha: float = 2.0
    beta: float = 2.0
    rho: float = 0.2
    initial_pheromone: float = 0.01
    aant_probability: float = 0.001
    event_radius: float = 20.0
