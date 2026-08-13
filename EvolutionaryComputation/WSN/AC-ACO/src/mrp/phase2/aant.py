"""MRP_PHASE_II Algorithm-2 abnormal-ant trigger and random selection."""

from collections.abc import Sequence
import math

from mrp.config import MRPConfig
from mrp.phase2.sant_validation import SANTInputError, draw_unit_interval


def should_create_aant(config: MRPConfig, rng: object) -> bool:
    """Algorithm 2: create AANT when ``RAND(x) < aant_probability``."""

    if not isinstance(config, MRPConfig):
        raise SANTInputError("AANT requires an MRPConfig")
    probability = config.aant_probability
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise SANTInputError("AANT probability must be numeric")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise SANTInputError("AANT probability must be finite and in [0, 1]")
    return draw_unit_interval(rng) < probability


def choose_aant_sensor_neighbor(candidates: Sequence[int], rng: object) -> int:
    """Uniformly choose from already-validated eligible sensor candidates.

    Algorithm 2 says only that AANT randomly chooses a node. The caller's
    candidate set is a SIMULATOR_COMPATIBILITY_ASSUMPTION, not a paper rule.
    """

    if not candidates:
        raise SANTInputError("AANT cannot choose from an empty sensor candidate set")
    return candidates[int(draw_unit_interval(rng) * len(candidates))]
