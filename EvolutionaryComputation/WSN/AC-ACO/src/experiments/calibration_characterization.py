"""Calibration-only scale references derived before staged parameter search."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
import random
import statistics

from mrp import PheromoneLifecycle, PureMRPParameters, PureMRPState
from mrp.phase1.clustering import select_cluster_head
from mrp.phase2.heuristic import HeuristicBounds, HeuristicInputError, calculate_mu
from mrp.phase2.routing import discover_mrp_phase2_routes

from .config import ExperimentConfig, ExperimentNetwork, derive_seed_streams
from .hop_counts import derive_hop_counts
from .scenario_generation import FrozenEventScenario


class CharacterizationError(ValueError):
    """Raised when calibration-only observations cannot establish a scale."""


@dataclass(frozen=True)
class CalibrationCharacterization:
    b_ref: float
    mu_quantiles: dict[str, float]
    hop_scale_h: int
    network_scale_s: int
    f_ref: float
    successful_reference_routes: int

    @property
    def medium_bounds(self) -> HeuristicBounds:
        return HeuristicBounds(
            self.mu_quantiles["q10"], self.mu_quantiles["q90"],
            self.mu_quantiles["q10"], self.mu_quantiles["q90"],
        )


def characterize_calibration_space(
    network: ExperimentNetwork,
    base_config: ExperimentConfig,
    scenarios: Sequence[FrozenEventScenario],
    calibration_seeds: Sequence[int],
) -> CalibrationCharacterization:
    """Derive B, μ, TTL, network, and quality scales from calibration data only."""

    if len(scenarios) < 2 or not calibration_seeds:
        raise CharacterizationError("characterization requires frozen calibration scenarios and seeds")
    parameters = base_config.mrp_parameters
    live = tuple(index for index, energy in enumerate(network.initial_residual_e) if energy > 0)
    hops = derive_hop_counts(network, live, parameters.communication_radius)
    b_ref = _median_local_base(network, live, parameters.communication_radius)
    mu_quantiles = _mu_quantiles(network, live, hops.hop_counts, parameters)
    h = _hop_scale(scenarios)
    s = math.ceil(math.sqrt(len(live)))
    if h < 1 or s < 1:
        raise CharacterizationError("calibration topology must expose positive hop and network scales")
    medium = HeuristicBounds(mu_quantiles["q10"], mu_quantiles["q90"], mu_quantiles["q10"], mu_quantiles["q90"])
    f_ref, successes = _reference_quality(
        network, base_config, scenarios, calibration_seeds,
        replace(parameters, heuristic_bounds=medium, lambda_coefficient=parameters.config.initial_pheromone / b_ref,
                ttl=2 * h, num_sants=1, c0=1.0, c=0.0, c1=0.0),
    )
    return CalibrationCharacterization(b_ref, mu_quantiles, h, s, f_ref, successes)


def _median_local_base(network, live, radius):
    values = [
        (network.initial_residual_e[source] + network.initial_residual_e[target]) / network.dist_matrix[source][target] ** 2
        for source in live for target in live
        if source != target and 0 < network.dist_matrix[source][target] <= radius
    ]
    return _positive_median(values, "B_ref")


def _mu_quantiles(network, live, hop_counts, parameters: PureMRPParameters):
    values = []
    for source in live:
        for target in live:
            if source == target or network.dist_matrix[source][target] > parameters.communication_radius:
                continue
            try:
                values.append(calculate_mu(
                    source, target, network.initial_residual_e, network.dist_matrix,
                    network.base_dists, hop_counts, parameters.config,
                ))
            except HeuristicInputError:
                continue
    if not values:
        raise CharacterizationError("calibration snapshots yielded no valid Eq. (28) μ values")
    result = {f"q{int(probability * 100):02d}": _quantile(values, probability) for probability in (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)}
    if result["q05"] == result["q95"] or result["q10"] > result["q90"]:
        raise CharacterizationError("calibration μ distribution is degenerate; useful heuristic bounds cannot be formed")
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise CharacterizationError("calibration μ quantiles must be finite and non-negative")
    return result


def _hop_scale(scenarios):
    required = []
    for scenario in scenarios:
        missing = [node_id for node_id in scenario.event_nodes if node_id not in scenario.initial_hop_counts]
        if missing:
            raise CharacterizationError(f"calibration event nodes are unreachable from Sink: {missing}")
        required.extend(scenario.initial_hop_counts[node_id] for node_id in scenario.event_nodes)
    if not required or any(not isinstance(value, int) or value < 1 for value in required):
        raise CharacterizationError("calibration H requires positive finite event-to-Sink hop counts")
    return max(required)


def _reference_quality(network, config, scenarios, seeds, parameters):
    qualities = []
    for seed in seeds:
        rng = random.Random(derive_seed_streams(seed).pure_mrp_seed)
        for scenario in scenarios:
            state = PureMRPState(network.initial_residual_e, tuple(range(len(network.nodes))), frozenset())
            try:
                phase1 = select_cluster_head(
                    network.nodes, state.live_nodes, state.residual_e, scenario.event_nodes,
                    scenario.event_signal_strengths, network.dist_matrix, parameters.communication_radius,
                    parameters.config,
                )
                phase2 = discover_mrp_phase2_routes(
                    phase1.cluster_head, state.live_nodes, state.residual_e, network.dist_matrix,
                    network.base_dists, scenario.initial_hop_counts, parameters.communication_radius,
                    parameters.heuristic_bounds, parameters.config, parameters.lambda_coefficient,
                    parameters.ttl, 1, rng, parameters.c0, parameters.e_elec,
                    parameters.free_space_coeff, parameters.multipath_coeff, parameters.d0,
                    parameters.bit_count, parameters.c, parameters.c1,
                )
            except Exception:
                continue
            if phase2.successful_ant_results:
                quality = phase2.successful_ant_results[0].route_quality.quality
                if math.isfinite(quality) and quality > 0:
                    qualities.append(quality)
    return _positive_median(qualities, "f_ref"), len(qualities)


def _quantile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _positive_median(values, label):
    if not values:
        raise CharacterizationError(f"{label} requires at least one valid calibration observation")
    value = statistics.median(values)
    if not math.isfinite(value) or value <= 0:
        raise CharacterizationError(f"{label} must be finite and positive")
    return value
