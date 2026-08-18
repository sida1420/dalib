"""Three-mode orchestration; algorithms remain delegated to their existing runners."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
import random

from ac_aco_mrp import (
    DirectMRPTopKParameters, HybridParameters, HybridRoundContext, HybridState,
    initialize_ac_aco_phase1_state, run_direct_mrp_topk_round,
    run_hybrid_lifetime_round, run_hybrid_round,
)
from mrp import (
    PureMRPRoundContext, PureMRPState, configured_target_cluster_head_count,
    run_pure_mrp_round,
)

from .baseline_adapter import BaselineState, run_baseline_round
from .config import EvaluationView, ExperimentConfig, ExperimentMode, ExperimentNetwork, ExperimentRoundScenario, StopCondition, TrialSeedStreams, derive_seed_streams
from .metrics import (
    aggregate_mode_trials, assess_trial_fairness, baseline_metrics, derive_lifetime_metrics,
    hybrid_metrics, network_wide_mrp_metrics, pure_mrp_metrics,
)
from .network_wide_mrp import run_network_wide_mrp_round
from .types import ComparisonTrialResult, ExperimentResult, ModeTrialResult


@dataclass
class _Runtime:
    nodes: Sequence[object]
    state: object
    rngs: tuple[object, ...]
    rows: list
    raw: list
    active: bool = True


def run_trial(
    network: ExperimentNetwork, scenarios: Sequence[ExperimentRoundScenario], config: ExperimentConfig,
    trial_id: int, trial_seed: int, *,
    repeat_scenarios: bool = False,
    scenario_provider: Callable[[ExperimentRoundScenario, object, ExperimentNetwork], ExperimentRoundScenario] | None = None,
) -> ComparisonTrialResult:
    """Run isolated mode states over caller-provided scenarios; failures remain raw records.

    ``scenario_provider`` is experiment infrastructure for refreshing external
    context such as hop counts.  It cannot alter the frozen event membership
    unless the caller explicitly supplies a different scenario object.
    """

    _validate_trial(network, scenarios, config)
    seeds = derive_seed_streams(trial_seed)
    runtimes = _initialize_runtimes(network, config, seeds)
    round_count = config.max_rounds if repeat_scenarios else min(config.max_rounds, len(scenarios))
    for round_index in range(round_count):
        scenario = scenarios[round_index % len(scenarios)]
        for mode, runtime in runtimes.items():
            if not runtime.active:
                continue
            current_scenario = scenario if scenario_provider is None else scenario_provider(scenario, runtime.state, network)
            if not isinstance(current_scenario, ExperimentRoundScenario):
                raise ValueError("scenario provider must return an ExperimentRoundScenario")
            if (
                current_scenario.event_nodes != scenario.event_nodes
                or dict(current_scenario.event_signal_strengths) != dict(scenario.event_signal_strengths)
            ):
                raise ValueError("scenario provider may refresh hop counts only; frozen event inputs cannot change")
            result, row = _run_mode(mode, runtime, network, current_scenario, config, trial_id, seeds, round_index)
            runtime.raw.append(result)
            runtime.rows.append(row)
            if not result.success or _should_stop(runtime.rows, len(network.nodes), config.stop_condition):
                runtime.active = False
    mode_results = {
        mode: ModeTrialResult(mode, tuple(runtime.rows), tuple(runtime.raw), derive_lifetime_metrics(runtime.rows, len(network.nodes)))
        for mode, runtime in runtimes.items()
    }
    return ComparisonTrialResult(trial_id, seeds, mode_results, assess_trial_fairness(mode_results, config.evaluation_view))


def run_experiment(
    network: ExperimentNetwork, scenarios: Sequence[ExperimentRoundScenario], config: ExperimentConfig,
    trial_seeds: Sequence[int],
) -> ExperimentResult:
    """Run fresh isolated trials and retain raw outcomes before aggregation."""

    if len(trial_seeds) != config.trial_count:
        raise ValueError("trial_seeds length must equal ExperimentConfig.trial_count")
    trials = tuple(run_trial(network, scenarios, config, index, seed) for index, seed in enumerate(trial_seeds))
    aggregate = {
        mode.value: aggregate_mode_trials(
            [trial.mode_results[mode] for trial in trials if mode in trial.mode_results], config.evaluation_view,
        )
        for mode in config.modes
    }
    return ExperimentResult(trials, aggregate)


def smoke_trace(trial: ComparisonTrialResult) -> str:
    """Describe a tiny validation run without naming a numerical winner."""

    lines = [f"TRIAL: {trial.trial_id}; seed={trial.seeds.trial_seed}; view={trial.fairness.reporting_view.value}"]
    for mode, result in trial.mode_results.items():
        row = result.rounds[0] if result.rounds else None
        if row is None:
            lines.append(f"{mode.value}: no round record")
            continue
        lines.append(
            f"{mode.value}: sources={row.declared_source_sensors}; CHs={row.selected_cluster_heads}; "
            f"routes={row.selected_routes}; physical_energy={row.actual_round_energy}; "
            f"residual_total={row.residual_total}; success={row.success}"
        )
    lines.append(f"FAIRNESS: raw-energy {trial.fairness.raw_energy_status}; {trial.fairness.reason}")
    return "\n".join(lines)


def _initialize_runtimes(network, config, seeds):
    runtimes = {}
    if ExperimentMode.BASELINE in config.modes:
        rng = random.Random(seeds.baseline_ac_aco_seed)
        runtimes[ExperimentMode.BASELINE] = _Runtime(
            deepcopy(network.nodes), BaselineState(_physical_state(network), initialize_ac_aco_phase1_state(len(network.nodes), config.ac_aco_parameters, rng)), (rng,), [], [],
        )
    if ExperimentMode.PURE_MRP in config.modes:
        runtimes[ExperimentMode.PURE_MRP] = _Runtime(deepcopy(network.nodes), _physical_state(network), (random.Random(seeds.pure_mrp_seed),), [], [])
    if ExperimentMode.MRP_NETWORK_WIDE in config.modes:
        runtimes[ExperimentMode.MRP_NETWORK_WIDE] = _Runtime(
            deepcopy(network.nodes), _physical_state(network),
            (random.Random(seeds.pure_mrp_seed),), [], [],
        )
    if ExperimentMode.HYBRID in config.modes:
        ac_rng, mrp_rng = random.Random(seeds.hybrid_ac_aco_seed), random.Random(seeds.hybrid_mrp_seed)
        runtimes[ExperimentMode.HYBRID] = _Runtime(
            deepcopy(network.nodes), HybridState(_physical_state(network), initialize_ac_aco_phase1_state(len(network.nodes), config.ac_aco_parameters, ac_rng)), (ac_rng, mrp_rng), [], [],
        )
    return runtimes


def _run_mode(
    mode, runtime, network, scenario, config, trial_id, seeds, round_index, *,
    hybrid_phase3_top_k=None,
):
    params = config.mrp_parameters
    if mode is ExperimentMode.BASELINE:
        result = run_baseline_round(runtime.nodes, network.dist_matrix, network.base_dists, runtime.state, config.ac_aco_parameters, params, runtime.rngs[0], round_index)
        if result.success:
            runtime.state = result.state_after
        return result, baseline_metrics(result, trial_id, seeds.trial_seed, params.bit_count)
    if mode is ExperimentMode.PURE_MRP:
        context = PureMRPRoundContext(scenario.event_nodes, scenario.event_signal_strengths, scenario.hop_counts)
        result = run_pure_mrp_round(runtime.nodes, network.dist_matrix, network.base_dists, runtime.state, context, params, runtime.rngs[0], config.mrp_pheromone_lifecycle, round_index)
        runtime.state = result.state_after
        return result, pure_mrp_metrics(result, trial_id, seeds.trial_seed, scenario.event_nodes, params.bit_count)
    if mode is ExperimentMode.MRP_NETWORK_WIDE:
        target = configured_target_cluster_head_count(
            len(network.nodes), config.ac_aco_parameters.ch_proportion,
        )
        result = run_network_wide_mrp_round(
            runtime.nodes, network.dist_matrix, network.base_dists, runtime.state,
            scenario.hop_counts, params, runtime.rngs[0],
            config.mrp_pheromone_lifecycle, target, round_index,
        )
        runtime.state = result.state_after
        return result, network_wide_mrp_metrics(
            result, trial_id, seeds.trial_seed, params.bit_count,
        )
    context = HybridRoundContext(scenario.hop_counts)
    result = run_hybrid_round(
        runtime.nodes, network.dist_matrix, network.base_dists, runtime.state, context,
        HybridParameters(config.ac_aco_parameters, params, hybrid_phase3_top_k),
        runtime.rngs[0], runtime.rngs[1], config.mrp_pheromone_lifecycle, round_index,
    )
    runtime.state = result.state_after
    return result, hybrid_metrics(result, trial_id, seeds.trial_seed, params.bit_count)


def _run_hybrid_lifetime_mode(
    mode, runtime, network, scenario, config, trial_id, seeds, round_index,
):
    """Manual-only experimental dispatch; frozen ExperimentMode remains HYBRID."""

    if mode is not ExperimentMode.HYBRID:
        raise ValueError("lifetime-aware Hybrid dispatch requires the Hybrid runtime")
    params = config.mrp_parameters
    result = run_hybrid_lifetime_round(
        runtime.nodes, network.dist_matrix, network.base_dists, runtime.state,
        HybridRoundContext(scenario.hop_counts),
        HybridParameters(config.ac_aco_parameters, params), runtime.rngs[0],
        runtime.rngs[1], config.mrp_pheromone_lifecycle, round_index,
    )
    runtime.state = result.state_after
    return result, hybrid_metrics(result, trial_id, seeds.trial_seed, params.bit_count)


def _run_direct_mrp_topk_mode(
    mode, runtime, network, scenario, config, trial_id, seeds, round_index, *,
    phase3_top_k,
):
    """Manual-only dispatch for the Direct-first project adaptation."""

    if mode is not ExperimentMode.HYBRID:
        raise ValueError("Direct-first MRP dispatch requires the Hybrid runtime")
    params = config.mrp_parameters
    result = run_direct_mrp_topk_round(
        runtime.nodes,
        network.dist_matrix,
        network.base_dists,
        runtime.state,
        HybridRoundContext(scenario.hop_counts),
        DirectMRPTopKParameters(
            config.ac_aco_parameters, params, phase3_top_k,
        ),
        runtime.rngs[0],
        runtime.rngs[1],
        config.mrp_pheromone_lifecycle,
        round_index,
    )
    runtime.state = result.state_after
    return result, hybrid_metrics(
        result, trial_id, seeds.trial_seed, params.bit_count,
    )


def _physical_state(network):
    residual = tuple(network.initial_residual_e)
    live = tuple(index for index, energy in enumerate(residual) if energy > 0)
    return PureMRPState(residual, live, frozenset(set(range(len(residual))) - set(live)))


def _should_stop(rows, node_count, condition):
    if condition is StopCondition.MAX_ROUNDS:
        return False
    lifetime = derive_lifetime_metrics(rows, node_count)
    if condition is StopCondition.FND:
        return lifetime.fnd_round is not None
    if condition is StopCondition.HND:
        return lifetime.hnd_round is not None
    if condition is StopCondition.LND:
        return lifetime.lnd_round is not None
    return not (set(rows[-1].declared_source_sensors or ()) & set(rows[-1].live_after))


def _validate_trial(network, scenarios, config):
    if not isinstance(config, ExperimentConfig) or not _positive_int(config.max_rounds) or not _positive_int(config.trial_count):
        raise ValueError("ExperimentConfig requires positive trial and round counts")
    if not config.modes or any(not isinstance(mode, ExperimentMode) for mode in config.modes) or len(set(config.modes)) != len(config.modes):
        raise ValueError("ExperimentConfig.modes must contain unique ExperimentMode values")
    if not isinstance(config.evaluation_view, EvaluationView):
        raise ValueError("ExperimentConfig.evaluation_view must be an EvaluationView")
    if not isinstance(config.stop_condition, StopCondition):
        raise ValueError("ExperimentConfig.stop_condition must be a StopCondition")
    if not network.nodes or len(network.initial_residual_e) != len(network.nodes):
        raise ValueError("network initial residual energy must match a non-empty node list")
    if len(network.dist_matrix) != len(network.nodes) or len(network.base_dists) != len(network.nodes):
        raise ValueError("network distance snapshots must match node count")
    if any(not _finite_nonnegative(value) for value in network.initial_residual_e + tuple(network.base_dists)):
        raise ValueError("network residual energies and Sink distances must be finite and non-negative")
    if any(len(row) != len(network.nodes) or any(not _finite_nonnegative(value) for value in row) for row in network.dist_matrix):
        raise ValueError("network distance matrix must be finite, non-negative, and square")
    if not scenarios:
        raise ValueError("experiments require caller-provided event scenarios")
    if any(not isinstance(item, ExperimentRoundScenario) for item in scenarios):
        raise ValueError("scenarios must contain ExperimentRoundScenario values")


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
