"""Validity-first Pure-MRP parameter calibration; no Baseline/Hybrid outcome enters ranking."""

from collections.abc import Callable, Sequence
from dataclasses import replace
import math
import statistics
import json

from .calibration_audit import audit_calibration_parameters
from .calibration_config import candidate_configuration, enumerate_calibration_candidates, validate_calibration_plan
from .calibration_types import (
    CalibrationCandidate, CalibrationCandidateResult, CalibrationPlan, CalibrationResult,
    CalibrationRoundRecord, CalibrationSummary, CalibrationTrialResult,
)
from .config import ExperimentConfig, ExperimentMode, ExperimentNetwork
from .runners import run_trial


SELECTION_RULE = (
    "validity gates; completion rate descending; route-discovery failure rate ascending; "
    "multipath-ready rate descending; normalized own-energy ascending; config ID ascending"
)


def run_mrp_calibration(
    network: ExperimentNetwork, base_config: ExperimentConfig, plan: CalibrationPlan,
) -> CalibrationResult:
    """Evaluate caller-provided MRP candidates on calibration-only seeds/scenarios."""

    return run_mrp_calibration_candidates(
        network, base_config, plan, enumerate_calibration_candidates(plan),
    )


def run_mrp_calibration_candidates(
    network: ExperimentNetwork,
    base_config: ExperimentConfig,
    plan: CalibrationPlan,
    candidates: Sequence[CalibrationCandidate],
    *,
    repeat_scenarios: bool = False,
    scenario_provider: Callable[[object, object, ExperimentNetwork], object] | None = None,
    independent_scenario_trials: bool = False,
) -> CalibrationResult:
    """Run a caller-selected finite candidate list on Pure MRP only.

    The explicit-list variant supports transparent staged calibration without
    silently evaluating cross-products that were not approved as a stage.
    """

    return _run_mrp_calibration_candidate_list(
        network, base_config, plan, candidates, repeat_scenarios, scenario_provider,
        independent_scenario_trials, require_exact_approved_candidates=True,
    )


def _run_mrp_calibration_candidate_subset(
    network: ExperimentNetwork,
    base_config: ExperimentConfig,
    plan: CalibrationPlan,
    candidates: Sequence[CalibrationCandidate],
    *,
    repeat_scenarios: bool,
    scenario_provider: Callable[[object, object, ExperimentNetwork], object] | None,
    independent_scenario_trials: bool,
) -> CalibrationResult:
    """Evaluate an approved proper subset inside an isolated worker only."""

    return _run_mrp_calibration_candidate_list(
        network, base_config, plan, candidates, repeat_scenarios, scenario_provider,
        independent_scenario_trials, require_exact_approved_candidates=False,
    )


def _run_mrp_calibration_candidate_list(
    network, base_config, plan, candidates, repeat_scenarios, scenario_provider,
    independent_scenario_trials, *, require_exact_approved_candidates,
) -> CalibrationResult:
    validate_calibration_plan(plan)
    if not candidates or len({candidate.config_id for candidate in candidates}) != len(candidates):
        raise ValueError("calibration requires unique explicit candidates")
    _validate_candidates_against_plan(plan, candidates, require_exact_approved_candidates)
    audit = audit_calibration_parameters(base_config)
    results = tuple(
        _run_candidate(network, base_config, plan, candidate, repeat_scenarios, scenario_provider, independent_scenario_trials)
        for candidate in candidates
    )
    ranked = _rank(results)
    selected = next((item.candidate.config_id for item in ranked if item.rank == 1), None)
    return CalibrationResult(plan, audit, ranked, SELECTION_RULE, selected, _scenario_content_overlap(plan))


def _run_candidate(network, base_config, plan, candidate: CalibrationCandidate, repeat_scenarios, scenario_provider, independent_scenario_trials) -> CalibrationCandidateResult:
    trial_count = len(plan.calibration_seed_set) * (len(plan.calibration_scenarios) if independent_scenario_trials else 1)
    config = candidate_configuration(base_config, candidate, trial_count)
    rounds, trials = [], []
    trial_id = 0
    for seed in plan.calibration_seed_set:
        scenario_batches = ((scenario,) for scenario in plan.calibration_scenarios) if independent_scenario_trials else (plan.calibration_scenarios,)
        for scenarios in scenario_batches:
            comparison = run_trial(
                network, scenarios, config, trial_id, seed,
                repeat_scenarios=repeat_scenarios, scenario_provider=scenario_provider,
            )
            mode = comparison.mode_results[ExperimentMode.PURE_MRP]
            scenario_id = scenarios[0].scenario_id if len(scenarios) == 1 else None
            records = tuple(
                _round_record(candidate.config_id, seed, metric, raw, scenario_id)
                for metric, raw in zip(mode.rounds, mode.raw_round_results)
            )
            rounds.extend(records)
            trials.append(_trial_record(candidate.config_id, seed, records, scenario_id))
            trial_id += 1
    summary = _summary(candidate.config_id, tuple(rounds), tuple(trials), plan.minimum_completion_rate)
    return CalibrationCandidateResult(candidate, tuple(rounds), tuple(trials), summary, None)


def _round_record(config_id, seed, metric, raw, scenario_id=None) -> CalibrationRoundRecord:
    phase2 = raw.phase2_result
    status = None if phase2 is None else phase2.status
    unique_routes = None if phase2 is None else phase2.num_unique_routes
    ant_hops = None if phase2 is None else _mean([len(item.sant_result.path) - 1 for item in phase2.ant_results])
    minimum_tau = None if phase2 is None else _minimum_tau(phase2.final_pheromone_state)
    negative = status == "invalid_negative_pheromone_state" or (minimum_tau is not None and minimum_tau < 0)
    reason = metric.failure_reason
    topology = metric.failure_stage == "topology" or "ROUTE_TOPOLOGY_CONFLICT" in (reason or "")
    no_route = phase2 is not None and phase2.num_unique_routes == 0
    return CalibrationRoundRecord(
        config_id, seed, metric.round_index, metric.success, metric.failure_stage, reason, status,
        phase2 is not None and phase2.num_unique_routes > 0,
        None if phase2 is None else phase2.multipath_ready, unique_routes,
        None if phase2 is None else phase2.num_sants_requested,
        None if phase2 is None else phase2.num_sants_succeeded, ant_hops, negative,
        None if phase2 is None else phase2.halted_ant_index, _domain_failure(metric, negative, no_route),
        topology, no_route, minimum_tau, metric.actual_round_energy, metric.delivered_payload_count,
        metric.energy_per_delivered_payload, scenario_id,
    )


def _trial_record(config_id, seed, records, scenario_id=None) -> CalibrationTrialResult:
    completed = bool(records) and all(item.success for item in records)
    energies = [item.actual_round_energy for item in records if item.actual_round_energy is not None]
    payloads = [item.delivered_payload_count for item in records if item.delivered_payload_count is not None]
    minima = [item.minimum_observed_tau for item in records if item.minimum_observed_tau is not None]
    return CalibrationTrialResult(
        config_id, seed, completed, sum(item.phase2_status is not None for item in records),
        sum(item.route_discovery_success for item in records), sum(item.multipath_ready is True for item in records),
        sum(item.negative_pheromone_failure for item in records), sum(item.domain_failure for item in records),
        sum(item.topology_conflict for item in records),
        sum(item.no_route for item in records), min(minima) if minima else None,
        _ratio(sum(energies), sum(payloads)) if completed and len(energies) == len(records) and len(payloads) == len(records) else None,
        tuple(item.failure_reason for item in records if item.failure_reason), scenario_id,
    )


def _summary(config_id, records, trials, minimum_completion_rate) -> CalibrationSummary:
    attempts = sum(item.route_discovery_attempts for item in trials)
    successes = sum(item.route_discovery_successes for item in trials)
    multipath = sum(item.multipath_ready_count for item in trials)
    complete = sum(item.completed for item in trials)
    energies = [item.normalized_energy for item in trials if item.normalized_energy is not None]
    routes = [item.unique_route_count for item in records if item.unique_route_count is not None]
    minima = [item.minimum_observed_tau for item in records if item.minimum_observed_tau is not None]
    requested = [item.sants_requested for item in records if item.sants_requested is not None]
    succeeded = [item.successful_sants for item in records if item.successful_sants is not None]
    hops = [item.mean_ant_hops for item in records if item.mean_ant_hops is not None]
    negative = sum(item.negative_pheromone_failures for item in trials)
    domain = sum(item.domain_failure_count for item in trials)
    topology = sum(item.topology_conflicts for item in trials)
    no_route = sum(item.no_route_count for item in trials)
    reasons = _invalid_reasons(complete, len(trials), negative, domain, topology, minimum_completion_rate)
    return CalibrationSummary(
        config_id, len(trials), complete, len(trials) - complete, complete / len(trials),
        _ratio(successes, attempts), negative, _ratio(negative, attempts), domain, topology,
        _ratio(topology, len(records)), no_route, _ratio(no_route, attempts), _ratio(multipath, attempts), _mean(routes),
        _mean(energies), _stddev(energies), len(energies), min(minima) if minima else None,
        _mean(requested), _mean(succeeded), _mean(hops), not reasons, tuple(reasons),
    )


def _rank(results):
    valid = sorted((item for item in results if item.summary.valid), key=_rank_key)
    ranked = {item.candidate.config_id: index + 1 for index, item in enumerate(valid)}
    return tuple(replace(item, rank=ranked.get(item.candidate.config_id)) for item in results)


def _validate_candidates_against_plan(plan, candidates, require_exact_approved_candidates=True):
    """Ensure an explicit staged list cannot escape its declared provenance grid."""

    expected_paths = set(plan.search_space)
    for candidate in candidates:
        overrides = dict(candidate.overrides)
        if set(overrides) != expected_paths:
            raise ValueError("explicit candidate paths must exactly match the declared calibration search space")
        for path, value in overrides.items():
            if value not in plan.search_space[path]:
                raise ValueError(f"explicit candidate value is outside the approved search space: {path}")
    if plan.approved_candidate_overrides is not None:
        approved = {tuple(sorted(overrides)) for overrides in plan.approved_candidate_overrides}
        actual = {tuple(sorted(candidate.overrides)) for candidate in candidates}
        if (actual != approved) if require_exact_approved_candidates else not actual <= approved:
            raise ValueError("explicit candidates do not match the approved staged candidate list")


def _rank_key(item):
    summary = item.summary
    return (-summary.completion_rate, -_safe(summary.route_discovery_success_rate), -_safe(summary.multipath_ready_rate), _energy(summary.normalized_energy_mean), item.candidate.config_id)


def _invalid_reasons(completed, attempted, negative, domain, topology, threshold):
    reasons = []
    if not completed:
        reasons.append("zero_valid_trials")
    if negative:
        reasons.append("invalid_negative_pheromone_state")
    if domain:
        reasons.append("invalid_parameter_or_metric_domain")
    if topology:
        reasons.append("topology_conflict")
    if threshold is not None and completed / attempted < threshold:
        reasons.append("minimum_completion_rate_not_met")
    return reasons


def _minimum_tau(state):
    values = [tau for outgoing in state.values() for tau in outgoing.values()]
    return min(values) if values else None


def _domain_failure(metric, negative, no_route):
    return (
        not negative and not no_route and metric.failure_stage in {"phase2", "phase3"}
    )


def _scenario_content_overlap(plan):
    calibration = {_scenario_key(item) for item in plan.calibration_scenarios}
    evaluation = {_scenario_key(item) for item in plan.evaluation_scenarios}
    return bool(calibration & evaluation)


def _scenario_key(scenario):
    return json.dumps({
        "event_nodes": scenario.event_nodes,
        "event_signal_strengths": dict(scenario.event_signal_strengths),
        "hop_counts": dict(scenario.hop_counts),
    }, sort_keys=True, separators=(",", ":"))


def _ratio(numerator, denominator):
    return None if not denominator else numerator / denominator


def _mean(values):
    return None if not values else statistics.fmean(values)


def _stddev(values):
    return None if not values else statistics.stdev(values) if len(values) > 1 else 0.0


def _safe(value):
    return -1.0 if value is None else value


def _energy(value):
    return math.inf if value is None else value
