"""One isolated network-wide TTL calibration trial for Pure or Hybrid MRP."""

from dataclasses import replace
import time

from .calibration_freeze import load_frozen_configuration
from .config import EvaluationView, ExperimentMode, StopCondition, derive_seed_streams
from .hop_counts import scenario_with_current_hops
from .process_metrics import process_memory_bytes
from .real_calibration import primary_calibration_config
from .runners import _initialize_runtimes, _run_mode
from .scenario_generation import load_existing_map_network, load_frozen_scenario_sets
from mrp.runner_types import PheromoneLifecycle


MODE_MAP = {
    "mrp_pure": ExperimentMode.MRP_NETWORK_WIDE,
    "mrp_hybrid": ExperimentMode.HYBRID,
}


def run_ttl_trial(root, *, mode, ttl, seed, scenario_id, round_budget=10):
    """Run one bounded trial and retain only compact aggregate diagnostics."""

    started_wall, started_cpu = time.perf_counter(), time.process_time()
    existing = load_existing_map_network(root / "map.pkl")
    frozen = load_frozen_configuration(
        root / "real_calibration_v2" / "calibration_outputs" / "selected_config.json",
        primary_calibration_config(existing),
    )
    scenarios = load_frozen_scenario_sets(root / "real_calibration_v2" / "scenario_manifest.json")
    scenario = next(item for item in scenarios.calibration if item.scenario_id == scenario_id)
    experiment_mode = MODE_MAP[mode]
    config = replace(
        frozen,
        max_rounds=round_budget,
        trial_count=1,
        modes=(experiment_mode,),
        evaluation_view=EvaluationView.NATIVE_ALGORITHM_SEMANTICS,
        stop_condition=StopCondition.MAX_ROUNDS,
        mrp_parameters=replace(frozen.mrp_parameters, ttl=ttl),
    )
    _validate_ttl_only_configuration(config, ttl)
    seeds = derive_seed_streams(seed)
    runtime = _initialize_runtimes(existing.network, config, seeds)[experiment_mode]
    totals = _empty_totals(
        mode, ttl, config.mrp_parameters.num_sants, seed, scenario_id,
        scenario.event_anchor_sensor, round_budget,
    )
    for round_index in range(round_budget):
        physical = getattr(runtime.state, "physical_state", runtime.state)
        current_scenario = scenario_with_current_hops(
            scenario.round_scenario(), existing.network, physical.live_nodes,
            config.mrp_parameters.communication_radius,
        )[0]
        raw, row = _run_mode(
            experiment_mode, runtime, existing.network, current_scenario,
            config, 0, seeds, round_index,
        )
        _record_round(totals, raw, row, existing.network.base_dists, config.mrp_parameters)
        if not raw.success:
            break
        del raw, row
    totals["completed_horizon"] = totals["successful_complete_rounds"] == round_budget
    totals["status"] = totals["terminal_status"] or "COMPLETED_CALIBRATION_HORIZON"
    totals["cpu_seconds"] = time.process_time() - started_cpu
    totals["wall_seconds"] = time.perf_counter() - started_wall
    totals["current_rss_bytes"], totals["peak_rss_bytes"] = process_memory_bytes()
    return totals


def _empty_totals(mode, ttl, num_sants, seed, scenario_id, anchor, round_budget):
    return {
        "mode": mode, "ttl": ttl, "num_sants": num_sants, "seed": seed,
        "scenario_id": scenario_id, "anchor": anchor, "round_budget": round_budget,
        "attempted_complete_rounds": 0, "successful_complete_rounds": 0,
        "mrp_search_failures": 0, "ch_discoveries_attempted": 0,
        "ch_routes_succeeded": 0, "sants_executed": 0, "sants_succeeded": 0,
        "unique_route_total": 0, "selected_route_count": 0,
        "selected_route_hop_total": 0, "selected_route_hop_max": None,
        "ttl_exhausted_sants": 0, "sink_neighbor_at_ttl_zero": 0,
        "max_first_hop_probability": None, "physical_energy": 0.0,
        "delivered_payloads": 0, "ch99_discoveries": 0,
        "ch99_sants_executed": 0, "ch99_sants_succeeded": 0,
        "ch99_ttl_exhausted_sants": 0, "ch99_sink_neighbor_at_ttl_zero": 0,
        "terminal_status": None, "terminal_reason": None, "failed_ch": None,
    }


def _validate_ttl_only_configuration(config, ttl):
    parameters = config.mrp_parameters
    expected = {
        "ttl": ttl, "num_sants": 20, "lambda_coefficient": 9.04421850555005,
        "c0": 1.0, "c": 0.005, "c1": 0.003696666640752299,
    }
    actual = {name: getattr(parameters, name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"TTL-only parameter guard failed: {actual!r}")
    paper_expected = {
        "alpha": 2.0, "beta": 2.0, "rho": 0.2,
        "initial_pheromone": 0.01, "aant_probability": 0.001,
    }
    paper_actual = {name: getattr(parameters.config, name) for name in paper_expected}
    if paper_actual != paper_expected:
        raise RuntimeError(f"fixed MRP configuration guard failed: {paper_actual!r}")
    bounds = parameters.heuristic_bounds
    if (bounds.mu_min, bounds.mu_max, bounds.eta_min, bounds.eta_max) != (
        0.00770435077647942, 0.019383167511744133,
        0.00770435077647942, 0.019383167511744133,
    ):
        raise RuntimeError("fixed heuristic clipping guard failed")
    if config.mrp_pheromone_lifecycle is not PheromoneLifecycle.RESET_PER_DISCOVERY:
        raise RuntimeError("fixed pheromone lifecycle guard failed")


def _record_round(totals, raw, row, base_dists, parameters):
    totals["attempted_complete_rounds"] += 1
    completed = tuple(raw.ch_routing_results or ())
    phase2_results = [item.phase2_result for item in completed]
    failed_phase2 = getattr(raw, "failed_phase2_result", None)
    if failed_phase2 is not None:
        phase2_results.append(failed_phase2)
    totals["ch_discoveries_attempted"] += len(completed) + (failed_phase2 is not None)
    totals["ch_routes_succeeded"] += len(completed)
    for phase2 in phase2_results:
        _record_phase2(totals, phase2, base_dists, parameters)
    for item in completed:
        hops = len(item.phase3_result.selected_route) - 1
        totals["selected_route_count"] += 1
        totals["selected_route_hop_total"] += hops
        current_max = totals["selected_route_hop_max"]
        totals["selected_route_hop_max"] = hops if current_max is None else max(current_max, hops)
    if raw.success:
        totals["successful_complete_rounds"] += 1
        totals["physical_energy"] += row.actual_round_energy
        totals["delivered_payloads"] += row.delivered_payload_count
        return
    classification = getattr(raw, "routing_failure_classification", None)
    totals["terminal_status"] = classification or raw.failure_stage
    totals["terminal_reason"] = raw.failure_reason
    totals["failed_ch"] = getattr(raw, "failed_cluster_head", None)
    totals["mrp_search_failures"] += classification == "MRP_SEARCH_FAILURE"


def _record_phase2(totals, phase2, base_dists, parameters):
    is_ch99 = phase2.start_cluster_head == 99
    totals["unique_route_total"] += phase2.num_unique_routes
    totals["sants_executed"] += phase2.num_sants_executed
    totals["sants_succeeded"] += phase2.num_sants_succeeded
    if is_ch99:
        totals["ch99_discoveries"] += 1
        totals["ch99_sants_executed"] += phase2.num_sants_executed
        totals["ch99_sants_succeeded"] += phase2.num_sants_succeeded
    for ant in phase2.ant_results:
        sant = ant.sant_result
        exhausted = sant.failure_reason == "ttl_exhausted"
        sink_neighbor = (
            exhausted and sant.remaining_ttl == 0
            and base_dists[sant.visited_nodes[-1]] <= parameters.communication_radius
        )
        totals["ttl_exhausted_sants"] += exhausted
        totals["sink_neighbor_at_ttl_zero"] += sink_neighbor
        if is_ch99:
            totals["ch99_ttl_exhausted_sants"] += exhausted
            totals["ch99_sink_neighbor_at_ttl_zero"] += sink_neighbor
        for hop in sant.trace:
            if hop.probabilities:
                maximum = max(hop.probabilities.values())
                current = totals["max_first_hop_probability"]
                totals["max_first_hop_probability"] = maximum if current is None else max(current, maximum)
                break
