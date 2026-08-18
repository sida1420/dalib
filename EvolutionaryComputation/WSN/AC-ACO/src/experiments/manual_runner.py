"""One-mode manual route-exhaustion diagnostics.

This module deliberately composes the existing experiment runners rather than
implementing routing or energy logic.  It is separate from frozen final
evaluation: it rejects reserved evaluation seeds and never writes final
evaluation artifacts.
"""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
import gc
import os
from pathlib import Path
import time

from .calibration_freeze import load_frozen_configuration
from .config import EvaluationView, ExperimentMode, StopCondition, derive_seed_streams
from .hop_counts import scenario_with_current_hops
from .native_case import (
    NATIVE_CASE_ID, NATIVE_CASE_SOURCE, build_native_round_context, native_case_identity,
)
from .process_metrics import process_memory_bytes
from .real_calibration import EVALUATION_SEEDS, primary_calibration_config
from .runners import (
    _initialize_runtimes,
    _run_direct_mrp_topk_mode,
    _run_hybrid_lifetime_mode,
    _run_mode,
)
from .scenario_generation import load_existing_map_network, load_frozen_scenario_sets


DEFAULT_SEED = 33001
DEFAULT_SCENARIO_ID = "calibration-v1-c1"
DEFAULT_ANCHOR = 49
DEFAULT_SAFETY_MAX_ROUNDS = 100000
DEFAULT_MRP_SEARCH_RETRIES = 3

_MODE_ALIASES = {
    "direct": "direct",
    "mrp pure": "mrp_pure",
    "pure": "mrp_pure",
    "mrp event": "mrp_event",
    "event": "mrp_event",
    "paper": "mrp_event",
    "mrp hybrid": "mrp_hybrid",
    "hybrid": "mrp_hybrid",
    "mrp hybrid lifetime": "mrp_hybrid_lifetime",
    "hybrid lifetime": "mrp_hybrid_lifetime",
    "lifetime": "mrp_hybrid_lifetime",
    "direct mrp topk": "direct_mrp_topk",
    "direct mrp top k": "direct_mrp_topk",
    "ac aco direct mrp topk": "direct_mrp_topk",
}
_EXPERIMENT_MODES = {
    "direct": ExperimentMode.BASELINE,
    "mrp_pure": ExperimentMode.MRP_NETWORK_WIDE,
    "mrp_event": ExperimentMode.PURE_MRP,
    "mrp_hybrid": ExperimentMode.HYBRID,
    "mrp_hybrid_lifetime": ExperimentMode.HYBRID,
    "direct_mrp_topk": ExperimentMode.HYBRID,
}
_EXHAUSTED_STATUSES = {
    "direct": "DIRECT_ROUTE_EXHAUSTED",
    "mrp_event": "MRP_EVENT_ROUTE_EXHAUSTED",
    "mrp_hybrid": "MRP_HYBRID_ROUTE_EXHAUSTED",
    "direct_mrp_topk": "DIRECT_MRP_TOPK_ROUTE_EXHAUSTED",
}


class ManualRunnerError(ValueError):
    """Raised for invalid manual-diagnostic input or runner invariants."""


def normalize_manual_mode(value: str) -> str:
    """Normalize a documented CLI value to one canonical one-mode identifier."""

    if not isinstance(value, str):
        raise ManualRunnerError("Invalid mode. Valid modes: direct, mrp_pure, mrp_hybrid, direct_mrp_topk, mrp_hybrid_lifetime, mrp_event")
    normalized = " ".join(value.strip().casefold().replace("_", " ").replace("-", " ").split())
    try:
        return _MODE_ALIASES[normalized]
    except KeyError as error:
        raise ManualRunnerError("Invalid mode. Valid modes: direct, mrp_pure, mrp_hybrid, direct_mrp_topk, mrp_hybrid_lifetime, mrp_event") from error


def run_manual_route_exhaustion(
    repository_root: str | Path,
    *,
    mode: str,
    seed: int = DEFAULT_SEED,
    scenario_id: str = DEFAULT_SCENARIO_ID,
    anchor: int | None = DEFAULT_ANCHOR,
    safety_max_rounds: int = DEFAULT_SAFETY_MAX_ROUNDS,
    native_case: bool = False,
    ttl_override: int | None = None,
    phase3_top_k: int | None = None,
    charge_ant_energy: bool = False,
    ant_control_packet_bits: int | None = None,
    mrp_search_retries: int = DEFAULT_MRP_SEARCH_RETRIES,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run exactly one current mode until service routing fails or a safety guard fires."""

    canonical_mode = normalize_manual_mode(mode)
    _validate_request(seed, anchor, safety_max_rounds, native_case, mrp_search_retries)
    if ttl_override is not None and (
        isinstance(ttl_override, bool) or not isinstance(ttl_override, int) or ttl_override < 0
    ):
        raise ManualRunnerError("ttl_override must be a non-negative integer")
    if ttl_override is not None and canonical_mode == "direct":
        raise ManualRunnerError("--ttl applies only to MRP-routing modes")
    if canonical_mode == "direct_mrp_topk" and ttl_override is not None and ttl_override <= 0:
        raise ManualRunnerError("direct_mrp_topk requires --ttl greater than zero")
    if phase3_top_k is not None and (
        not isinstance(phase3_top_k, int) or isinstance(phase3_top_k, bool)
        or phase3_top_k < 0
    ):
        raise ManualRunnerError("phase3_top_k must be None or a non-negative integer")
    if phase3_top_k is not None and canonical_mode not in {"mrp_hybrid", "direct_mrp_topk"}:
        raise ManualRunnerError("--phase3-top-k applies only to mrp_hybrid or direct_mrp_topk")
    if canonical_mode == "direct_mrp_topk" and (
        phase3_top_k is None or phase3_top_k <= 0
    ):
        raise ManualRunnerError("direct_mrp_topk requires --phase3-top-k greater than zero")
    if not isinstance(charge_ant_energy, bool):
        raise ManualRunnerError("charge_ant_energy must be a boolean")
    if ant_control_packet_bits is not None and (
        not isinstance(ant_control_packet_bits, int)
        or isinstance(ant_control_packet_bits, bool)
        or ant_control_packet_bits <= 0
    ):
        raise ManualRunnerError("ant_control_packet_bits must be a positive integer")
    if charge_ant_energy and ant_control_packet_bits is None:
        raise ManualRunnerError(
            "--ant-control-packet-bits is required with --charge-ant-energy"
        )
    if ant_control_packet_bits is not None and not charge_ant_energy:
        raise ManualRunnerError(
            "--ant-control-packet-bits requires --charge-ant-energy"
        )
    if charge_ant_energy and canonical_mode == "mrp_hybrid_lifetime":
        raise ManualRunnerError(
            "ant energy is not supported by the experimental lifetime selector"
        )
    if native_case and canonical_mode == "mrp_event":
        raise ManualRunnerError("mrp_event requires an explicit event scenario; native case is network-wide only")
    root = Path(repository_root).resolve()
    existing_map = load_existing_map_network(root / "map.pkl")
    frozen_config = load_frozen_configuration(
        root / "real_calibration_v2" / "calibration_outputs" / "selected_config.json",
        primary_calibration_config(existing_map),
    )
    if native_case:
        round_scenario = build_native_round_context(
            existing_map, frozen_config.mrp_parameters.communication_radius,
        )
        scenario_id, anchor = NATIVE_CASE_ID, None
    else:
        scenarios = load_frozen_scenario_sets(root / "real_calibration_v2" / "scenario_manifest.json")
        frozen_scenario = _select_calibration_scenario(scenarios.calibration, scenario_id, anchor)
        round_scenario = frozen_scenario.round_scenario()
        scenario_id, anchor = frozen_scenario.scenario_id, frozen_scenario.event_anchor_sensor
    experiment_mode = _EXPERIMENT_MODES[canonical_mode]
    mrp_parameters = (
        frozen_config.mrp_parameters if ttl_override is None
        else replace(frozen_config.mrp_parameters, ttl=ttl_override)
    )
    mrp_parameters = replace(
        mrp_parameters,
        charge_ant_energy=charge_ant_energy,
        ant_control_packet_bits=ant_control_packet_bits,
    )
    if canonical_mode == "direct_mrp_topk" and mrp_parameters.ttl <= 0:
        raise ManualRunnerError("direct_mrp_topk requires an effective TTL greater than zero")
    config = replace(
        frozen_config,
        max_rounds=safety_max_rounds,
        trial_count=1,
        modes=(experiment_mode,),
        evaluation_view=EvaluationView.NATIVE_ALGORITHM_SEMANTICS,
        stop_condition=StopCondition.MAX_ROUNDS,
        mrp_parameters=mrp_parameters,
    )
    seeds = derive_seed_streams(seed)
    runtime = _initialize_runtimes(existing_map.network, config, seeds)[experiment_mode]
    return execute_route_exhaustion(
        canonical_mode=canonical_mode,
        experiment_mode=experiment_mode,
        network=existing_map.network,
        scenario=round_scenario,
        runtime=runtime,
        config=config,
        seeds=seeds,
        seed=seed,
        scenario_id=scenario_id,
        anchor=anchor,
        safety_max_rounds=safety_max_rounds,
        mrp_search_retries=mrp_search_retries,
        phase3_top_k=phase3_top_k,
        charge_ant_energy=charge_ant_energy,
        ant_control_packet_bits=ant_control_packet_bits,
        metadata=_native_metadata(existing_map, frozen_config) if native_case else None,
        progress=progress,
        round_executor=(
            _run_hybrid_lifetime_mode
            if canonical_mode == "mrp_hybrid_lifetime"
            else partial(_run_direct_mrp_topk_mode, phase3_top_k=phase3_top_k)
            if canonical_mode == "direct_mrp_topk"
            else partial(_run_mode, hybrid_phase3_top_k=phase3_top_k)
            if canonical_mode == "mrp_hybrid"
            else _run_mode
        ),
    )


def execute_route_exhaustion(
    *,
    canonical_mode: str,
    experiment_mode: ExperimentMode,
    network,
    scenario,
    runtime,
    config,
    seeds,
    seed: int,
    scenario_id: str,
    anchor: int | None,
    safety_max_rounds: int,
    mrp_search_retries: int = 0,
    phase3_top_k: int | None = None,
    charge_ant_energy: bool = False,
    ant_control_packet_bits: int | None = None,
    metadata: dict[str, object] | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
    round_executor: Callable = _run_mode,
) -> dict[str, object]:
    """Execute one already-initialized runtime without retaining round histories.

    The injectable executor keeps tests dispatch-only and proves that FND is a
    recorded milestone, rather than a stop policy, without running a simulator.
    """

    if canonical_mode not in _EXPERIMENT_MODES or _EXPERIMENT_MODES[canonical_mode] is not experiment_mode:
        raise ManualRunnerError("manual mode and experiment mode do not match")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    summary = _new_summary(canonical_mode, seed, scenario_id, anchor, safety_max_rounds)
    summary["ttl"] = (
        getattr(config.mrp_parameters, "ttl", None)
        if canonical_mode != "direct" else None
    )
    summary.update({
        "charge_ant_energy": charge_ant_energy,
        "ant_control_packet_bits": ant_control_packet_bits,
        "ant_tx_rx_count_scope": "sensor_battery_radio_operations",
        "sant_tx_count": 0,
        "sant_rx_count": 0,
        "bant_tx_count": 0,
        "bant_rx_count": 0,
        "aant_tx_count": 0,
        "aant_rx_count": 0,
        "sant_energy": 0.0,
        "bant_energy": 0.0,
        "aant_energy": 0.0,
        "total_ant_control_energy": 0.0,
        "data_energy": 0.0,
        "total_physical_energy": 0.0,
    })
    if canonical_mode == "mrp_hybrid":
        summary.update({
            "configured_top_k": phase3_top_k,
            "discovered_routes_before_pruning": 0,
            "candidate_routes_after_pruning": 0,
            "pruned_routes_count": 0,
            "mean_routes_before_pruning": None,
            "mean_routes_after_pruning": None,
        })
    if canonical_mode == "direct_mrp_topk":
        summary.update({
            "configured_top_k": phase3_top_k,
            "direct_ch_count": 0,
            "fallback_ch_count": 0,
            "direct_transmissions": 0,
            "fallback_transmissions": 0,
            "direct_successes": 0,
            "fallback_successes": 0,
            "fallback_failures": 0,
            "direct_ratio": None,
            "fallback_ratio": None,
            "mrp_discoveries": 0,
            "cached_route_uses": 0,
            "routes_before_topk": 0,
            "routes_after_topk": 0,
            "routes_pruned": 0,
            "mean_routes_pruned": None,
            "sant_count": 0,
            "bant_count": 0,
            "aant_count": 0,
            "routing_failures": 0,
            "dropped_packets": 0,
        })
    if metadata is not None:
        summary.update(metadata)
    for round_index in range(safety_max_rounds):
        phase1_rng_state = _capture_phase1_rng(experiment_mode, runtime)
        search_attempt = 0
        while True:
            current_scenario = _current_scenario(
                experiment_mode, scenario, runtime, network, config,
            )
            raw, row = round_executor(
                experiment_mode, runtime, network, current_scenario, config, 0, seeds, round_index,
            )
            if canonical_mode in {"mrp_pure", "mrp_hybrid", "mrp_hybrid_lifetime", "direct_mrp_topk"}:
                _record_mrp_search_diagnostics(summary, raw, network.base_dists, config.mrp_parameters)
            if canonical_mode == "direct_mrp_topk":
                _record_direct_mrp_diagnostics(summary, raw)
            _record_ant_energy_diagnostics(summary, raw, row)
            if not raw.success and _retryable_mrp_search_failure(raw):
                summary["mrp_search_failures"] += 1
            if raw.success or not _retryable_mrp_search_failure(raw) or search_attempt >= mrp_search_retries:
                break
            search_attempt += 1
            summary["mrp_search_retries"] += 1
            _restore_phase1_rng(runtime, phase1_rng_state)
            del raw, row
        if raw.success:
            _validate_successful_mode(canonical_mode, raw)
            _record_success(summary, row, len(network.nodes), _success_details(canonical_mode, raw))
            if progress is not None:
                progress(_progress_record(summary, row))
            del raw, row
            if round_index % 25 == 0:
                gc.collect()
            continue
        _record_terminal_state(summary, row)
        if summary["total_ant_control_energy"] > 0:
            summary["physical_energy"] = row.cumulative_actual_energy
        summary["routing_failure_round_zero_based"] = round_index
        summary["routing_failure_round_one_based"] = round_index + 1
        summary["failure_stage"] = raw.failure_stage
        summary["failure_reason"] = raw.failure_reason
        summary["failure_details"] = _failure_details(canonical_mode, raw)
        summary["status"] = _terminal_status(canonical_mode, raw)
        if canonical_mode == "direct_mrp_topk":
            diagnostics = raw.direct_mrp_diagnostics
            if raw.failed_cluster_head in set(diagnostics.fallback_cluster_heads):
                summary["fallback_failures"] += 1
                summary["routing_failures"] += 1
                summary["dropped_packets"] += 1
        del raw, row
        break
    else:
        summary["status"] = "SAFETY_CAP_REACHED"
    summary["end_time"] = _now()
    summary["wall_seconds"] = time.perf_counter() - started_wall
    summary["cpu_seconds"] = time.process_time() - started_cpu
    summary["current_rss_bytes"], summary["peak_rss_bytes"] = process_memory_bytes()
    summary["energy_per_delivered_payload"] = (
        summary["physical_energy"] / summary["delivered_payloads"]
        if summary["delivered_payloads"] else None
    )
    summary["total_physical_energy"] = summary["physical_energy"]
    summary["sant_success_rate"] = _ratio(summary["sants_succeeded"], summary["sants_executed"])
    summary["mean_unique_routes"] = _ratio(
        summary["unique_route_total"], summary["mrp_ch_discoveries_attempted"]
    )
    summary["mean_selected_route_hops"] = _ratio(
        summary["selected_route_hop_total"], summary["selected_route_count"]
    )
    if "discovered_routes_before_pruning" in summary:
        summary["mean_routes_before_pruning"] = _ratio(
            summary["discovered_routes_before_pruning"], summary["selected_route_count"]
        )
        summary["mean_routes_after_pruning"] = _ratio(
            summary["candidate_routes_after_pruning"], summary["selected_route_count"]
        )
    if canonical_mode == "direct_mrp_topk":
        transmissions = summary["direct_transmissions"] + summary["fallback_transmissions"]
        summary["direct_ratio"] = _ratio(summary["direct_transmissions"], transmissions)
        summary["fallback_ratio"] = _ratio(summary["fallback_transmissions"], transmissions)
        summary["mean_routes_pruned"] = _ratio(
            summary["routes_pruned"], summary["mrp_discoveries"]
        )
    return summary


def _validate_request(
    seed: int, anchor: int | None, safety_max_rounds: int, native_case: bool,
    mrp_search_retries: int,
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ManualRunnerError("seed must be an integer")
    if seed in EVALUATION_SEEDS:
        raise ManualRunnerError("reserved evaluation seeds 22001..22030 are not permitted in the manual runner")
    if not native_case and (isinstance(anchor, bool) or not isinstance(anchor, int) or anchor < 0):
        raise ManualRunnerError("anchor must be a non-negative integer")
    if isinstance(safety_max_rounds, bool) or not isinstance(safety_max_rounds, int) or safety_max_rounds <= 0:
        raise ManualRunnerError("safety-max-rounds must be a positive integer")
    if (
        isinstance(mrp_search_retries, bool) or not isinstance(mrp_search_retries, int)
        or mrp_search_retries < 0
    ):
        raise ManualRunnerError("mrp-search-retries must be a non-negative integer")


def _select_calibration_scenario(scenarios, scenario_id: str, anchor: int):
    selected = next((item for item in scenarios if item.scenario_id == scenario_id), None)
    if selected is None:
        available = ", ".join(item.scenario_id for item in scenarios)
        raise ManualRunnerError(f"unknown calibration scenario {scenario_id!r}; available: {available}")
    if selected.event_anchor_sensor != anchor:
        raise ManualRunnerError(
            f"scenario {scenario_id!r} has frozen anchor {selected.event_anchor_sensor}, not requested anchor {anchor}"
        )
    return selected


def _current_scenario(experiment_mode, scenario, runtime, network, config):
    if experiment_mode is ExperimentMode.BASELINE:
        return scenario
    physical = getattr(runtime.state, "physical_state", runtime.state)
    return scenario_with_current_hops(
        scenario, network, physical.live_nodes, config.mrp_parameters.communication_radius,
    )[0]


def _new_summary(mode, seed, scenario_id, anchor, safety_max_rounds):
    return {
        "mode": mode,
        "seed": seed,
        "scenario_id": scenario_id,
        "anchor": anchor,
        "pid": os.getpid(),
        "start_time": _now(),
        "safety_max_rounds": safety_max_rounds,
        "stop_policy": "ROUTE_EXHAUSTION",
        "status": None,
        "successful_rounds": 0,
        "last_successful_round_zero_based": None,
        "last_successful_round_one_based": None,
        "routing_failure_round_zero_based": None,
        "routing_failure_round_one_based": None,
        "fnd_round_zero_based": None,
        "fnd_round_one_based": None,
        "hnd_round_zero_based": None,
        "hnd_round_one_based": None,
        "lnd_round_zero_based": None,
        "lnd_round_one_based": None,
        "physical_energy": 0.0,
        "delivered_payloads": 0,
        "energy_per_delivered_payload": None,
        "final_residual_energy": None,
        "mean_residual_energy": None,
        "final_live_nodes": None,
        "final_dead_nodes": None,
        "last_success_details": None,
        "failure_stage": None,
        "failure_reason": None,
        "failure_details": None,
        "mrp_search_failures": 0,
        "mrp_search_retries": 0,
        "mrp_ch_discoveries_attempted": 0,
        "sants_executed": 0,
        "sants_succeeded": 0,
        "sant_success_rate": None,
        "unique_route_total": 0,
        "mean_unique_routes": None,
        "ttl_exhausted_sants": 0,
        "sink_neighbor_at_ttl_zero": 0,
        "selected_route_count": 0,
        "selected_route_hop_total": 0,
        "mean_selected_route_hops": None,
        "max_selected_route_hops": None,
        "selector_branches_visited": 0,
        "selector_branches_pruned": 0,
        "selector_complete_sets_evaluated": 0,
        "selector_theoretical_combinations": 0,
    }


def _record_mrp_search_diagnostics(summary, raw, base_dists, parameters):
    completed = tuple(getattr(raw, "ch_routing_results", ()) or ())
    phase2_results = [item.phase2_result for item in completed]
    failed_phase2 = getattr(raw, "failed_phase2_result", None)
    if failed_phase2 is not None:
        phase2_results.append(failed_phase2)
    summary["mrp_ch_discoveries_attempted"] += len(phase2_results)
    for phase2 in phase2_results:
        summary["sants_executed"] += getattr(
            phase2, "num_sants_executed", phase2.num_sants_requested
        )
        summary["sants_succeeded"] += phase2.num_sants_succeeded
        summary["unique_route_total"] += phase2.num_unique_routes
        for ant in getattr(phase2, "ant_results", ()):
            sant = ant.sant_result
            exhausted = sant.failure_reason == "ttl_exhausted"
            summary["ttl_exhausted_sants"] += exhausted
            summary["sink_neighbor_at_ttl_zero"] += (
                exhausted and sant.remaining_ttl == 0
                and base_dists[sant.visited_nodes[-1]] <= parameters.communication_radius
            )
    for item in completed:
        hops = len(item.phase3_result.selected_route) - 1
        summary["selected_route_count"] += 1
        summary["selected_route_hop_total"] += hops
        if "discovered_routes_before_pruning" in summary:
            before = item.phase2_result.num_unique_routes
            after = len(item.phase3_result.candidates)
            summary["discovered_routes_before_pruning"] += before
            summary["candidate_routes_after_pruning"] += after
            summary["pruned_routes_count"] += before - after
        current = summary["max_selected_route_hops"]
        summary["max_selected_route_hops"] = hops if current is None else max(current, hops)
    selection = getattr(raw, "lifetime_selection", None)
    if selection is not None:
        summary["selector_branches_visited"] += selection.branches_visited
        summary["selector_branches_pruned"] += selection.branches_pruned
        summary["selector_complete_sets_evaluated"] += selection.candidate_sets_evaluated
        summary["selector_theoretical_combinations"] += selection.theoretical_combinations


def _record_direct_mrp_diagnostics(summary, raw) -> None:
    diagnostics = getattr(raw, "direct_mrp_diagnostics", None)
    if diagnostics is None:
        raise ManualRunnerError("direct_mrp_topk result lacks routing diagnostics")
    direct_count = len(diagnostics.direct_cluster_heads)
    fallback_count = len(diagnostics.fallback_cluster_heads)
    summary["direct_ch_count"] += direct_count
    summary["fallback_ch_count"] += fallback_count
    summary["mrp_discoveries"] += diagnostics.mrp_discoveries
    summary["cached_route_uses"] += diagnostics.cached_route_uses
    summary["routes_before_topk"] += diagnostics.routes_before_top_k
    summary["routes_after_topk"] += diagnostics.routes_after_top_k
    summary["routes_pruned"] += diagnostics.routes_pruned
    summary["sant_count"] += diagnostics.sant_count
    summary["bant_count"] += diagnostics.bant_count
    summary["aant_count"] += diagnostics.aant_count
    summary["selected_route_count"] += direct_count
    summary["selected_route_hop_total"] += direct_count
    if direct_count:
        current = summary["max_selected_route_hops"]
        summary["max_selected_route_hops"] = 1 if current is None else max(current, 1)
    if raw.success:
        summary["direct_transmissions"] += direct_count
        summary["fallback_transmissions"] += fallback_count
        summary["direct_successes"] += direct_count
        summary["fallback_successes"] += fallback_count


def _record_ant_energy_diagnostics(summary, raw, row) -> None:
    control = getattr(raw, "ant_control_energy", None)
    if control is not None:
        for name in (
            "sant_tx_count", "sant_rx_count", "bant_tx_count",
            "bant_rx_count", "aant_tx_count", "aant_rx_count",
            "sant_energy", "bant_energy", "aant_energy",
        ):
            summary[name] += getattr(control, name)
        summary["total_ant_control_energy"] += control.total_energy
    if raw.success:
        data_energy = getattr(raw, "data_energy", None)
        if data_energy is None:
            data_energy = getattr(row, "actual_round_energy", None)
        if data_energy is None:
            data_energy = (
                row.cumulative_actual_energy
                - summary["physical_energy"]
            )
        summary["data_energy"] += data_energy


def _ratio(numerator, denominator):
    return None if not denominator else numerator / denominator


def _record_success(summary, row, node_count, details):
    summary["successful_rounds"] += 1
    summary["last_successful_round_zero_based"] = row.round_index
    summary["last_successful_round_one_based"] = row.round_index + 1
    summary["physical_energy"] = row.cumulative_actual_energy
    summary["delivered_payloads"] += row.delivered_payload_count or 0
    summary["last_success_details"] = details
    _record_terminal_state(summary, row)
    dead = row.dead_count
    if dead >= 1 and summary["fnd_round_zero_based"] is None:
        summary["fnd_round_zero_based"], summary["fnd_round_one_based"] = row.round_index, row.round_index + 1
    if dead >= (node_count + 1) // 2 and summary["hnd_round_zero_based"] is None:
        summary["hnd_round_zero_based"], summary["hnd_round_one_based"] = row.round_index, row.round_index + 1
    if dead >= node_count and summary["lnd_round_zero_based"] is None:
        summary["lnd_round_zero_based"], summary["lnd_round_one_based"] = row.round_index, row.round_index + 1


def _record_terminal_state(summary, row):
    summary["final_live_nodes"] = len(row.live_after)
    summary["final_dead_nodes"] = row.dead_count
    summary["final_residual_energy"] = row.residual_total
    summary["mean_residual_energy"] = row.residual_mean


def _progress_record(summary, row):
    current_rss, peak_rss = process_memory_bytes()
    return {
        "round": summary["last_successful_round_one_based"],
        "alive_nodes": summary["final_live_nodes"],
        "round_energy": row.actual_round_energy,
        "live": summary["final_live_nodes"],
        "dead": summary["final_dead_nodes"],
        "cumulative_energy": summary["physical_energy"],
        "fnd_round_one_based": summary["fnd_round_one_based"],
        "delivered_payloads": summary["delivered_payloads"],
        "current_rss_bytes": current_rss,
        "peak_rss_bytes": peak_rss,
    }


def _retryable_mrp_search_failure(raw) -> bool:
    """A stochastic search miss is not proof that physical routing is exhausted."""

    return (
        getattr(raw, "routing_failure_classification", None) == "MRP_SEARCH_FAILURE"
        or str(getattr(raw, "failure_reason", "") or "").startswith("MRP_SEARCH_FAILURE")
    )


def _capture_phase1_rng(experiment_mode, runtime):
    """Keep Hybrid AC-ACO selection stable while MRP search RNG advances."""

    if experiment_mode is not ExperimentMode.HYBRID or not getattr(runtime, "rngs", None):
        return None
    getstate = getattr(runtime.rngs[0], "getstate", None)
    return None if not callable(getstate) else getstate()


def _restore_phase1_rng(runtime, state) -> None:
    if state is not None:
        runtime.rngs[0].setstate(state)


def _validate_successful_mode(mode, raw) -> None:
    if mode not in {"mrp_pure", "mrp_hybrid", "mrp_hybrid_lifetime", "direct_mrp_topk"}:
        return
    heads = tuple(raw.phase1_result.selected_cluster_heads)
    selected = dict(raw.selected_routes or {})
    if set(heads) != set(selected):
        raise ManualRunnerError("successful multi-flow MRP round must retain one selected route for every selected CH")


def _success_details(mode, raw):
    if mode == "direct":
        return {
            "selected_chs": list(raw.phase1_result.selected_cluster_heads),
            "ch_count": len(raw.phase1_result.selected_cluster_heads),
            "greedy_routing_status": "SUCCESS",
        }
    if mode == "mrp_event":
        phase1, phase2, phase3 = raw.phase1_result, raw.phase2_result, raw.phase3_result
        return {
            "event_ch": phase1.cluster_head,
            "sants_requested": phase2.num_sants_requested,
            "sants_successful": phase2.num_sants_succeeded,
            "unique_routes": phase2.num_unique_routes,
            "multipath_ready": phase2.multipath_ready,
            "selected_route": list(phase3.selected_route),
            "selected_route_hops": len(phase3.selected_route) - 1,
        }
    routes = {int(head): tuple(route) for head, route in raw.selected_routes.items()}
    details = {
        "selected_chs": list(raw.phase1_result.selected_cluster_heads),
        "ch_count": len(raw.phase1_result.selected_cluster_heads),
        "selected_flow_count": len(routes),
        "selected_routes": {str(head): list(route) for head, route in routes.items()},
        "shared_relays": _shared_relays(routes),
        "mean_route_hops": sum(len(route) - 1 for route in routes.values()) / len(routes),
        "max_route_hops": max(len(route) - 1 for route in routes.values()),
    }
    if mode in {"mrp_hybrid", "direct_mrp_topk"}:
        details["phase3_candidate_pruning"] = [
            {
                "ch": item.cluster_head,
                "discovered_routes_before_pruning": item.phase2_result.num_unique_routes,
                "candidate_routes_after_pruning": len(item.phase3_result.candidates),
                "pruned_routes_count": (
                    item.phase2_result.num_unique_routes - len(item.phase3_result.candidates)
                ),
            }
            for item in raw.ch_routing_results
        ]
    if mode == "direct_mrp_topk":
        diagnostics = raw.direct_mrp_diagnostics
        details.update({
            "adaptation": "AC-ACO + Direct-first + MRP-fallback + Top-K",
            "direct_chs": list(diagnostics.direct_cluster_heads),
            "fallback_chs": list(diagnostics.fallback_cluster_heads),
            "mrp_discoveries": diagnostics.mrp_discoveries,
            "cached_route_uses": diagnostics.cached_route_uses,
            "sant_count": diagnostics.sant_count,
            "bant_count": diagnostics.bant_count,
            "aant_count": diagnostics.aant_count,
        })
    selection = getattr(raw, "lifetime_selection", None)
    if selection is not None:
        details.update({
            "experimental": True,
            "objective": "MAXIMIZE_MINIMUM_POST_ROUND_RESIDUAL_ENERGY",
            "minimum_residual_energy_after_round": selection.minimum_residual_after_round,
            "candidate_sets_evaluated": selection.candidate_sets_evaluated,
            "branches_visited": selection.branches_visited,
            "branches_pruned": selection.branches_pruned,
            "theoretical_combinations": selection.theoretical_combinations,
            "total_hop_count": selection.total_hop_count,
        })
    return details


def _terminal_status(mode, raw) -> str:
    reason = raw.failure_reason or ""
    if mode == "direct":
        if raw.failure_stage == "final_greedy" and "topology was unavailable" in reason:
            return "GREEDY_ROUTING_FAILURE"
        if raw.failure_stage == "ac_aco_phase1" and "no valid cluster-head set" in reason:
            return "AC_ACO_SELECTION_FAILURE"
        return "IMPLEMENTATION_OR_DOMAIN_FAILURE"
    if mode == "mrp_event":
        if reason.startswith("MRP_SEARCH_FAILURE"):
            return "MRP_SEARCH_FAILURE"
        if reason.startswith("PHYSICAL_CONNECTIVITY_EXHAUSTED"):
            return "PHYSICAL_CONNECTIVITY_EXHAUSTED"
        exhausted = (
            raw.failure_stage == "phase3" and "no unique successful routes" in reason
        ) or (
            raw.failure_stage == "phase1" and "no live CH candidate" in reason
        )
        return _EXHAUSTED_STATUSES[mode] if exhausted else "IMPLEMENTATION_OR_DOMAIN_FAILURE"
    if raw.failure_stage in {"mrp_clustering", "member_assignment"}:
        return "MRP_CLUSTERING_EXHAUSTED"
    classification = getattr(raw, "routing_failure_classification", None)
    if classification is not None:
        return classification
    if mode in {"mrp_hybrid", "mrp_hybrid_lifetime", "direct_mrp_topk"} and raw.failure_stage == "ac_aco_phase1":
        return "AC_ACO_SELECTION_FAILURE"
    return "IMPLEMENTATION_OR_DOMAIN_FAILURE"


def _failure_details(mode, raw):
    if mode == "direct":
        heads = () if raw.phase1_result is None else raw.phase1_result.selected_cluster_heads
        return {"selected_chs": list(heads), "greedy_status": raw.failure_stage, "reason": raw.failure_reason}
    if mode == "mrp_event":
        phase1, phase2 = raw.phase1_result, raw.phase2_result
        return {
            "event_ch": None if phase1 is None else phase1.cluster_head,
            "live_nodes": len(raw.state_before.live_nodes),
            "sants_requested": None if phase2 is None else phase2.num_sants_requested,
            "sants_successful": None if phase2 is None else phase2.num_sants_succeeded,
            "unique_routes": None if phase2 is None else phase2.num_unique_routes,
            "multipath_ready": None if phase2 is None else phase2.multipath_ready,
            "selected_route": None if raw.selected_route is None else list(raw.selected_route),
            "reason": raw.failure_reason,
        }
    phase1, completed = raw.phase1_result, raw.ch_routing_results
    failed_phase2 = getattr(raw, "failed_phase2_result", None)
    all_heads = () if phase1 is None else phase1.selected_cluster_heads
    diagnostics = getattr(raw, "direct_mrp_diagnostics", None)
    heads = (
        tuple(diagnostics.fallback_cluster_heads)
        if mode == "direct_mrp_topk" and diagnostics is not None
        else all_heads
    )
    completed_heads = {item.cluster_head for item in completed}
    failed = [head for head in heads if head not in completed_heads]
    details = {
        "selected_chs": list(all_heads),
        "selected_flow_count": len(completed),
        "required_flow_count": len(all_heads),
        "failed_ch_ids": (
            [raw.failed_cluster_head]
            if getattr(raw, "failed_cluster_head", None) is not None else failed[:1]
        ),
        "not_attempted_after_first_failure": failed[1:],
        "failed_ch_phase2": None if failed_phase2 is None else {
            "status": failed_phase2.status,
            "sants_requested": failed_phase2.num_sants_requested,
            "sants_successful": failed_phase2.num_sants_succeeded,
            "unique_routes": failed_phase2.num_unique_routes,
            "multipath_ready": failed_phase2.multipath_ready,
            "ttl_exhausted_sants": sum(
                ant.sant_result.failure_reason == "ttl_exhausted"
                for ant in getattr(failed_phase2, "ant_results", ())
            ),
        },
        "completed_chs": [
            {
                "ch": item.cluster_head,
                "phase2_status": item.phase2_result.status,
                "sants_successful": item.phase2_result.num_sants_succeeded,
                "unique_routes": item.phase2_result.num_unique_routes,
                "selected_route": list(item.phase3_result.selected_route),
            }
            for item in completed
        ],
        "routing_failure_classification": getattr(raw, "routing_failure_classification", None),
        "reason": raw.failure_reason,
    }
    if mode == "mrp_pure":
        details["routing_status_by_ch"] = {
            str(head): status
            for head, status in (getattr(raw, "routing_status_by_ch", None) or {}).items()
        }
    if mode == "direct_mrp_topk" and diagnostics is not None:
        details.update({
            "adaptation": "AC-ACO + Direct-first + MRP-fallback + Top-K",
            "direct_chs": list(diagnostics.direct_cluster_heads),
            "fallback_chs": list(diagnostics.fallback_cluster_heads),
            "mrp_discoveries": diagnostics.mrp_discoveries,
            "cached_route_uses": diagnostics.cached_route_uses,
            "sant_count": diagnostics.sant_count,
            "bant_count": diagnostics.bant_count,
            "aant_count": diagnostics.aant_count,
        })
    return details


def _shared_relays(routes):
    by_sensor: dict[int, list[dict[str, int]]] = {}
    for source_ch, route in routes.items():
        for index, sensor_id in enumerate(route[:-1]):
            by_sensor.setdefault(sensor_id, []).append({"flow_ch": source_ch, "next_hop": route[index + 1]})
    return {str(sensor): entries for sensor, entries in by_sensor.items() if len(entries) > 1}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _native_metadata(existing_map, config) -> dict[str, object]:
    identity = native_case_identity(existing_map, config)
    return {
        "case_id": NATIVE_CASE_ID,
        "case_source": NATIVE_CASE_SOURCE,
        "initial_state_hash": identity.canonical_sha256,
        "node_count": identity.node_count,
        "sink": list(identity.sink),
        "initial_energy": identity.initial_energy,
        "communication_radius": identity.communication_radius,
        "ch_proportion": identity.ch_proportion,
        "target_cluster_head_count": identity.target_cluster_head_count,
        "event_input_used": False,
    }
