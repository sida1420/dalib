"""Read-only workload, energy, routing, fairness, and lifetime derivations."""

from collections.abc import Mapping, Sequence
import math
import statistics

from .config import EvaluationView, ExperimentMode
from .types import FairnessAudit, LifetimeMetrics, ModeTrialResult, RoundMetrics
def baseline_metrics(result, trial_id: int, trial_seed: int, bit_count: float) -> RoundMetrics:
    heads = None if result.phase1_result is None else result.phase1_result.selected_cluster_heads
    return _record(
        ExperimentMode.BASELINE, trial_id, trial_seed, result, result.state_before.physical_state,
        result.state_after.physical_state, tuple(result.state_before.physical_state.live_nodes), heads,
        None, _cluster_sizes_from_topology(result.final_topology, heads), None, None, None, bit_count,
    )

def pure_mrp_metrics(result, trial_id: int, trial_seed: int, event_nodes: tuple[int, ...], bit_count: float) -> RoundMetrics:
    heads = None if result.phase1_result is None else (result.phase1_result.cluster_head,)
    route = None if result.selected_route is None else (result.selected_route,)
    discovered = None if result.phase2_result is None else (result.phase2_result.num_unique_routes,)
    ready = None if result.phase2_result is None else (result.phase2_result.multipath_ready,)
    lengths = None if result.phase3_result is None else (result.phase3_result.candidates[result.phase3_result.selected_index].path_length,)
    sizes = None if result.phase1_result is None else {result.phase1_result.cluster_head: len(result.phase1_result.members)}
    return _record(
        ExperimentMode.PURE_MRP, trial_id, trial_seed, result, result.state_before, result.state_after,
        event_nodes, heads, route, sizes, discovered, ready, lengths, bit_count,
    )

def network_wide_mrp_metrics(result, trial_id: int, trial_seed: int, bit_count: float) -> RoundMetrics:
    heads = None if result.phase1_result is None else result.phase1_result.selected_cluster_heads
    routes = (
        None if result.selected_routes is None
        else tuple(result.selected_routes[head] for head in heads if head in result.selected_routes)
    )
    discovered = tuple(item.phase2_result.num_unique_routes for item in result.ch_routing_results) or None
    ready = tuple(item.phase2_result.multipath_ready for item in result.ch_routing_results) or None
    lengths = tuple(
        item.phase3_result.candidates[item.phase3_result.selected_index].path_length
        for item in result.ch_routing_results
    ) or None
    return _record(
        ExperimentMode.MRP_NETWORK_WIDE, trial_id, trial_seed, result,
        result.state_before, result.state_after, tuple(result.state_before.live_nodes),
        heads, routes, result.members_by_head, discovered, ready, lengths,
        bit_count, result.final_plan,
    )

def hybrid_metrics(result, trial_id: int, trial_seed: int, bit_count: float) -> RoundMetrics:
    heads = None if result.phase1_result is None else result.phase1_result.selected_cluster_heads
    routes = None if result.selected_routes is None else tuple(result.selected_routes[head] for head in heads)
    discovered = tuple(item.phase2_result.num_unique_routes for item in result.ch_routing_results) or None
    ready = tuple(item.phase2_result.multipath_ready for item in result.ch_routing_results) or None
    lengths = tuple(item.phase3_result.candidates[item.phase3_result.selected_index].path_length for item in result.ch_routing_results) or None
    return _record(
        ExperimentMode.HYBRID, trial_id, trial_seed, result, result.state_before.physical_state,
        result.state_after.physical_state, tuple(result.state_before.physical_state.live_nodes), heads,
        routes, result.members_by_head, discovered, ready, lengths, bit_count, result.final_plan,
    )

def derive_lifetime_metrics(rounds: Sequence[RoundMetrics], node_count: int) -> LifetimeMetrics:
    """Use zero-based round indices; unreached milestones are explicitly censored."""

    threshold = math.ceil(node_count / 2)
    fnd = _first_round(rounds, lambda item: item.dead_count >= 1)
    hnd = _first_round(rounds, lambda item: item.dead_count >= threshold)
    lnd = _first_round(rounds, lambda item: item.dead_count >= node_count)
    return LifetimeMetrics(fnd, hnd, lnd, fnd is None, hnd is None, lnd is None)

def assess_trial_fairness(
    mode_results: Mapping[ExperimentMode, ModeTrialResult],
    evaluation_view: EvaluationView = EvaluationView.NATIVE_ALGORITHM_SEMANTICS,
) -> FairnessAudit:
    """Classify raw comparisons from measured workload rather than algorithm names."""

    if not isinstance(evaluation_view, EvaluationView):
        raise ValueError("evaluation_view must be an EvaluationView")
    totals = {
        mode: _total_payloads(result.rounds)
        for mode, result in mode_results.items()
    }
    values = tuple(totals.values())
    if not values or any(value is None for value in values):
        return FairnessAudit(
            "NATIVE-SEMANTICS-ONLY", "UNAVAILABLE", "NATIVE-SEMANTICS-ONLY",
            "one or more modes had no successful delivered workload", evaluation_view,
        )
    equal_workload = len(set(values)) == 1
    raw_status = "DIRECTLY_COMPARABLE" if equal_workload else "COMPARABLE_ONLY_WITH_NORMALIZATION"
    normalized_status = _normalized_status(mode_results, evaluation_view)
    lifetime_status, lifetime_reason = _lifetime_status(mode_results, equal_workload)
    return FairnessAudit(
        raw_status, normalized_status, lifetime_status,
        lifetime_reason if equal_workload else "represented delivered payload counts differ across modes; " + lifetime_reason,
        evaluation_view,
    )

def aggregate_mode_trials(
    mode_results: Sequence[ModeTrialResult], evaluation_view: EvaluationView,
) -> dict[str, object]:
    """Aggregate only complete trials; raw failed rows remain separately available."""

    if not isinstance(evaluation_view, EvaluationView):
        raise ValueError("evaluation_view must be an EvaluationView")
    complete = [result for result in mode_results if _complete_trial(result)]
    totals = [sum(item.actual_round_energy for item in result.rounds) for result in complete]
    normalized = [
        total / sum(item.delivered_payload_count for item in result.rounds)
        for total, result in zip(totals, complete)
    ]
    incomplete_count = len(mode_results) - len(complete)
    return {
        "reporting_view": evaluation_view.value,
        "completion": {
            "complete_trial_count": len(complete), "incomplete_trial_count": incomplete_count,
            "status": "COMPLETE" if not incomplete_count else "INCOMPLETE_TRIALS_EXCLUDED",
        },
        "actual_energy": _summary(totals),
        "energy_per_delivered_payload": (
            _summary(normalized) if evaluation_view is EvaluationView.NORMALIZED_REPORTING
            else _not_selected_summary()
        ),
    }

def _record(mode, trial_id, seed, result, before, after, declared, heads, routes, sizes, discovered, ready, lengths, bit_count, physical_plan=None):
    topology_ids = _topology_sensor_ids(result.final_topology if physical_plan is None else physical_plan)
    success = result.success
    represented = len(topology_ids) if success else None
    delivered = represented if success else None
    bits = None if delivered is None else delivered * bit_count
    energy = result.e_sum if success else None
    residual = tuple(after.residual_e)
    return RoundMetrics(
        mode, trial_id, seed, result.round_index, tuple(before.live_nodes), tuple(after.live_nodes),
        len(after.dead_nodes), tuple(result.newly_dead), tuple(declared), represented, delivered, bits,
        heads, routes, _as_cluster_sizes(sizes), energy,
        None if energy is None or not delivered else energy / delivered,
        None if energy is None or not bits else energy / bits,
        None if energy is None or not represented else energy / represented,
        after.cumulative_actual_energy, sum(residual), statistics.fmean(residual), statistics.pstdev(residual), min(residual),
        None if routes is None else tuple(len(route) - 1 for route in routes), lengths, discovered, ready,
        success, result.failure_stage, result.failure_reason,
    )

def _topology_sensor_ids(root: object | None) -> set[int]:
    if root is None:
        return set()
    if hasattr(root, "physical_sensor_ids"):
        return set(root.physical_sensor_ids)
    found, stack = set(), [root]
    while stack:
        node = stack.pop()
        if node.idx != -1:
            found.add(node.idx)
        stack.extend(node.branches)
    return found

def _cluster_sizes_from_topology(root, heads):
    if root is None or heads is None:
        return None
    nodes, stack = {}, [root]
    while stack:
        node = stack.pop()
        nodes[node.idx] = node
        stack.extend(node.branches)
    return {head: sum(not child.isCH for child in nodes[head].branches) for head in heads}

def _as_cluster_sizes(values):
    if values is None:
        return None
    return {
        head: len(size) if isinstance(size, (tuple, list, set, frozenset)) else int(size)
        for head, size in values.items()
    }

def _first_round(rounds, predicate):
    return next((item.round_index for item in rounds if predicate(item)), None)

def _total_payloads(rounds):
    if not rounds or any(not item.success or item.delivered_payload_count is None for item in rounds):
        return None
    return sum(item.delivered_payload_count for item in rounds)

def _normalized_status(mode_results, evaluation_view):
    usable = all(_complete_trial(result) for result in mode_results.values())
    if not usable:
        return "UNAVAILABLE"
    return "DIRECTLY_COMPARABLE" if evaluation_view is EvaluationView.NORMALIZED_REPORTING else "AVAILABLE_NOT_PRIMARY"

def _lifetime_status(mode_results, equal_workload):
    horizons = {len(result.rounds) for result in mode_results.values()}
    censored = any(
        result.lifetime.fnd_censored or result.lifetime.hnd_censored or result.lifetime.lnd_censored
        for result in mode_results.values()
    )
    if not equal_workload:
        return "NATIVE-SEMANTICS-ONLY", "lifetime traffic workloads differ"
    if censored:
        status = "CENSORED_SURVIVAL_ANALYSIS_REQUIRED" if len(horizons) == 1 else "CENSORED_HORIZON_MISMATCH"
        return status, "one or more lifetime milestones are censored"
    return "DIRECTLY_COMPARABLE", "all modes represented the same delivered payload count"

def _complete_trial(result):
    return bool(result.rounds) and all(
        item.success and item.actual_round_energy is not None and item.delivered_payload_count not in (None, 0)
        for item in result.rounds
    )

def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "stddev": None, "min": None, "max": None, "sample_count": 0}
    return {
        "mean": statistics.fmean(values), "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values), "sample_count": len(values),
    }


def _not_selected_summary() -> dict[str, float | int | None | str]:
    return {"mean": None, "stddev": None, "min": None, "max": None, "sample_count": 0, "status": "NOT_SELECTED"}
