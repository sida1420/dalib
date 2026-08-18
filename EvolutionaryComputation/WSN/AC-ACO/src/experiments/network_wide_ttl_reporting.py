"""Aggregate compact network-wide TTL calibration worker records."""

from collections import defaultdict
import statistics


def aggregate_ttl_trials(trials):
    """Return one reliability/cost record per TTL and fair MRP mode."""

    groups = defaultdict(list)
    for trial in trials:
        groups[(trial["ttl"], trial["mode"])].append(trial)
    return [
        _aggregate(ttl, mode, items)
        for (ttl, mode), items in sorted(groups.items())
    ]


def _aggregate(ttl, mode, items):
    summed = {
        field: sum(item[field] for item in items)
        for field in (
            "attempted_complete_rounds", "successful_complete_rounds",
            "mrp_search_failures", "ch_discoveries_attempted", "ch_routes_succeeded",
            "sants_executed", "sants_succeeded", "unique_route_total",
            "selected_route_count", "selected_route_hop_total", "ttl_exhausted_sants",
            "sink_neighbor_at_ttl_zero", "physical_energy", "delivered_payloads",
            "ch99_discoveries", "ch99_sants_executed", "ch99_sants_succeeded",
            "ch99_ttl_exhausted_sants", "ch99_sink_neighbor_at_ttl_zero",
        )
    }
    return {
        "ttl": ttl,
        "mode": mode,
        "trial_count": len(items),
        "completed_horizon_trials": sum(item["completed_horizon"] for item in items),
        **summed,
        "complete_round_success_fraction": _ratio(
            summed["successful_complete_rounds"], summed["attempted_complete_rounds"]
        ),
        "ch_route_success_fraction": _ratio(
            summed["ch_routes_succeeded"], summed["ch_discoveries_attempted"]
        ),
        "sant_success_fraction": _ratio(
            summed["sants_succeeded"], summed["sants_executed"]
        ),
        "mean_unique_routes": _ratio(
            summed["unique_route_total"], summed["ch_discoveries_attempted"]
        ),
        "mean_selected_route_hops": _ratio(
            summed["selected_route_hop_total"], summed["selected_route_count"]
        ),
        "max_selected_route_hops": max(
            (item["selected_route_hop_max"] for item in items if item["selected_route_hop_max"] is not None),
            default=None,
        ),
        "energy_per_delivered_payload": _ratio(
            summed["physical_energy"], summed["delivered_payloads"]
        ),
        "cpu_seconds_total": sum(item["cpu_seconds"] for item in items),
        "cpu_seconds_mean": statistics.fmean(item["cpu_seconds"] for item in items),
        "peak_rss_bytes": max(item["peak_rss_bytes"] for item in items),
        "max_first_hop_probability": max(
            (item["max_first_hop_probability"] for item in items if item["max_first_hop_probability"] is not None),
            default=None,
        ),
        "terminal_status_counts": _counts(item["status"] for item in items),
    }


def _counts(values):
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(sorted(counts.items()))


def _ratio(numerator, denominator):
    return None if not denominator else numerator / denominator
