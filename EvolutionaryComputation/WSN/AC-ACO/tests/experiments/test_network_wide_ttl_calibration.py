from types import SimpleNamespace
from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.network_wide_ttl_reporting import aggregate_ttl_trials
from experiments.network_wide_ttl_trial import _validate_ttl_only_configuration
from mrp.runner_types import PheromoneLifecycle


def _trial(**overrides):
    value = {
        "ttl": 4, "mode": "mrp_pure", "completed_horizon": True,
        "attempted_complete_rounds": 10, "successful_complete_rounds": 10,
        "mrp_search_failures": 0, "ch_discoveries_attempted": 100,
        "ch_routes_succeeded": 100, "sants_executed": 2000,
        "sants_succeeded": 400, "unique_route_total": 300,
        "selected_route_count": 100, "selected_route_hop_total": 250,
        "selected_route_hop_max": 4, "ttl_exhausted_sants": 200,
        "sink_neighbor_at_ttl_zero": 50, "physical_energy": 2.0,
        "delivered_payloads": 1000, "ch99_discoveries": 3,
        "ch99_sants_executed": 60, "ch99_sants_succeeded": 12,
        "ch99_ttl_exhausted_sants": 6,
        "ch99_sink_neighbor_at_ttl_zero": 2, "cpu_seconds": 1.5,
        "peak_rss_bytes": 1234, "max_first_hop_probability": 0.8,
        "status": "COMPLETED_CALIBRATION_HORIZON",
    }
    value.update(overrides)
    return value


def test_ttl_aggregate_preserves_round_reliability_and_cost_denominators():
    result = aggregate_ttl_trials([_trial(), _trial()])[0]
    assert result["trial_count"] == 2
    assert result["complete_round_success_fraction"] == 1.0
    assert result["ch_route_success_fraction"] == 1.0
    assert result["sant_success_fraction"] == 0.2
    assert result["mean_unique_routes"] == 3.0
    assert result["mean_selected_route_hops"] == 2.5
    assert result["energy_per_delivered_payload"] == 0.002
    assert result["cpu_seconds_total"] == 3.0


def test_ttl_only_guard_rejects_any_other_parameter_change():
    fixed = SimpleNamespace(
        ttl=4, num_sants=20, lambda_coefficient=9.04421850555005,
        c0=1.0, c=0.005, c1=0.003696666640752299,
        config=SimpleNamespace(alpha=2.0, beta=2.0, rho=0.2,
                               initial_pheromone=0.01, aant_probability=0.001),
        heuristic_bounds=SimpleNamespace(
            mu_min=0.00770435077647942, mu_max=0.019383167511744133,
            eta_min=0.00770435077647942, eta_max=0.019383167511744133,
        ),
    )
    config = SimpleNamespace(
        mrp_parameters=fixed,
        mrp_pheromone_lifecycle=PheromoneLifecycle.RESET_PER_DISCOVERY,
    )
    _validate_ttl_only_configuration(config, 4)
    config.mrp_parameters.c = 0.006
    with pytest.raises(RuntimeError, match="TTL-only parameter guard"):
        _validate_ttl_only_configuration(config, 4)
