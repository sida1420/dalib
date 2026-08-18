from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import E_data_receiving, E_transmitting
from mrp import (
    HeuristicBounds, MRPConfig, PheromoneLifecycle, PureMRPParameters,
    PureMRPState, discover_mrp_phase2_routes,
)
from mrp.multi_ch_routing import MRPRoutingFailure, route_cluster_heads_with_mrp
import mrp.phase2.routing as routing_module
from mrp.phase2.control_energy import (
    AntControlEnergyLedger, validate_ant_energy_parameters,
)


class FixedRNG:
    def __init__(self, *draws):
        self.draws = iter(draws)

    def random(self):
        return next(self.draws)


def _ledger(bits=1):
    return AntControlEnergyLedger(
        2, bits, ((0.0, 3.0), (3.0, 0.0)), (7.0, 3.0),
        2.0, 1.0, 1.0, 10.0,
    )


def _discovery(**overrides):
    values = dict(
        start_cluster_head=0, live_nodes={0, 1}, residual_e=(10.0, 8.0),
        dist_matrix=((0.0, 3.0), (3.0, 0.0)), base_dists=(7.0, 3.0),
        hop_counts={0: 2, 1: 1}, communication_radius=5.0,
        bounds=HeuristicBounds(0.1, 1000.0, 0.1, 1000.0),
        config=MRPConfig(aant_probability=0.0), lambda_coefficient=0.1,
        ttl=2, num_sants=1, rng=FixedRNG(0.5, 0.0), c0=1.0,
        e_elec=2.0, free_space_coeff=1.0, multipath_coeff=1.0,
        d0=10.0, bit_count=1.0, c=0.0, c1=0.01,
    )
    values.update(overrides)
    return discover_mrp_phase2_routes(**values)


def test_sensor_hop_and_sink_hop_charge_only_sensor_radios():
    ledger = _ledger()
    ledger.record_sant_hop("NORMAL_SANT", 0, 1)
    ledger.record_sant_hop("TERMINAL_SINK", 1, -1)
    result = ledger.snapshot()

    tx_01 = E_transmitting(2.0, 1.0, 1.0, 3.0, 10.0, 1)
    tx_sink = E_transmitting(2.0, 1.0, 1.0, 3.0, 10.0, 1)
    rx = E_data_receiving(2.0, 1)
    assert result.e_m_list == pytest.approx((tx_01, rx + tx_sink))
    assert (result.sant_tx_count, result.sant_rx_count) == (2, 1)
    assert result.sant_energy == pytest.approx(tx_01 + rx + tx_sink)


def test_bant_reverse_path_charges_sensor_receivers_but_not_sink_sender():
    ledger = _ledger()
    ledger.record_bant_route((-1, 1, 0))
    result = ledger.snapshot()

    rx = E_data_receiving(2.0, 1)
    tx = E_transmitting(2.0, 1.0, 1.0, 3.0, 10.0, 1)
    assert result.e_m_list == pytest.approx((rx, rx + tx))
    assert (result.bant_tx_count, result.bant_rx_count) == (1, 2)
    assert result.bant_energy == pytest.approx(2 * rx + tx)


def test_aant_is_one_forwarding_packet_not_sant_plus_aant():
    ledger = _ledger()
    ledger.record_sant_hop("AANT", 0, 1)
    result = ledger.snapshot()

    assert (result.aant_tx_count, result.aant_rx_count) == (1, 1)
    assert (result.sant_tx_count, result.sant_rx_count) == (0, 0)
    assert result.total_energy == pytest.approx(result.aant_energy)


def test_ttl_failure_charges_only_the_hop_that_was_transmitted():
    result = _discovery(
        ttl=1, charge_ant_energy=True, ant_control_packet_bits=1,
    )

    assert result.ant_results[0].sant_result.failure_reason == "ttl_exhausted"
    assert result.control_energy.sant_tx_count == 1
    assert result.control_energy.sant_rx_count == 1
    assert result.control_energy.bant_tx_count == 0


def test_success_charges_sant_and_complete_bant_reverse_path():
    result = _discovery(
        charge_ant_energy=True, ant_control_packet_bits=1,
    )

    assert result.successful_ant_results
    assert (result.control_energy.sant_tx_count, result.control_energy.sant_rx_count) == (2, 1)
    assert (result.control_energy.bant_tx_count, result.control_energy.bant_rx_count) == (1, 2)
    assert result.control_energy.total_energy > 0


def test_exception_after_transmitted_hop_preserves_active_discovery_energy(monkeypatch):
    def fail_after_sant(*args, **kwargs):
        raise RuntimeError("route-quality fault after radio transmission")

    monkeypatch.setattr(
        routing_module, "_evaluate_route_quality_from_validated_snapshot",
        fail_after_sant,
    )
    parameters = PureMRPParameters(
        communication_radius=5.0,
        heuristic_bounds=HeuristicBounds(0.1, 1000.0, 0.1, 1000.0),
        config=MRPConfig(aant_probability=0.0),
        lambda_coefficient=0.1,
        ttl=2,
        num_sants=1,
        c0=1.0,
        c=0.0,
        c1=0.01,
        d0=10.0,
        bit_count=1.0,
        ctrl_bit=0.0,
        e_elec=2.0,
        e_agg=0.0,
        free_space_coeff=1.0,
        multipath_coeff=1.0,
        charge_ant_energy=True,
        ant_control_packet_bits=1,
    )
    with pytest.raises(MRPRoutingFailure) as captured:
        route_cluster_heads_with_mrp(
            (0,), {0, 1}, PureMRPState((10.0, 8.0), (0, 1), frozenset()),
            ((0.0, 3.0), (3.0, 0.0)), (7.0, 3.0), {0: 2, 1: 1},
            parameters, FixedRNG(0.5, 0.0), PheromoneLifecycle.RESET_PER_DISCOVERY,
        )

    assert captured.value.control_energy.sant_tx_count == 2
    assert captured.value.control_energy.total_energy > 0


def test_disabled_feature_preserves_routing_and_rng_behavior():
    original = _discovery()
    disabled = _discovery(
        charge_ant_energy=False, ant_control_packet_bits=None,
    )

    assert original == disabled
    assert disabled.control_energy.is_zero


@pytest.mark.parametrize(
    "enabled,bits",
    ((True, None), (True, 0), (True, -1), (True, 1.5), (1, 1)),
)
def test_invalid_accounting_configuration_is_rejected(enabled, bits):
    with pytest.raises(ValueError):
        validate_ant_energy_parameters(enabled, bits)
