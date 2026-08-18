import pytest

from ac_aco_mrp.multiflow import HybridMultiFlowError, build_hybrid_multiflow_plan
from ac_aco_mrp.multiflow_energy import evaluate_hybrid_multiflow_energy
from evaluate import E_data_receiving, E_transmitting, energy_consumption
from mrp.topology import materialize_topology


def test_conflict_free_multiflow_energy_matches_existing_tree(hybrid_case):
    plan = build_hybrid_multiflow_plan(
        (0, 1), {0: (2,), 1: (3,)},
        {0: (0, 2, -1), 1: (1, -1)}, {0, 1, 2, 3}, 4,
    )
    tree = materialize_topology(
        (0, 1), {0: (2,), 1: (3,)},
        {0: (0, 2, -1), 1: (1, -1)}, {0, 1, 2, 3}, 4,
    )
    old_e_m, old_sum = energy_consumption(
        hybrid_case["nodes"], tree, hybrid_case["parameters"].mrp.d0,
        hybrid_case["parameters"].mrp.bit_count, hybrid_case["parameters"].mrp.ctrl_bit,
        hybrid_case["base_dists"], hybrid_case["distances"], hybrid_case["parameters"].mrp.e_elec,
        hybrid_case["parameters"].mrp.e_agg, hybrid_case["parameters"].mrp.free_space_coeff,
        hybrid_case["parameters"].mrp.multipath_coeff,
    )
    actual = evaluate_hybrid_multiflow_energy(
        plan, hybrid_case["distances"], hybrid_case["base_dists"], hybrid_case["parameters"].mrp,
    )
    assert actual.e_m_list == pytest.approx(old_e_m)
    assert actual.e_sum == pytest.approx(old_sum)


def test_different_parent_shared_relay_and_disjoint_flows_are_valid(hybrid_case):
    conflict = build_hybrid_multiflow_plan(
        (0, 1), {0: (2,), 1: (3,)},
        {0: (0, 2, -1), 1: (1, 2, 3, -1)}, {0, 1, 2, 3}, 4,
    )
    disjoint = build_hybrid_multiflow_plan(
        (0, 1), {0: (2,), 1: (3,)},
        {0: (0, 2, -1), 1: (1, 3, -1)}, {0, 1, 2, 3}, 4,
    )
    result = evaluate_hybrid_multiflow_energy(
        conflict, hybrid_case["distances"], hybrid_case["base_dists"], hybrid_case["parameters"].mrp,
    )
    assert {flow.source_ch for flow in conflict.selected_flows} == {0, 1}
    assert result.e_m_list[2] > 0 and len(disjoint.selected_flows) == 2


def test_observed_84_different_parent_flows_are_retained_without_a_global_parent_map():
    plan = build_hybrid_multiflow_plan(
        (66, 84, 10), {66: (), 84: (), 10: (23,)},
        {66: (66, -1), 84: (84, 66, -1), 10: (10, 84, 23, -1)},
        {10, 23, 66, 84}, 100,
    )
    assert {flow.source_ch for flow in plan.selected_flows} == {10, 66, 84}
    assert plan.local_payload_flow_by_sensor[84] == 84
    assert tuple(flow.route for flow in plan.selected_flows) == (
        (66, -1), (84, 66, -1), (10, 84, 23, -1),
    )


def test_same_edge_payloads_accumulate_without_double_own_payload(hybrid_case):
    parameters = hybrid_case["parameters"].mrp
    distances = tuple(tuple(row[:3]) for row in hybrid_case["distances"][:3])
    plan = build_hybrid_multiflow_plan(
        (0, 1), {0: (2,), 1: ()},
        {0: (0, 2, -1), 1: (1, 0, 2, -1)}, {0, 1, 2}, 3,
    )
    result = evaluate_hybrid_multiflow_energy(plan, distances, hybrid_case["base_dists"][:3], parameters)
    tx = lambda distance, bits: E_transmitting(
        parameters.e_elec, parameters.free_space_coeff, parameters.multipath_coeff,
        distance, parameters.d0, bits,
    )
    assert result.edge_payload_bits[(0, 2)] == pytest.approx(2 * parameters.bit_count)
    assert result.edge_payload_bits[(2, -1)] == pytest.approx(3 * parameters.bit_count)
    assert result.e_m_list[0] == pytest.approx(
        E_data_receiving(parameters.e_elec, parameters.bit_count) + tx(2.0, 2 * parameters.bit_count)
    )
    assert result.e_m_list[2] == pytest.approx(
        E_data_receiving(parameters.e_elec, 2 * parameters.bit_count) + tx(3.0, 3 * parameters.bit_count)
    )


def test_loop_inside_one_flow_is_rejected(hybrid_case):
    with pytest.raises(HybridMultiFlowError, match="loop within route"):
        build_hybrid_multiflow_plan(
            (0,), {0: (1, 2, 3)}, {0: (0, 2, 0, -1)}, {0, 1, 2, 3}, 4,
        )
