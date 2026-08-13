import ast
from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import energy_consumption
from mrp.topology import (
    TopologyValidationError,
    assign_members_to_cluster_heads,
    materialize_topology,
    validate_topology,
)
from node import Node
from point import Point


def _nodes() -> list[Point]:
    return [Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0), Point(4, 0)]


def _distances(nodes: list[Point]) -> list[list[float]]:
    return [[abs(left - right) for right in nodes] for left in nodes]


def _children_by_id(root):
    result = {}
    stack = [root]
    while stack:
        node = stack.pop()
        result[node.idx] = node
        stack.extend(node.branches)
    return result


def test_assigns_each_live_member_to_nearest_cluster_head():
    nodes = _nodes()
    assignments = assign_members_to_cluster_heads(
        nodes, [0, 4], 10, _distances(nodes), [1, 1, 1, 1, 1]
    )
    # Sensor 2 is equidistant.  The expected result deliberately follows the
    # existing KDTree's deterministic traversal/tie behavior.
    assert assignments == {0: [1], 4: [2, 3]}


def test_assignment_rejects_member_outside_radius():
    nodes = _nodes()
    assert assign_members_to_cluster_heads(
        nodes, [0], 1, _distances(nodes), [1, 1, 1, 1, 1]
    ) is None


def test_assignment_excludes_dead_nodes():
    nodes = _nodes()
    assignments = assign_members_to_cluster_heads(
        nodes, [0, 4], 10, _distances(nodes), [1, 1, 0, 1, 1]
    )
    assert assignments == {0: [1], 4: [3]}


def test_materializes_direct_ch_to_sink_topology():
    root = materialize_topology([1], {1: [0]}, {1: [1, -1]}, {0, 1}, 2)
    assert root is not None
    nodes = _children_by_id(root)
    assert nodes[1].p_idx == -1 and nodes[1].isCH
    assert nodes[0].p_idx == 1 and not nodes[0].isCH


def test_materializes_multihop_route_with_correct_parent_links():
    root = materialize_topology([0], {0: [1]}, {0: [0, 2, -1]}, {0, 1, 2}, 3)
    assert root is not None
    nodes = _children_by_id(root)
    assert nodes[0].p_idx == 2 and nodes[0].isCH
    assert nodes[2].p_idx == -1 and nodes[2].isCH
    assert nodes[1].p_idx == 0 and not nodes[1].isCH


def test_materializes_converging_routes_using_one_relay_node():
    root = materialize_topology(
        [0, 1], {0: [], 1: []}, {0: [0, 2, -1], 1: [1, 2, -1]}, {0, 1, 2}, 3
    )
    assert root is not None
    nodes = _children_by_id(root)
    assert nodes[2].p_idx == -1
    assert sorted(branch.idx for branch in nodes[2].branches) == [0, 1]


def test_rejects_relay_with_conflicting_parents():
    assert materialize_topology(
        [0, 1],
        {0: [], 1: []},
        {0: [0, 2, -1], 1: [1, 2, 3, -1]},
        {0, 1, 2, 3},
        4,
    ) is None


def test_rejects_route_cycle():
    assert materialize_topology([0], {0: []}, {0: [0, 1, 0, -1]}, {0, 1}, 2) is None


def test_member_can_relay_another_ch_when_tree_parent_is_compatible():
    root = materialize_topology(
        [0, 1],
        {0: [2], 1: []},
        {0: [0, -1], 1: [1, 2, 0, -1]},
        {0, 1, 2},
        3,
    )
    assert root is not None
    nodes = _children_by_id(root)
    assert nodes[2].p_idx == 0
    assert nodes[2].isCH  # forwarding-data flag, despite cluster membership
    assert nodes[1].p_idx == 2
    assert len(nodes) == 4  # one root plus one Node per physical sensor


def test_member_can_relay_after_its_own_cluster_head_without_duplication():
    root = materialize_topology(
        [1], {1: [4]}, {1: [1, 4, 7, -1]}, {1, 4, 7}, 8
    )
    assert root is not None
    nodes = _children_by_id(root)
    assert nodes[1].p_idx == 4
    assert nodes[4].p_idx == 7
    assert nodes[4].isCH  # forwarding-data flag, not a new cluster-head claim
    assert len(nodes) == 4  # one root plus physical sensors 1, 4, 7


def test_validator_rejects_inconsistent_parent_link():
    root = Node(-1)
    child = Node(0, isCH=True)
    child.set_parent(99)
    root.branches.append(child)

    with pytest.raises(TopologyValidationError, match="inconsistent p_idx"):
        validate_topology(
            root,
            node_count=1,
            live_nodes={0},
            cluster_heads=[0],
            members_by_head={0: []},
            selected_routes={0: [0, -1]},
        )


def test_existing_energy_consumer_accepts_materialized_topology():
    nodes = [Point(0, 0), Point(2, 0), Point(4, 0)]
    root = materialize_topology([0], {0: [1]}, {0: [0, 2, -1]}, {0, 1, 2}, 3)
    assert root is not None

    e_m_list, e_sum = energy_consumption(
        nodes=nodes,
        net=root,
        d0=10,
        bit_count=2000,
        ctrl_bit=100,
        base_dists=[10, 8, 6],
        dist_matrix=_distances(nodes),
        E_elec=50e-9,
        E_agg=5e-9,
        free_space_coeff=10e-12,
        multipath_coeff=0.0013e-12,
    )
    assert len(e_m_list) == len(nodes)
    assert e_sum > 0
    assert all(energy >= 0 for energy in e_m_list)


def test_existing_energy_consumer_accepts_member_relay_topology():
    nodes = [Point(0, 0), Point(2, 0), Point(4, 0)]
    root = materialize_topology(
        [0, 1],
        {0: [2], 1: []},
        {0: [0, -1], 1: [1, 2, 0, -1]},
        {0, 1, 2},
        3,
    )
    assert root is not None

    e_m_list, e_sum = energy_consumption(
        nodes=nodes,
        net=root,
        d0=10,
        bit_count=2000,
        ctrl_bit=100,
        base_dists=[10, 8, 6],
        dist_matrix=_distances(nodes),
        E_elec=50e-9,
        E_agg=5e-9,
        free_space_coeff=10e-12,
        multipath_coeff=0.0013e-12,
    )
    relay_member = _children_by_id(root)[2]
    assert relay_member.relay_data == 4000
    assert e_sum > 0 and all(energy >= 0 for energy in e_m_list)


def test_existing_energy_consumer_accepts_member_relay_after_own_ch():
    nodes = [Point(index * 2, 0) for index in range(8)]
    root = materialize_topology(
        [1], {1: [4]}, {1: [1, 4, 7, -1]}, {1, 4, 7}, 8
    )
    assert root is not None

    e_m_list, e_sum = energy_consumption(
        nodes=nodes,
        net=root,
        d0=10,
        bit_count=2000,
        ctrl_bit=100,
        base_dists=[20 - index * 2 for index in range(8)],
        dist_matrix=_distances(nodes),
        E_elec=50e-9,
        E_agg=5e-9,
        free_space_coeff=10e-12,
        multipath_coeff=0.0013e-12,
    )
    topology_nodes = _children_by_id(root)
    assert topology_nodes[4].relay_data == 4000
    assert len(e_m_list) == len(nodes) and e_sum > 0


def test_mrp_package_does_not_duplicate_shared_calculations():
    source = (SRC / "mrp" / "topology.py").read_text(encoding="utf-8")
    forbidden_definitions = (
        "def mrp_distance",
        "def E_data_receiving",
        "def E_data_aggregating",
        "def E_transmitting",
        "def E_m(",
        "def energy_consumption",
    )
    assert not any(definition in source for definition in forbidden_definitions)
    tree = ast.parse(source)
    residual_mutations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        and "residual_e" in ast.unparse(node)
    ]
    assert residual_mutations == []
