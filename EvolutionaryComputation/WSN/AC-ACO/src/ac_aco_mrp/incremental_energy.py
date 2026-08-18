"""Exact rollback ledger for experimental multi-flow route-set search."""

from collections import defaultdict

from evaluate import E_data_receiving, E_transmitting


class IncrementalMultiFlowLedger:
    """Accumulate the existing multi-flow payload semantics route by route."""

    def __init__(self, heads, members, node_count, dist_matrix, base_dists, parameters):
        self.heads = frozenset(heads)
        self.members = members
        self.dist_matrix = dist_matrix
        self.base_dists = base_dists
        self.parameters = parameters
        self.e_m = [0.0] * node_count
        self.edge_bits = defaultdict(float)
        self.owners = {head: head for head in heads}
        self.payload_added = set()
        self.route_counts = defaultdict(int)

    @property
    def total_energy(self):
        return sum(self.e_m)

    def push_flow(self, source_ch, route):
        token = {"edges": [], "owners": [], "payloads": [], "route_nodes": []}
        for sensor in route[:-1]:
            self.route_counts[sensor] += 1
            token["route_nodes"].append(sensor)
        self._add_payload(route, 0, token)
        self.payload_added.add(source_ch)
        token["payloads"].append(source_ch)
        for index, sensor in enumerate(route[:-1]):
            if sensor in self.owners:
                continue
            self.owners[sensor] = source_ch
            token["owners"].append(sensor)
            self._add_payload(route, index, token)
            self.payload_added.add(sensor)
            token["payloads"].append(sensor)
        return token

    def push_terminal_members(self):
        token = {"edges": [], "owners": [], "payloads": [], "route_nodes": []}
        for head, members in self.members.items():
            for member in members:
                if not self.route_counts[member]:
                    self._add_edge(member, head, self.parameters.bit_count, token)
        return token

    def rollback(self, token):
        for edge, old_bits, sender_energy, receiver_energy in reversed(token["edges"]):
            sender, receiver = edge
            self.e_m[sender] = sender_energy
            if receiver != -1:
                self.e_m[receiver] = receiver_energy
            if old_bits:
                self.edge_bits[edge] = old_bits
            else:
                self.edge_bits.pop(edge, None)
        for sensor in reversed(token["payloads"]):
            self.payload_added.remove(sensor)
        for sensor in reversed(token["owners"]):
            del self.owners[sensor]
        for sensor in reversed(token["route_nodes"]):
            self.route_counts[sensor] -= 1

    def _add_payload(self, route, start, token):
        for sender, receiver in zip(route[start:], route[start + 1:]):
            self._add_edge(sender, receiver, self.parameters.bit_count, token)

    def _add_edge(self, sender, receiver, bits, token):
        edge = (sender, receiver)
        old_bits = self.edge_bits[edge]
        token["edges"].append((
            edge, old_bits, self.e_m[sender],
            None if receiver == -1 else self.e_m[receiver],
        ))
        self._set_edge(edge, old_bits + bits)

    def _set_edge(self, edge, bits):
        sender, receiver = edge
        old_bits = self.edge_bits[edge]
        old_tx, old_rx = self._cost(edge, old_bits)
        new_tx, new_rx = self._cost(edge, bits)
        self.e_m[sender] += new_tx - old_tx
        if receiver != -1:
            self.e_m[receiver] += new_rx - old_rx
        if bits:
            self.edge_bits[edge] = bits
        else:
            self.edge_bits.pop(edge, None)

    def _cost(self, edge, bits):
        if not bits:
            return 0.0, 0.0
        sender, receiver = edge
        distance = (
            self.base_dists[sender]
            if receiver == -1 else self.dist_matrix[sender][receiver]
        )
        p = self.parameters
        charged_bits = bits + p.ctrl_bit
        tx = E_transmitting(
            p.e_elec, p.free_space_coeff, p.multipath_coeff,
            distance, p.d0, charged_bits,
        )
        rx = 0.0 if receiver == -1 else E_data_receiving(p.e_elec, charged_bits)
        return tx, rx


def payload_only_energy(route, node_count, dist_matrix, base_dists, parameters):
    """Lower-bound vector for one mandatory CH payload, excluding edge control bits."""

    values = [0.0] * node_count
    for sender, receiver in zip(route, route[1:]):
        distance = base_dists[sender] if receiver == -1 else dist_matrix[sender][receiver]
        p = parameters
        values[sender] += E_transmitting(
            p.e_elec, p.free_space_coeff, p.multipath_coeff,
            distance, p.d0, p.bit_count,
        )
        if receiver != -1:
            values[receiver] += E_data_receiving(p.e_elec, p.bit_count)
    return tuple(values)
