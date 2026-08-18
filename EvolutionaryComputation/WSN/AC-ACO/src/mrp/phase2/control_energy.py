"""Optional physical radio ledger for MRP Phase-II control ants."""

from dataclasses import dataclass
import math

from evaluate import E_data_receiving, E_transmitting


@dataclass(frozen=True)
class AntControlEnergy:
    """Sensor-side radio operations and energy caused by control ants."""

    e_m_list: tuple[float, ...]
    sant_tx_count: int = 0
    sant_rx_count: int = 0
    bant_tx_count: int = 0
    bant_rx_count: int = 0
    aant_tx_count: int = 0
    aant_rx_count: int = 0
    sant_energy: float = 0.0
    bant_energy: float = 0.0
    aant_energy: float = 0.0

    @property
    def total_energy(self) -> float:
        return math.fsum((self.sant_energy, self.bant_energy, self.aant_energy))

    @property
    def is_zero(self) -> bool:
        return self.total_energy == 0.0


class AntControlEnergyLedger:
    """Mutable per-discovery ledger; snapshots returned from Phase II are frozen."""

    _KINDS = ("sant", "bant", "aant")

    def __init__(
        self, node_count, packet_bits, dist_matrix, base_dists,
        e_elec, free_space_coeff, multipath_coeff, d0,
    ):
        _validate_inputs(node_count, packet_bits, dist_matrix, base_dists)
        self._energy = [0.0] * node_count
        self._bits = packet_bits
        self._dist_matrix = dist_matrix
        self._base_dists = base_dists
        self._radio = (e_elec, free_space_coeff, multipath_coeff, d0)
        self._tx = {kind: 0 for kind in self._KINDS}
        self._rx = {kind: 0 for kind in self._KINDS}
        self._energy_by_kind = {kind: 0.0 for kind in self._KINDS}

    def record_sant_hop(self, mode: str, sender: int, receiver: int) -> None:
        self.record_hop("aant" if mode == "AANT" else "sant", sender, receiver)

    def record_bant_route(self, reverse_path) -> None:
        for sender, receiver in zip(reverse_path, reverse_path[1:]):
            self.record_hop("bant", sender, receiver)

    def record_hop(self, kind: str, sender: int, receiver: int) -> None:
        if kind not in self._KINDS:
            raise ValueError(f"unknown control-ant kind: {kind}")
        hop_energy = 0.0
        if sender != -1:
            distance = (
                self._base_dists[sender]
                if receiver == -1 else self._dist_matrix[sender][receiver]
            )
            tx = E_transmitting(*self._radio[:3], distance, self._radio[3], self._bits)
            self._energy[sender] += tx
            self._tx[kind] += 1
            hop_energy += tx
        if receiver != -1:
            rx = E_data_receiving(self._radio[0], self._bits)
            self._energy[receiver] += rx
            self._rx[kind] += 1
            hop_energy += rx
        self._energy_by_kind[kind] += hop_energy

    def snapshot(self) -> AntControlEnergy:
        return AntControlEnergy(
            tuple(self._energy),
            self._tx["sant"], self._rx["sant"],
            self._tx["bant"], self._rx["bant"],
            self._tx["aant"], self._rx["aant"],
            self._energy_by_kind["sant"],
            self._energy_by_kind["bant"],
            self._energy_by_kind["aant"],
        )


def zero_ant_control_energy(node_count: int) -> AntControlEnergy:
    return AntControlEnergy((0.0,) * node_count)


def validate_ant_energy_parameters(enabled, packet_bits) -> None:
    if not isinstance(enabled, bool):
        raise ValueError("charge_ant_energy must be a boolean")
    if packet_bits is not None and (
        not isinstance(packet_bits, int) or isinstance(packet_bits, bool)
        or packet_bits <= 0
    ):
        raise ValueError("ant_control_packet_bits must be None or a positive integer")
    if enabled and packet_bits is None:
        raise ValueError(
            "ant_control_packet_bits is required when charge_ant_energy is enabled"
        )


def combine_ant_control_energy(*items: AntControlEnergy) -> AntControlEnergy:
    if not items:
        return zero_ant_control_energy(0)
    node_count = len(items[0].e_m_list)
    if any(len(item.e_m_list) != node_count for item in items):
        raise ValueError("control-energy ledgers must have the same node count")
    vectors = zip(*(item.e_m_list for item in items))
    return AntControlEnergy(
        tuple(math.fsum(values) for values in vectors),
        sum(item.sant_tx_count for item in items),
        sum(item.sant_rx_count for item in items),
        sum(item.bant_tx_count for item in items),
        sum(item.bant_rx_count for item in items),
        sum(item.aant_tx_count for item in items),
        sum(item.aant_rx_count for item in items),
        math.fsum(item.sant_energy for item in items),
        math.fsum(item.bant_energy for item in items),
        math.fsum(item.aant_energy for item in items),
    )


def _validate_inputs(node_count, packet_bits, dist_matrix, base_dists) -> None:
    if not isinstance(node_count, int) or isinstance(node_count, bool) or node_count <= 0:
        raise ValueError("control-energy node count must be positive")
    if (
        isinstance(packet_bits, bool) or not isinstance(packet_bits, (int, float))
        or not math.isfinite(packet_bits) or packet_bits <= 0
    ):
        raise ValueError("ant_control_packet_bits must be finite and positive")
    if len(base_dists) != node_count or len(dist_matrix) != node_count:
        raise ValueError("control-energy geometry must match node count")
