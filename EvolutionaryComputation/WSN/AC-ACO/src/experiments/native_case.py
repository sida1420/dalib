"""Operational identity/context for the repository's native AC-ACO map.

This module does not generate a scenario.  It adapts the existing experiment
interface with an event-free hop-count context for network-wide routing modes.
"""

from dataclasses import dataclass
import hashlib
import json
import math

from .config import ExperimentRoundScenario
from .hop_counts import derive_hop_counts


NATIVE_CASE_ID = "ORIGINAL_AC_ACO_NATIVE_CASE"
NATIVE_CASE_SOURCE = "map.pkl + legacy src/AC_ACO.py/src/map_gen.py configuration"


@dataclass(frozen=True)
class NativeCaseIdentity:
    canonical_sha256: str
    node_count: int
    sink: tuple[float, float]
    initial_energy: float
    communication_radius: float
    ch_proportion: float
    target_cluster_head_count: int


def native_case_identity(existing_map, config) -> NativeCaseIdentity:
    """Hash the initial physical state and common physical/workload settings."""

    network = existing_map.network
    common = config.mrp_parameters
    ac_aco = config.ac_aco_parameters
    sensors = [
        {
            "id": sensor_id,
            "x": float(node.x),
            "y": float(node.y),
            "initial_residual_energy": float(network.initial_residual_e[sensor_id]),
        }
        for sensor_id, node in enumerate(network.nodes)
    ]
    payload = {
        "node_count": len(network.nodes),
        "sensors": sensors,
        "sink": {"x": float(existing_map.sink_position.x), "y": float(existing_map.sink_position.y)},
        "communication_radius": float(existing_map.communication_radius),
        "ch_proportion": float(ac_aco.ch_proportion),
        "bit_count": float(common.bit_count),
        "ctrl_bit": float(common.ctrl_bit),
        "e_elec": float(common.e_elec),
        "e_agg": float(common.e_agg),
        "free_space_coeff": float(common.free_space_coeff),
        "multipath_coeff": float(common.multipath_coeff),
        "d0": float(common.d0),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    initial_values = set(network.initial_residual_e)
    if len(initial_values) != 1:
        raise ValueError("native AC-ACO identity expects one common initial-energy value")
    return NativeCaseIdentity(
        hashlib.sha256(canonical).hexdigest(), len(network.nodes),
        (float(existing_map.sink_position.x), float(existing_map.sink_position.y)),
        float(next(iter(initial_values))), float(existing_map.communication_radius),
        float(ac_aco.ch_proportion), math.ceil(ac_aco.ch_proportion * len(network.nodes)),
    )


def build_native_round_context(existing_map, communication_radius: float) -> ExperimentRoundScenario:
    """Build an event-free routing context from the unchanged native map."""

    network = existing_map.network
    live = tuple(index for index, energy in enumerate(network.initial_residual_e) if energy > 0)
    hops = derive_hop_counts(network, live, communication_radius)
    return ExperimentRoundScenario((), {}, hops.hop_counts, NATIVE_CASE_ID)
