"""Caller-owned experiment inputs, provenance, and deterministic seed streams."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
import random

from ac_aco_mrp.types import ACACOParameters
from mrp.runner_types import PheromoneLifecycle, PureMRPParameters


class ExperimentMode(str, Enum):
    BASELINE = "BASELINE"
    PURE_MRP = "PURE_MRP"
    MRP_NETWORK_WIDE = "MRP_NETWORK_WIDE"
    HYBRID = "HYBRID"


class EvaluationView(str, Enum):
    NATIVE_ALGORITHM_SEMANTICS = "NATIVE_ALGORITHM_SEMANTICS"
    NORMALIZED_REPORTING = "NORMALIZED_REPORTING"


class StopCondition(str, Enum):
    MAX_ROUNDS = "MAX_ROUNDS"
    FND = "FND"
    HND = "HND"
    LND = "LND"
    NO_USABLE_SOURCE = "NO_USABLE_SOURCE"


@dataclass(frozen=True)
class ExperimentNetwork:
    """One physical trial map; callers provide all geometry-derived quantities."""

    map_id: str
    nodes: Sequence[object]
    dist_matrix: Sequence[Sequence[float]]
    base_dists: Sequence[float]
    initial_residual_e: tuple[float, ...]


@dataclass(frozen=True)
class ExperimentRoundScenario:
    """External event and hop-count input; this is not an RSS/event model."""

    event_nodes: tuple[int, ...]
    event_signal_strengths: Mapping[int, float]
    hop_counts: Mapping[int, int]
    scenario_id: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen trial settings; unresolved MRP values stay explicit in parameters."""

    max_rounds: int
    trial_count: int
    modes: tuple[ExperimentMode, ...]
    evaluation_view: EvaluationView
    stop_condition: StopCondition
    ac_aco_parameters: ACACOParameters
    mrp_parameters: PureMRPParameters
    mrp_pheromone_lifecycle: PheromoneLifecycle
    parameter_provenance: Mapping[str, str]
    implementation_version: str | None = None


@dataclass(frozen=True)
class TrialSeedStreams:
    trial_seed: int
    baseline_ac_aco_seed: int
    pure_mrp_seed: int
    hybrid_ac_aco_seed: int
    hybrid_mrp_seed: int


def derive_seed_streams(trial_seed: int) -> TrialSeedStreams:
    """Derive documented independent RNG streams from one caller-supplied seed."""

    if not isinstance(trial_seed, int) or isinstance(trial_seed, bool):
        raise ValueError("trial_seed must be an integer")
    source = random.Random(trial_seed)
    draws = tuple(source.randrange(2**63) for _ in range(4))
    return TrialSeedStreams(trial_seed, *draws)


def parameter_manifest(config: ExperimentConfig) -> dict[str, str]:
    """Return provenance for every serialized runner parameter without defaults."""

    explicit = dict(config.parameter_provenance)
    manifest = {
        "experiment.max_rounds": "USER_SELECTED_PARAMETER",
        "experiment.trial_count": "USER_SELECTED_PARAMETER",
        "experiment.stop_condition": "USER_SELECTED_PARAMETER",
        "experiment.evaluation_view": "USER_SELECTED_PARAMETER",
        "experiment.mrp_pheromone_lifecycle": "USER_SELECTED_SIMULATION_POLICY",
    }
    for field in fields(config.ac_aco_parameters):
        manifest[f"ac_aco.{field.name}"] = "EXISTING_SIMULATOR_PARAMETER"
    for field in fields(config.mrp_parameters):
        if field.name in {"charge_ant_energy", "ant_control_packet_bits"}:
            continue
        label = "EXISTING_SIMULATOR_PARAMETER" if field.name in {
            "communication_radius", "d0", "bit_count", "ctrl_bit", "e_elec", "e_agg",
            "free_space_coeff", "multipath_coeff",
        } else "USER_SELECTED_PARAMETER"
        manifest[f"mrp.{field.name}"] = label
    for field in fields(config.mrp_parameters.config):
        manifest[f"mrp.config.{field.name}"] = (
            "PAPER_ALGORITHM_RULE" if field.name == "aant_probability"
            else "PAPER_SIMULATION_PARAMETER"
        )
    manifest.update(explicit)
    return manifest
