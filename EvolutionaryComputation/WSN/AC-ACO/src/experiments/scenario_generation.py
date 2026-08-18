"""Deterministic, experiment-only event scenarios derived from ``map.pkl``."""

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
from types import MappingProxyType

from .config import ExperimentNetwork, ExperimentRoundScenario
from .hop_counts import HopCountSnapshot, derive_hop_counts


EVENT_RADIUS_METERS = 20.0
CALIBRATION_QUANTILES = (0.15, 0.40, 0.65, 0.90)
EVALUATION_QUANTILES = (0.25, 0.50, 0.75, 0.95)
MINIMUM_EVENT_SIZE = 3


class ScenarioGenerationError(ValueError):
    """Raised when the existing map cannot satisfy the approved scenario policy."""


@dataclass(frozen=True)
class FrozenEventScenario:
    scenario_set_id: str
    scenario_id: str
    event_anchor_sensor: int
    event_radius: float
    event_nodes: tuple[int, ...]
    event_signal_strengths: MappingProxyType
    anchor_base_distance_to_sink: float
    initial_hop_counts: MappingProxyType
    initial_unreachable_nodes: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "event_signal_strengths", MappingProxyType(dict(self.event_signal_strengths)))
        object.__setattr__(self, "initial_hop_counts", MappingProxyType(dict(self.initial_hop_counts)))

    def round_scenario(self) -> ExperimentRoundScenario:
        return ExperimentRoundScenario(
            self.event_nodes, dict(self.event_signal_strengths), dict(self.initial_hop_counts), self.scenario_id,
        )


@dataclass(frozen=True)
class FrozenScenarioSets:
    calibration: tuple[FrozenEventScenario, ...]
    evaluation: tuple[FrozenEventScenario, ...]
    eligible_region_count: int
    scenario_hashes: MappingProxyType

    def __post_init__(self):
        object.__setattr__(self, "scenario_hashes", MappingProxyType(dict(self.scenario_hashes)))

    @property
    def calibration_set_id(self) -> str:
        return _single_set_id(self.calibration, "calibration")

    @property
    def evaluation_set_id(self) -> str:
        return _single_set_id(self.evaluation, "evaluation")


@dataclass(frozen=True)
class ExistingMap:
    """Trusted repository map plus the generic simulator inputs it exposes."""

    network: ExperimentNetwork
    communication_radius: float
    sink_position: object


def load_existing_map_network(map_path: str | Path = "map.pkl") -> ExistingMap:
    """Load the repository's trusted map without importing its auto-running runner."""

    path = Path(map_path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {"base_pos", "nodes", "init_energy", "radius"}
    if not isinstance(payload, dict) or not required <= set(payload):
        raise ScenarioGenerationError("map.pkl lacks the required existing-map fields")
    nodes, base = tuple(payload["nodes"]), payload["base_pos"]
    if not nodes:
        raise ScenarioGenerationError("existing map has no sensors")
    try:
        dist_matrix = tuple(tuple(abs(left - right) for right in nodes) for left in nodes)
        base_dists = tuple(abs(base - node) for node in nodes)
    except Exception as error:
        raise ScenarioGenerationError("existing map coordinates cannot provide distance snapshots") from error
    _finite_nonnegative(dist_matrix, "sensor distance")
    _finite_nonnegative(base_dists, "Sink distance")
    initial_energy = payload["init_energy"]
    _positive_finite(initial_energy, "initial node energy")
    radius = payload["radius"]
    _positive_finite(radius, "communication radius")
    network = ExperimentNetwork(
        f"existing-map:{path.name}", nodes, dist_matrix, base_dists, (initial_energy,) * len(nodes),
    )
    return ExistingMap(network, float(radius), base)


def generate_frozen_scenario_sets(
    network: ExperimentNetwork, communication_radius: float,
) -> FrozenScenarioSets:
    """Freeze the approved 4+4 geometry-only event regions before calibration."""

    live = tuple(index for index, energy in enumerate(network.initial_residual_e) if energy > 0)
    regions = _eligible_regions(network, live, communication_radius)
    if len(regions) < 8:
        sizes = tuple(len(item[1]) for item in regions)
        raise ScenarioGenerationError(
            f"insufficient distinct 20m event regions: eligible={len(regions)}, sizes={sizes}"
        )
    ordered = tuple(sorted(regions, key=lambda item: (network.base_dists[item[0]], item[0])))
    used: set[tuple[int, ...]] = set()
    calibration = _select_set(network, ordered, CALIBRATION_QUANTILES, "calibration-v1", "c", used, communication_radius)
    evaluation = _select_set(network, ordered, EVALUATION_QUANTILES, "evaluation-v1", "e", used, communication_radius)
    hashes = {
        "calibration-v1": _scenario_hash(calibration),
        "evaluation-v1": _scenario_hash(evaluation),
    }
    return FrozenScenarioSets(calibration, evaluation, len(regions), hashes)


def scenario_manifest_payload(scenarios: FrozenScenarioSets) -> dict[str, object]:
    """Canonical, explicit provenance for the frozen scenario artifact."""

    return {
        "event_radius": EVENT_RADIUS_METERS,
        "event_radius_provenance": "PAPER_SIMULATION_PARAMETER",
        "event_center_policy": "USER_SELECTED_SCENARIO_MAPPING: sensor-anchored",
        "event_membership_provenance": "SIMULATOR_IMPLEMENTATION_MAPPING: dist_matrix[a][i] <= 20m",
        "signal_model": "NORMALIZED_DISTANCE_PROXY",
        "signal_provenance": "USER_SELECTED_SCENARIO_SIGNAL_PROXY",
        "signal_equation": "1 / (1 + d / r_event)",
        "signal_limitation": "Dimensionless monotonic event-strength proxy, not RSS/RSSI/dBm or an RF propagation model.",
        "deduplication": "exact sorted event_nodes tuple; lowest-ID anchor retained",
        "minimum_event_size": MINIMUM_EVENT_SIZE,
        "minimum_event_size_provenance": "USER_SELECTED_SCENARIO_VALIDITY_CONSTRAINT",
        "calibration_quantiles": CALIBRATION_QUANTILES,
        "evaluation_quantiles": EVALUATION_QUANTILES,
        "eligible_region_count": scenarios.eligible_region_count,
        "hop_count_provider": "SIMULATOR_IMPLEMENTATION_MAPPING: BFS on current live communication graph; direct Sink = 1 hop",
        "scenario_hashes": dict(scenarios.scenario_hashes),
        "scenario_sets": {
            "calibration-v1": [_scenario_payload(item) for item in scenarios.calibration],
            "evaluation-v1": [_scenario_payload(item) for item in scenarios.evaluation],
        },
    }


def load_frozen_scenario_sets(path: str | Path) -> FrozenScenarioSets:
    """Load the pre-calibration scenario artifact and reject any hash/content mismatch."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        sets = payload["scenario_sets"]
        calibration = _frozen_scenarios_from_payload(sets["calibration-v1"])
        evaluation = _frozen_scenarios_from_payload(sets["evaluation-v1"])
        hashes = {str(key): str(value) for key, value in payload["scenario_hashes"].items()}
        eligible_region_count = payload["eligible_region_count"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ScenarioGenerationError("frozen scenario manifest is invalid") from error
    if not isinstance(eligible_region_count, int) or eligible_region_count < 8:
        raise ScenarioGenerationError("frozen scenario manifest has an invalid eligible-region count")
    result = FrozenScenarioSets(calibration, evaluation, eligible_region_count, hashes)
    return validate_frozen_scenario_sets(result)


def _eligible_regions(network, live, communication_radius):
    representatives: dict[tuple[int, ...], int] = {}
    for anchor in live:
        event_nodes = tuple(index for index in live if network.dist_matrix[anchor][index] <= EVENT_RADIUS_METERS)
        if len(event_nodes) >= MINIMUM_EVENT_SIZE:
            representatives.setdefault(event_nodes, anchor)
    return tuple((anchor, event_nodes) for event_nodes, anchor in representatives.items())


def _select_set(network, ordered, quantiles, set_id, prefix, used, communication_radius):
    selected = []
    for number, quantile in enumerate(quantiles, 1):
        target = math.floor(quantile * (len(ordered) - 1) + 0.5)
        candidates = sorted(range(len(ordered)), key=lambda index: (abs(index - target), index))
        anchor, event_nodes = next((ordered[index] for index in candidates if ordered[index][1] not in used), (None, None))
        if anchor is None:
            raise ScenarioGenerationError("cannot select distinct geometry-only event regions")
        used.add(event_nodes)
        snapshot = derive_hop_counts(network, range(len(network.nodes)), communication_radius)
        selected.append(FrozenEventScenario(
            set_id, f"{set_id}-{prefix}{number}", anchor, EVENT_RADIUS_METERS, event_nodes,
            _signal_proxy(network, anchor, event_nodes), network.base_dists[anchor],
            snapshot.hop_counts, snapshot.unreachable_nodes,
        ))
    return tuple(selected)


def _signal_proxy(network, anchor, event_nodes):
    signals = {node_id: 1.0 / (1.0 + network.dist_matrix[anchor][node_id] / EVENT_RADIUS_METERS) for node_id in event_nodes}
    if any(not math.isfinite(value) or value <= 0 for value in signals.values()):
        raise ScenarioGenerationError("normalized event-strength proxy must be finite and positive")
    return signals


def _frozen_scenarios_from_payload(items: object) -> tuple[FrozenEventScenario, ...]:
    if not isinstance(items, list):
        raise ScenarioGenerationError("frozen scenario set must be a list")
    result = []
    for item in items:
        if not isinstance(item, dict):
            raise ScenarioGenerationError("frozen scenario entry must be an object")
        signals = item.get("event_signal_strengths")
        hops = item.get("initial_hop_counts")
        if not isinstance(signals, dict) or not isinstance(hops, dict):
            raise ScenarioGenerationError("frozen scenario lacks signal or hop-count data")
        result.append(FrozenEventScenario(
            str(item["scenario_set_id"]), str(item["scenario_id"]), int(item["event_anchor_sensor"]),
            float(item["event_radius"]), tuple(int(node) for node in item["event_nodes"]),
            {int(node): float(value) for node, value in signals.items()},
            float(item["anchor_base_distance_to_sink"]), {int(node): int(value) for node, value in hops.items()},
            tuple(int(node) for node in item["initial_unreachable_nodes"]),
        ))
    return tuple(result)


def _validate_frozen_sets(calibration, evaluation):
    if len(calibration) != 4 or len(evaluation) != 4:
        raise ScenarioGenerationError("frozen manifest must contain exactly four calibration and four evaluation scenarios")
    all_items = calibration + evaluation
    event_sets = [item.event_nodes for item in all_items]
    if len(set(event_sets)) != 8:
        raise ScenarioGenerationError("frozen manifest repeats an exact event-node set")
    for item in all_items:
        if item.event_radius != EVENT_RADIUS_METERS or len(item.event_nodes) < MINIMUM_EVENT_SIZE:
            raise ScenarioGenerationError("frozen manifest violates the approved event policy")
        if set(item.event_signal_strengths) != set(item.event_nodes):
            raise ScenarioGenerationError("frozen manifest signal proxy must cover exactly the event nodes")
        if item.event_anchor_sensor not in item.event_nodes or item.event_signal_strengths.get(item.event_anchor_sensor) != 1.0:
            raise ScenarioGenerationError("frozen manifest has an invalid anchor signal")
        if any(not math.isfinite(value) or value <= 0 for value in item.event_signal_strengths.values()):
            raise ScenarioGenerationError("frozen manifest has an invalid signal proxy")


def _scenario_hash(items: Iterable[FrozenEventScenario]) -> str:
    payload = json.dumps([_scenario_payload(item) for item in items], sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_frozen_scenario_sets(scenarios: FrozenScenarioSets) -> FrozenScenarioSets:
    """Verify immutable scenario contents still match their declared set hashes."""

    if not isinstance(scenarios, FrozenScenarioSets):
        raise ScenarioGenerationError("frozen scenario sets are required")
    if scenarios.calibration_set_id != "calibration-v1" or scenarios.evaluation_set_id != "evaluation-v1":
        raise ScenarioGenerationError("frozen scenario sets must retain calibration-v1 and evaluation-v1 identities")
    _validate_frozen_sets(scenarios.calibration, scenarios.evaluation)
    expected = {
        "calibration-v1": _scenario_hash(scenarios.calibration),
        "evaluation-v1": _scenario_hash(scenarios.evaluation),
    }
    if dict(scenarios.scenario_hashes) != expected:
        raise ScenarioGenerationError("frozen scenario set hash mismatch")
    return scenarios


def validate_scenarios_for_existing_map(scenarios: FrozenScenarioSets, existing_map: ExistingMap) -> FrozenScenarioSets:
    """Bind frozen scenario content to the existing physical map before calibration."""

    validate_frozen_scenario_sets(scenarios)
    network = existing_map.network
    valid_ids = set(range(len(network.nodes)))
    for item in scenarios.calibration + scenarios.evaluation:
        if item.event_anchor_sensor not in valid_ids or set(item.event_nodes) - valid_ids:
            raise ScenarioGenerationError("frozen scenario contains an invalid physical sensor ID")
        expected_nodes = tuple(
            sensor_id for sensor_id in range(len(network.nodes))
            if network.dist_matrix[item.event_anchor_sensor][sensor_id] <= EVENT_RADIUS_METERS
        )
        if item.event_nodes != expected_nodes:
            raise ScenarioGenerationError("frozen scenario event membership does not match the existing map at 20m")
        expected_signals = {
            sensor_id: 1.0 / (1.0 + network.dist_matrix[item.event_anchor_sensor][sensor_id] / EVENT_RADIUS_METERS)
            for sensor_id in item.event_nodes
        }
        if dict(item.event_signal_strengths) != expected_signals:
            raise ScenarioGenerationError("frozen scenario signal proxy does not match the existing map")
        if item.anchor_base_distance_to_sink != network.base_dists[item.event_anchor_sensor]:
            raise ScenarioGenerationError("frozen scenario anchor Sink distance does not match the existing map")
    return scenarios


def _scenario_payload(item: FrozenEventScenario) -> dict[str, object]:
    return {
        "scenario_set_id": item.scenario_set_id,
        "scenario_id": item.scenario_id,
        "event_anchor_sensor": item.event_anchor_sensor,
        "event_radius": item.event_radius,
        "event_nodes": item.event_nodes,
        "event_signal_strengths": dict(item.event_signal_strengths),
        "anchor_base_distance_to_sink": item.anchor_base_distance_to_sink,
        "initial_hop_counts": dict(item.initial_hop_counts),
        "initial_unreachable_nodes": item.initial_unreachable_nodes,
    }


def _single_set_id(items: tuple[FrozenEventScenario, ...], label: str) -> str:
    ids = {item.scenario_set_id for item in items}
    if len(ids) != 1:
        raise ScenarioGenerationError(f"frozen {label} scenarios have inconsistent set IDs")
    return next(iter(ids))


def _finite_nonnegative(values, label):
    for item in values:
        if isinstance(item, (tuple, list)):
            _finite_nonnegative(item, label)
        elif isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
            raise ScenarioGenerationError(f"{label} must be finite and non-negative")


def _positive_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ScenarioGenerationError(f"{label} must be finite and positive")
