from dataclasses import replace
from pathlib import Path
import inspect
import json
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments import (
    EVENT_RADIUS_METERS,
    ExperimentNetwork,
    ExperimentRoundScenario,
    derive_hop_counts,
    generate_frozen_scenario_sets,
    load_frozen_scenario_sets,
    load_existing_map_network,
    scenario_manifest_payload,
    scenario_with_current_hops,
    validate_frozen_scenario_sets,
)
from experiments import real_calibration, scenario_generation
from point import Point


def _network():
    nodes = (Point(0, 0), Point(10, 0), Point(20, 0), Point(30, 0))
    matrix = tuple(tuple(abs(left - right) for right in nodes) for left in nodes)
    return ExperimentNetwork("event-fixture", nodes, matrix, (30.0, 20.0, 10.0, 0.0), (1.0,) * 4)


def test_hop_counts_match_direct_sink_terminal_ttl_convention_and_live_updates():
    network = _network()
    first = derive_hop_counts(network, (0, 1, 2, 3), 11.0)
    assert first.hop_counts == {2: 1, 3: 1, 1: 2, 0: 3}
    assert not first.unreachable_nodes
    scenario = ExperimentRoundScenario((0, 1, 2), {0: 1.0, 1: 0.75, 2: 0.5}, first.hop_counts)
    refreshed, snapshot = scenario_with_current_hops(scenario, network, (0, 1, 3), 11.0)
    assert refreshed.event_nodes == scenario.event_nodes
    assert refreshed.event_signal_strengths == scenario.event_signal_strengths
    assert snapshot.unreachable_nodes == (0, 1)
    assert 2 not in refreshed.hop_counts and 2 not in snapshot.hop_counts


def test_existing_map_scenarios_are_20m_sensor_anchored_unique_and_deterministic():
    loaded = load_existing_map_network()
    one = generate_frozen_scenario_sets(loaded.network, loaded.communication_radius)
    two = generate_frozen_scenario_sets(loaded.network, loaded.communication_radius)
    combined = one.calibration + one.evaluation
    assert one == two and one.eligible_region_count >= 8
    assert len(combined) == 8 and len({item.event_nodes for item in combined}) == 8
    for item in combined:
        assert item.event_radius == EVENT_RADIUS_METERS and len(item.event_nodes) >= 3
        assert item.event_anchor_sensor in item.event_nodes
        assert item.event_signal_strengths[item.event_anchor_sensor] == pytest.approx(1.0)
        for node_id in item.event_nodes:
            distance = loaded.network.dist_matrix[item.event_anchor_sensor][node_id]
            assert distance <= EVENT_RADIUS_METERS
            assert item.event_signal_strengths[node_id] == pytest.approx(1 / (1 + distance / EVENT_RADIUS_METERS))
    payload = scenario_manifest_payload(one)
    assert payload["signal_model"] == "NORMALIZED_DISTANCE_PROXY"
    assert payload["scenario_hashes"] == one.scenario_hashes


def test_frozen_manifest_round_trips_with_its_content_hash(tmp_path):
    loaded = load_existing_map_network()
    generated = generate_frozen_scenario_sets(loaded.network, loaded.communication_radius)
    manifest = tmp_path / "scenario_manifest.json"
    manifest.write_text(json.dumps(scenario_manifest_payload(generated), sort_keys=True), encoding="utf-8")

    assert load_frozen_scenario_sets(manifest) == generated


def test_frozen_scenarios_reject_mapping_mutation_and_hash_tampering():
    loaded = load_existing_map_network()
    generated = generate_frozen_scenario_sets(loaded.network, loaded.communication_radius)
    with pytest.raises(TypeError):
        generated.calibration[0].event_signal_strengths[generated.calibration[0].event_anchor_sensor] = 0.5
    with pytest.raises(TypeError):
        generated.scenario_hashes["calibration-v1"] = "tampered"

    object.__setattr__(generated, "scenario_hashes", {"calibration-v1": "tampered"})
    with pytest.raises(scenario_generation.ScenarioGenerationError, match="hash mismatch"):
        validate_frozen_scenario_sets(generated)


def test_scenario_manifest_publication_validates_and_exposes_consistent_set_ids(tmp_path):
    loaded = load_existing_map_network()
    generated = generate_frozen_scenario_sets(loaded.network, loaded.communication_radius)

    directory = real_calibration.freeze_scenario_manifest(generated, tmp_path / "real-calibration")

    assert directory.name == "real-calibration"
    assert generated.calibration_set_id == "calibration-v1"
    assert generated.evaluation_set_id == "evaluation-v1"
    object.__setattr__(generated, "scenario_hashes", {"calibration-v1": "bad"})
    with pytest.raises(scenario_generation.ScenarioGenerationError, match="hash mismatch"):
        real_calibration.freeze_scenario_manifest(generated, tmp_path / "invalid")


def test_frozen_scenario_manifest_rejects_wrong_or_mixed_set_ids_even_with_matching_hashes():
    loaded = load_existing_map_network()
    generated = generate_frozen_scenario_sets(loaded.network, loaded.communication_radius)
    relabelled = tuple(replace(item, scenario_set_id="wrong-set") for item in generated.calibration)
    wrong = scenario_generation.FrozenScenarioSets(
        relabelled, generated.evaluation, generated.eligible_region_count,
        {
            "calibration-v1": scenario_generation._scenario_hash(relabelled),
            "evaluation-v1": scenario_generation._scenario_hash(generated.evaluation),
        },
    )
    with pytest.raises(scenario_generation.ScenarioGenerationError, match="identities"):
        validate_frozen_scenario_sets(wrong)

    mixed = scenario_generation.FrozenScenarioSets(
        (generated.calibration[0], replace(generated.calibration[1], scenario_set_id="other")) + generated.calibration[2:],
        generated.evaluation, generated.eligible_region_count, generated.scenario_hashes,
    )
    with pytest.raises(scenario_generation.ScenarioGenerationError, match="inconsistent"):
        validate_frozen_scenario_sets(mixed)


def test_signal_proxy_has_required_boundary_and_monotonic_behavior_without_rss_model():
    network = _network()
    signals = scenario_generation._signal_proxy(network, 0, (0, 1, 2))
    assert signals == pytest.approx({0: 1.0, 1: 1 / 1.5, 2: 0.5})
    source = inspect.getsource(scenario_generation)
    assert "NORMALIZED_DISTANCE_PROXY" in source
    assert "random" not in source and "threshold" not in source


def test_real_calibration_source_never_calls_baseline_or_hybrid_runners():
    source = inspect.getsource(real_calibration)
    forbidden = ("run_baseline_round", "run_hybrid_round", "AC_ACO", "network_config", "energy_consumption")
    assert all(name not in source for name in forbidden)
