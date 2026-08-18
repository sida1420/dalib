"""Approved real-9B calibration orchestration; Pure MRP only, never evaluation."""

from dataclasses import asdict, dataclass, replace
from functools import partial
import hashlib
import json
import math
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

from ac_aco_mrp import ACACOParameters
from mrp import HeuristicBounds, MRPConfig, PheromoneLifecycle, PureMRPParameters

from .calibration import (
    SELECTION_RULE,
    _rank,
    _run_mrp_calibration_candidate_subset,
    run_mrp_calibration_candidates,
)
from .calibration_characterization import CalibrationCharacterization, characterize_calibration_space
from .calibration_audit import audit_calibration_parameters, parameter_snapshot
from .calibration_config import candidate_configuration, candidate_from_overrides
from .calibration_freeze import freeze_selected_configuration
from .calibration_io import calibration_output_files
from .calibration_types import CalibrationPlan, CalibrationResult
from .config import EvaluationView, ExperimentConfig, ExperimentMode, StopCondition
from .hop_counts import scenario_with_current_hops
from .io import csv_text, json_text, publish_output_files
from .scenario_generation import (
    ExistingMap,
    FrozenScenarioSets,
    load_frozen_scenario_sets,
    scenario_manifest_payload,
    validate_frozen_scenario_sets,
    validate_scenarios_for_existing_map,
)


CALIBRATION_SEEDS = tuple(range(11001, 11011))
EVALUATION_SEEDS = tuple(range(22001, 22031))
CALIBRATION_STATUS = "SCIENTIFIC_CALIBRATION_COMPLETE"
CALIBRATION_CANDIDATE_WORKERS = 4


class RealCalibrationError(ValueError):
    """Raised when a user-approved real calibration cannot complete honestly."""


@dataclass(frozen=True)
class RealCalibrationRun:
    configuration: ExperimentConfig
    scenarios: FrozenScenarioSets
    characterization: CalibrationCharacterization
    stage_a: CalibrationResult
    stage_b: CalibrationResult
    selected_config_hash: str


def primary_calibration_config(existing_map: ExistingMap, max_rounds: int = 10) -> ExperimentConfig:
    """Create the documented existing-simulator/Paper-fixed calibration baseline."""

    energy = existing_map.network.initial_residual_e[0]
    free_space, multipath = 10e-12, 0.0013e-12
    mrp = PureMRPParameters(
        existing_map.communication_radius, HeuristicBounds(0.0, 1.0, 0.0, 1.0), MRPConfig(),
        0.0, 1, 1, 1.0, 0.0, 0.0, math.sqrt(free_space / multipath), 2000.0, 100.0,
        50e-9, 5e-9, free_space, multipath,
    )
    ac_aco = ACACOParameters(existing_map.sink_position, energy * len(existing_map.network.nodes))
    return ExperimentConfig(
        max_rounds, len(CALIBRATION_SEEDS), (ExperimentMode.PURE_MRP,),
        EvaluationView.NORMALIZED_REPORTING, StopCondition.MAX_ROUNDS, ac_aco, mrp,
        PheromoneLifecycle.RESET_PER_DISCOVERY,
        {
            "mrp.c0": "USER_SELECTED_PARAMETER: NORMALIZATION_CONVENTION",
            "experiment.mrp_pheromone_lifecycle": "USER_SELECTED_SIMULATION_POLICY",
            "experiment.max_rounds": "USER_SELECTED_CALIBRATION_BUDGET",
        },
        "checkpoint-9b-real-calibration",
    )


def freeze_scenario_manifest(scenarios: FrozenScenarioSets, output_directory: str | Path) -> Path:
    """Publish only the immutable scenario artifact before candidate execution."""

    validate_frozen_scenario_sets(scenarios)
    return publish_output_files(output_directory, {"scenario_manifest.json": json_text(scenario_manifest_payload(scenarios))})


def run_real_calibration(existing_map: ExistingMap, scenarios: FrozenScenarioSets) -> RealCalibrationRun:
    """Perform the approved Stage A then Stage B search without evaluation inputs."""

    validate_scenarios_for_existing_map(scenarios, existing_map)
    config = primary_calibration_config(existing_map)
    calibration = tuple(item.round_scenario() for item in scenarios.calibration)
    evaluation = tuple(item.round_scenario() for item in scenarios.evaluation)
    characterization = characterize_calibration_space(existing_map.network, config, scenarios.calibration, CALIBRATION_SEEDS)
    stage_base = _stage_base_config(config, characterization)
    provider = _current_hop_provider(stage_base.mrp_parameters.communication_radius)
    stage_a = _run_stage_a(existing_map, stage_base, calibration, evaluation, provider, characterization)
    top_pairs = _top_pairs(stage_a)
    if not top_pairs:
        raise RealCalibrationError("NO VALID BANT FEEDBACK CONFIGURATION")
    stage_b = _run_stage_b(existing_map, stage_base, calibration, evaluation, provider, characterization, top_pairs)
    if stage_b.selected_config_id is None:
        raise RealCalibrationError("no valid Stage-B calibration configuration")
    selected = next(item for item in stage_b.candidates if item.rank == 1)
    digest = _selected_hash(stage_b, stage_base)
    return RealCalibrationRun(stage_base, scenarios, characterization, stage_a, stage_b, digest)


def write_real_calibration_outputs(
    run: RealCalibrationRun, output_directory: str | Path, *,
    metadata_updates: dict[str, object] | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Publish results beneath an already frozen scenario directory without overwriting it."""

    directory = Path(output_directory)
    if not (directory / "scenario_manifest.json").is_file():
        raise RealCalibrationError("real calibration output requires a frozen scenario_manifest.json")
    validate_frozen_scenario_sets(run.scenarios)
    manifest_scenarios = load_frozen_scenario_sets(directory / "scenario_manifest.json")
    if scenario_manifest_payload(manifest_scenarios) != scenario_manifest_payload(run.scenarios):
        raise RealCalibrationError("published scenario manifest does not match the calibration scenario sets")
    result_directory = directory / "calibration_outputs"
    metadata = {
        "calibration_status": CALIBRATION_STATUS,
        "calibration_scenario_set_id": run.scenarios.calibration_set_id,
        "reserved_evaluation_scenario_set_id": run.scenarios.evaluation_set_id,
        "calibration_scenario_ids": [item.scenario_id for item in run.scenarios.calibration],
        "reserved_evaluation_scenario_ids": [item.scenario_id for item in run.scenarios.evaluation],
        "scenario_hashes": dict(run.scenarios.scenario_hashes),
        "calibration_seed_set": CALIBRATION_SEEDS,
        "reserved_evaluation_seed_set": EVALUATION_SEEDS,
        "evaluation_seeds_executed": False,
        "evaluation_scenarios_executed": False,
        "candidate_execution_workers": CALIBRATION_CANDIDATE_WORKERS,
        "characterization": asdict(run.characterization),
        "c0_policy": "1.0; USER_SELECTED_PARAMETER; NORMALIZATION_CONVENTION",
        "lifecycle_policy": "RESET_PER_DISCOVERY; USER_SELECTED_SIMULATION_POLICY",
        "stage_a_ranked_summaries": [_summary_payload(item) for item in run.stage_a.candidates],
        "stage_b_ranked_summaries": [_summary_payload(item) for item in run.stage_b.candidates],
        "selected_config_hash": run.selected_config_hash,
    }
    metadata.update(metadata_updates or {})
    frozen = freeze_selected_configuration(
        run.stage_b, run.configuration, calibration_status=CALIBRATION_STATUS,
        calibration_metadata=metadata,
    )
    files = calibration_output_files(
        run.stage_b, run.configuration, frozen,
    )
    files["stage_a_summary.csv"] = csv_text(_summary_payload(item) for item in run.stage_a.candidates)
    files["real_calibration_metadata.json"] = json_text(metadata)
    files.update(extra_files or {})
    return publish_output_files(result_directory, files)


def _stage_base_config(config, characterization):
    parameters = replace(
        config.mrp_parameters, heuristic_bounds=characterization.medium_bounds,
        lambda_coefficient=config.mrp_parameters.config.initial_pheromone / characterization.b_ref,
        ttl=2 * characterization.hop_scale_h, num_sants=characterization.network_scale_s,
        c0=1.0, c=0.0, c1=0.0,
    )
    return replace(config, mrp_parameters=parameters, mrp_pheromone_lifecycle=PheromoneLifecycle.RESET_PER_DISCOVERY)


def _run_stage_a(existing_map, config, calibration, evaluation, provider, characterization):
    c_values = (0.005, 0.010, 0.020, 0.040)
    f_ref = characterization.f_ref
    c1_values = (0.0, 0.25 * 0.01 / f_ref, 0.50 * 0.01 / f_ref, 1.00 * 0.01 / f_ref)
    candidates = tuple(candidate_from_overrides((("mrp.c", c), ("mrp.c1", c1), ("mrp_pheromone_lifecycle", PheromoneLifecycle.RESET_PER_DISCOVERY))) for c in c_values for c1 in c1_values)
    plan = _plan(
        calibration, evaluation,
        {"mrp.c": c_values, "mrp.c1": c1_values, "mrp_pheromone_lifecycle": (PheromoneLifecycle.RESET_PER_DISCOVERY,)},
        candidates,
    )
    return _run_candidates_with_file_workers(existing_map.network, config, plan, candidates, provider)


def _run_stage_b(existing_map, config, calibration, evaluation, provider, characterization, pairs):
    tau0 = config.mrp_parameters.config.initial_pheromone
    lambdas = (0.25 * tau0 / characterization.b_ref, tau0 / characterization.b_ref, 4.0 * tau0 / characterization.b_ref)
    profiles = (
        ("WIDE", HeuristicBounds(characterization.mu_quantiles["q05"], characterization.mu_quantiles["q95"], characterization.mu_quantiles["q05"], characterization.mu_quantiles["q95"])),
        ("MEDIUM", characterization.medium_bounds),
        ("STRONG_CLIP", HeuristicBounds(characterization.mu_quantiles["q25"], characterization.mu_quantiles["q75"], characterization.mu_quantiles["q25"], characterization.mu_quantiles["q75"])),
    )
    h, s = characterization.hop_scale_h, characterization.network_scale_s
    ttls = tuple(dict.fromkeys((h, math.ceil(1.5 * h), 2 * h)))
    ants = (s, 2 * s, 4 * s)
    space = {
        "mrp.c": tuple(pair[0] for pair in pairs), "mrp.c1": tuple(pair[1] for pair in pairs),
        "mrp.lambda_coefficient": lambdas, "mrp.ttl": ttls, "mrp.num_sants": ants,
        "mrp.heuristic_bounds.mu_min": tuple(profile[1].mu_min for profile in profiles),
        "mrp.heuristic_bounds.mu_max": tuple(profile[1].mu_max for profile in profiles),
        "mrp.heuristic_bounds.eta_min": tuple(profile[1].eta_min for profile in profiles),
        "mrp.heuristic_bounds.eta_max": tuple(profile[1].eta_max for profile in profiles),
        "mrp_pheromone_lifecycle": (PheromoneLifecycle.RESET_PER_DISCOVERY,),
    }
    candidates = []
    for c, c1 in pairs:
        for lambda_value in lambdas:
            for _, bounds in profiles:
                for ant_count in ants:
                    for ttl in ttls:
                        candidates.append(candidate_from_overrides((
                            ("mrp.c", c), ("mrp.c1", c1), ("mrp.lambda_coefficient", lambda_value),
                            ("mrp.ttl", ttl), ("mrp.num_sants", ant_count),
                            ("mrp.heuristic_bounds.mu_min", bounds.mu_min), ("mrp.heuristic_bounds.mu_max", bounds.mu_max),
                            ("mrp.heuristic_bounds.eta_min", bounds.eta_min), ("mrp.heuristic_bounds.eta_max", bounds.eta_max),
                            ("mrp_pheromone_lifecycle", PheromoneLifecycle.RESET_PER_DISCOVERY),
                        )))
    plan = _plan(calibration, evaluation, space, tuple(candidates))
    return _run_candidates_with_file_workers(existing_map.network, config, plan, tuple(candidates), provider)


def _plan(calibration, evaluation, search_space, candidates=None):
    return CalibrationPlan(
        CALIBRATION_SEEDS, EVALUATION_SEEDS, "calibration-v1", "evaluation-v1",
        tuple(calibration), tuple(evaluation), search_space,
        {path: ("USER_SELECTED_SIMULATION_POLICY" if path == "mrp_pheromone_lifecycle" else "USER_SELECTED_CALIBRATION_RANGE") for path in search_space},
        None,
        None if candidates is None else tuple(candidate.overrides for candidate in candidates),
    )


def _current_hop_provider(communication_radius):
    return partial(_refresh_scenario_hops, communication_radius=communication_radius)


def _refresh_scenario_hops(scenario, state, network, communication_radius):
    physical = state.physical_state if hasattr(state, "physical_state") else state
    return scenario_with_current_hops(scenario, network, physical.live_nodes, communication_radius)[0]


def _run_candidates_with_file_workers(network, config, plan, candidates, provider):
    """Run isolated candidate chunks in bounded OS processes without shared state.

    Windows sandbox policy denies the named pipes used by ``ProcessPoolExecutor``.
    This small file-backed transport keeps the deterministic candidate ordering
    while each worker retains its own process-local network state, RNG stream,
    and MRP pheromone state.
    """

    chunks = _balanced_chunks(candidates, CALIBRATION_CANDIDATE_WORKERS)
    if len(chunks) == 1:
        return run_mrp_calibration_candidates(
            network, config, plan, chunks[0], repeat_scenarios=True,
            scenario_provider=provider, independent_scenario_trials=True,
        )
    with tempfile.TemporaryDirectory(prefix="mrp-calibration-stage-") as temporary:
        root = Path(temporary)
        jobs, processes, error_handles = [], [], []
        for index, chunk in enumerate(chunks):
            job_path, result_path = root / f"job-{index}.pkl", root / f"result-{index}.pkl"
            error_path, stderr_path = root / f"worker-{index}.error.txt", root / f"worker-{index}.stderr.txt"
            with job_path.open("wb") as handle:
                pickle.dump((network, config, plan, chunk, provider), handle, protocol=pickle.HIGHEST_PROTOCOL)
            command = (
                "import sys; sys.path.insert(0, 'src'); "
                "from experiments.real_calibration import _run_candidate_chunk; "
                f"_run_candidate_chunk({str(job_path)!r}, {str(result_path)!r}, {str(error_path)!r})"
            )
            error_handle = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-c", command],
                cwd=Path(__file__).resolve().parents[2], stdout=subprocess.DEVNULL, stderr=error_handle,
            )
            jobs.append((result_path, error_path, stderr_path))
            processes.append(process)
            error_handles.append(error_handle)
        failures = []
        for process, (_, error_path, stderr_path), error_handle in zip(processes, jobs, error_handles):
            return_code = process.wait()
            error_handle.close()
            if return_code != 0:
                diagnostic = error_path if error_path.exists() else stderr_path
                message = diagnostic.read_text(encoding="utf-8") if diagnostic.exists() else "no worker diagnostic was written"
                failures.append(message)
        if failures:
            raise RealCalibrationError(f"isolated calibration worker failed: {failures[0]}")
        candidates_result = []
        for result_path, _, _ in jobs:
            with result_path.open("rb") as handle:
                candidates_result.extend(pickle.load(handle))
    positions = {candidate.config_id: index for index, candidate in enumerate(candidates)}
    ranked = _rank(tuple(sorted(candidates_result, key=lambda item: positions[item.candidate.config_id])))
    selected = next((item.candidate.config_id for item in ranked if item.rank == 1), None)
    overlap = bool({tuple(item.event_nodes) for item in plan.calibration_scenarios} & {tuple(item.event_nodes) for item in plan.evaluation_scenarios})
    return CalibrationResult(plan, audit_calibration_parameters(config), ranked, SELECTION_RULE, selected, overlap)


def _run_candidate_chunk(job_path: str, result_path: str, error_path: str) -> None:
    """Subprocess entry point; result file appears only after a full chunk completes."""

    try:
        with Path(job_path).open("rb") as handle:
            network, config, plan, candidates, provider = pickle.load(handle)
        result = _run_mrp_calibration_candidate_subset(
            network, config, plan, candidates, repeat_scenarios=True,
            scenario_provider=provider, independent_scenario_trials=True,
        )
        temporary = Path(result_path).with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(result.candidates, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(result_path)
    except BaseException:
        Path(error_path).write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _balanced_chunks(candidates, max_workers):
    count = min(max_workers, len(candidates))
    return tuple(tuple(candidates[index::count]) for index in range(count))


def _top_pairs(result):
    ranked = sorted((item for item in result.candidates if item.rank is not None), key=lambda item: item.rank)
    return tuple(
        (dict(item.candidate.overrides)["mrp.c"], dict(item.candidate.overrides)["mrp.c1"])
        for item in ranked
    )[:2]


def _selected_hash(result, config):
    selected = next(item for item in result.candidates if item.rank == 1)
    selected_config = candidate_configuration(config, selected.candidate, len(CALIBRATION_SEEDS))
    payload = json.dumps(parameter_snapshot(selected_config), default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _summary_payload(item):
    return {"config_id": item.candidate.config_id, "overrides": dict(item.candidate.overrides), "rank": item.rank, **asdict(item.summary)}
