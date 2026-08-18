"""Validation, enumeration, and application of caller-owned calibration grids."""

from dataclasses import replace
from itertools import product
import hashlib
import json
import math

from mrp import HeuristicBounds, PheromoneLifecycle

from .calibration_audit import SEARCH_RANGE_PROVENANCE, TUNABLE_PATHS
from .calibration_types import CalibrationCandidate, CalibrationPlan
from .config import EvaluationView, ExperimentConfig, ExperimentMode


class CalibrationInputError(ValueError):
    """Raised before calibration when caller input would hide a source/domain error."""


def validate_calibration_plan(plan: CalibrationPlan) -> None:
    if not isinstance(plan, CalibrationPlan):
        raise CalibrationInputError("calibration requires a CalibrationPlan")
    _seed_set(plan.calibration_seed_set, "calibration")
    _seed_set(plan.evaluation_seed_set, "evaluation")
    if set(plan.calibration_seed_set) & set(plan.evaluation_seed_set):
        raise CalibrationInputError("calibration and evaluation seed sets must be disjoint")
    if not plan.calibration_scenario_set_id or not plan.evaluation_scenario_set_id:
        raise CalibrationInputError("calibration and evaluation scenario-set IDs are required")
    if plan.calibration_scenario_set_id == plan.evaluation_scenario_set_id:
        raise CalibrationInputError("calibration and evaluation scenario-set IDs must differ")
    if not plan.calibration_scenarios or not plan.evaluation_scenarios:
        raise CalibrationInputError("calibration and evaluation scenarios must be supplied")
    if not plan.search_space:
        raise CalibrationInputError("calibration requires a non-empty caller-supplied search space")
    if set(plan.search_space) != set(plan.search_space_provenance):
        raise CalibrationInputError("each calibration search-space key requires provenance")
    for path, values in plan.search_space.items():
        if path not in TUNABLE_PATHS:
            raise CalibrationInputError(f"{path} is not an unresolved calibratable parameter")
        required_provenance = (
            "USER_SELECTED_SIMULATION_POLICY"
            if path == "mrp_pheromone_lifecycle" else SEARCH_RANGE_PROVENANCE
        )
        if plan.search_space_provenance[path] != required_provenance:
            raise CalibrationInputError(f"{path} must be labeled {required_provenance}")
        if not isinstance(values, tuple) or not values:
            raise CalibrationInputError(f"{path} requires a non-empty finite candidate tuple")
    if plan.minimum_completion_rate is not None:
        _finite_interval(plan.minimum_completion_rate, "minimum completion rate")


def enumerate_calibration_candidates(plan: CalibrationPlan) -> tuple[CalibrationCandidate, ...]:
    """Enumerate a finite caller grid in canonical-key order."""

    validate_calibration_plan(plan)
    paths = tuple(sorted(plan.search_space))
    candidates = []
    for values in product(*(plan.search_space[path] for path in paths)):
        overrides = tuple(zip(paths, values))
        _validate_overrides(overrides)
        candidates.append(CalibrationCandidate(_candidate_id(overrides), overrides))
    return tuple(candidates)


def candidate_configuration(
    base_config: ExperimentConfig, candidate: CalibrationCandidate, trial_count: int,
) -> ExperimentConfig:
    """Apply only approved unresolved paths and force calibration to Pure MRP."""

    if not isinstance(base_config, ExperimentConfig):
        raise CalibrationInputError("base configuration must be an ExperimentConfig")
    if not isinstance(trial_count, int) or isinstance(trial_count, bool) or trial_count <= 0:
        raise CalibrationInputError("calibration trial count must be positive")
    _validate_overrides(candidate.overrides)
    config = replace(
        base_config, modes=(ExperimentMode.PURE_MRP,), trial_count=trial_count,
        evaluation_view=EvaluationView.NORMALIZED_REPORTING,
    )
    for path, value in candidate.overrides:
        config = _apply_path(config, path, value)
    return config


def apply_parameter_snapshot(
    base_config: ExperimentConfig, snapshot: dict[str, object], runtime_overrides: dict[str, object] | None = None,
) -> ExperimentConfig:
    """Restore a frozen MRP snapshot; calibrated overrides are never accepted here."""

    if runtime_overrides:
        raise CalibrationInputError("frozen calibration configuration rejects runtime parameter overrides")
    config = base_config
    for path in sorted(snapshot):
        config = _apply_path(config, path, snapshot[path], allow_fixed=True)
    return config


def candidate_from_overrides(overrides: tuple[tuple[str, object], ...]) -> CalibrationCandidate:
    _validate_overrides(overrides)
    canonical = tuple(sorted(overrides))
    return CalibrationCandidate(_candidate_id(canonical), canonical)


def _apply_path(config: ExperimentConfig, path: str, value: object, allow_fixed: bool = False) -> ExperimentConfig:
    if path == "mrp_pheromone_lifecycle":
        lifecycle = value if isinstance(value, PheromoneLifecycle) else PheromoneLifecycle(value)
        return replace(config, mrp_pheromone_lifecycle=lifecycle)
    if not path.startswith("mrp."):
        raise CalibrationInputError(f"unsupported parameter path: {path}")
    parameters = config.mrp_parameters
    parts = path.split(".")
    if len(parts) == 2 and hasattr(parameters, parts[1]):
        return replace(config, mrp_parameters=replace(parameters, **{parts[1]: value}))
    if len(parts) == 3 and parts[1] == "heuristic_bounds" and hasattr(parameters.heuristic_bounds, parts[2]):
        bounds = replace(parameters.heuristic_bounds, **{parts[2]: value})
        _validate_bounds(bounds)
        return replace(config, mrp_parameters=replace(parameters, heuristic_bounds=bounds))
    if len(parts) == 3 and parts[1] == "config" and hasattr(parameters.config, parts[2]) and allow_fixed:
        return replace(config, mrp_parameters=replace(parameters, config=replace(parameters.config, **{parts[2]: value})))
    raise CalibrationInputError(f"unsupported parameter path: {path}")


def _validate_overrides(overrides: tuple[tuple[str, object], ...]) -> None:
    paths = tuple(path for path, _ in overrides)
    if len(paths) != len(set(paths)) or any(path not in TUNABLE_PATHS for path in paths):
        raise CalibrationInputError("calibration overrides must be unique unresolved parameter paths")
    values = dict(overrides)
    for name in ("mrp.lambda_coefficient",):
        if name in values:
            _finite_nonnegative(values[name], name)
    for name in ("mrp.c0",):
        if name in values:
            _finite_positive(values[name], name)
    for name in ("mrp.c", "mrp.c1"):
        if name in values:
            _finite(values[name], name)
    if "mrp.ttl" in values and (not isinstance(values["mrp.ttl"], int) or isinstance(values["mrp.ttl"], bool) or values["mrp.ttl"] < 0):
        raise CalibrationInputError("mrp.ttl must be a non-negative integer")
    if "mrp.num_sants" in values and (not isinstance(values["mrp.num_sants"], int) or isinstance(values["mrp.num_sants"], bool) or values["mrp.num_sants"] <= 0):
        raise CalibrationInputError("mrp.num_sants must be a positive integer")
    if "mrp_pheromone_lifecycle" in values and not isinstance(values["mrp_pheromone_lifecycle"], PheromoneLifecycle):
        raise CalibrationInputError("mrp_pheromone_lifecycle must be a PheromoneLifecycle")
    bounds = {name.rsplit(".", 1)[-1]: value for name, value in overrides if ".heuristic_bounds." in name}
    if bounds:
        for name, value in bounds.items():
            _finite_nonnegative(value, f"heuristic bound {name}")
        if "mu_min" in bounds and "mu_max" in bounds and bounds["mu_min"] > bounds["mu_max"]:
            raise CalibrationInputError("mu_min cannot exceed mu_max")
        if "eta_min" in bounds and "eta_max" in bounds and bounds["eta_min"] > bounds["eta_max"]:
            raise CalibrationInputError("eta_min cannot exceed eta_max")


def _validate_bounds(bounds: HeuristicBounds) -> None:
    _validate_overrides(tuple((f"mrp.heuristic_bounds.{name}", getattr(bounds, name)) for name in ("mu_min", "mu_max", "eta_min", "eta_max")))


def _candidate_id(overrides: tuple[tuple[str, object], ...]) -> str:
    payload = json.dumps(_json_value(dict(overrides)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _json_value(value):
    if isinstance(value, PheromoneLifecycle):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _seed_set(values, label):
    if not values or any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise CalibrationInputError(f"{label} seed set must contain integers")
    if len(values) != len(set(values)):
        raise CalibrationInputError(f"{label} seed set must be unique")


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CalibrationInputError(f"{label} must be finite")


def _finite_nonnegative(value, label):
    _finite(value, label)
    if value < 0:
        raise CalibrationInputError(f"{label} must be non-negative")


def _finite_positive(value, label):
    _finite_nonnegative(value, label)
    if value == 0:
        raise CalibrationInputError(f"{label} must be positive")


def _finite_interval(value, label):
    _finite_nonnegative(value, label)
    if value > 1:
        raise CalibrationInputError(f"{label} cannot exceed one")
