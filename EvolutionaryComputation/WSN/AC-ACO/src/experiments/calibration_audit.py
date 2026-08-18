"""MRP calibration provenance and nested-parameter snapshot helpers."""

from dataclasses import fields

from .calibration_types import ParameterAuditEntry, ParameterProvenance
from .config import ExperimentConfig


SEARCH_RANGE_PROVENANCE = "USER_SELECTED_CALIBRATION_RANGE"
TUNABLE_PATHS = frozenset({
    "mrp.lambda_coefficient", "mrp.ttl", "mrp.num_sants", "mrp.c0", "mrp.c", "mrp.c1",
    "mrp.heuristic_bounds.mu_min", "mrp.heuristic_bounds.mu_max",
    "mrp.heuristic_bounds.eta_min", "mrp.heuristic_bounds.eta_max",
    "mrp_pheromone_lifecycle",
})
RUNTIME_ADAPTATION_FIELDS = frozenset({
    "charge_ant_energy", "ant_control_packet_bits",
})


def audit_calibration_parameters(config: ExperimentConfig) -> tuple[ParameterAuditEntry, ...]:
    """Return source-backed and unresolved values without inventing a range."""

    parameters = config.mrp_parameters
    lifecycle_provenance = _configured_provenance(
        config, "experiment.mrp_pheromone_lifecycle", ParameterProvenance.USER_SELECTED_PARAMETER,
    )
    entries = [
        ParameterAuditEntry(
            "mrp_pheromone_lifecycle", config.mrp_pheromone_lifecycle,
            lifecycle_provenance, lifecycle_provenance == ParameterProvenance.USER_SELECTED_CALIBRATION_RANGE,
            "Paper does not establish cross-round reset versus persistence; this primary run fixes RESET_PER_DISCOVERY.",
        ),
        ParameterAuditEntry(
            "phase1.timer_scale_q", None, ParameterProvenance.PAPER_DEFINED_VALUE_UNREPORTED,
            False, "Eq. (25) names q but supplies no numeric value; current runner does not consume it.",
        ),
        ParameterAuditEntry(
            "phase1.rss_threshold", None, ParameterProvenance.PAPER_DEFINED_VALUE_UNREPORTED,
            False, "Event/RSS provider remains external to the algorithm.",
        ),
        ParameterAuditEntry(
            "phase1.neighbor_wait_ta", None, ParameterProvenance.PAPER_DEFINED_VALUE_UNREPORTED,
            False, "Central snapshot neighbor counting is the existing simulator mapping.",
        ),
        ParameterAuditEntry(
            "phase2.hop_counts", None, ParameterProvenance.SIMULATOR_IMPLEMENTATION_MAPPING,
            False, "Caller supplies hop counts; no calibration-side hop-count algorithm exists.",
        ),
    ]
    for field in fields(parameters.config):
        provenance = (
            ParameterProvenance.PAPER_ALGORITHM_RULE if field.name == "aant_probability"
            else ParameterProvenance.PAPER_SIMULATION_PARAMETER
        )
        entries.append(ParameterAuditEntry(
            f"mrp.config.{field.name}", getattr(parameters.config, field.name), provenance, False,
            "Primary reproduction fixes this paper-reported value.",
        ))
    for name in ("mu_min", "mu_max", "eta_min", "eta_max"):
        entries.append(ParameterAuditEntry(
            f"mrp.heuristic_bounds.{name}", getattr(parameters.heuristic_bounds, name),
            ParameterProvenance.PAPER_DEFINED_VALUE_UNREPORTED, True,
            "Eq. (27) calls these predetermined bounds but reports no numeric values.",
        ))
    for name in ("lambda_coefficient", "ttl", "num_sants", "c0", "c", "c1"):
        path = f"mrp.{name}"
        configured = _configured_provenance(
            config, path, ParameterProvenance.PAPER_DEFINED_VALUE_UNREPORTED,
        )
        entries.append(ParameterAuditEntry(
            path, getattr(parameters, name), configured,
            name != "c0" and configured == ParameterProvenance.PAPER_DEFINED_VALUE_UNREPORTED,
            _c0_reason() if name == "c0" and configured == ParameterProvenance.USER_SELECTED_PARAMETER else _unreported_reason(name),
        ))
    for field in fields(parameters):
        if field.name in {
            "config", "heuristic_bounds", "lambda_coefficient", "ttl",
            "num_sants", "c0", "c", "c1", *RUNTIME_ADAPTATION_FIELDS,
        }:
            continue
        entries.append(ParameterAuditEntry(
            f"mrp.{field.name}", getattr(parameters, field.name),
            ParameterProvenance.EXISTING_SIMULATOR_PARAMETER, False,
            "Shared repository radio/simulator setting; calibration does not alter it.",
        ))
    for field in fields(config.ac_aco_parameters):
        entries.append(ParameterAuditEntry(
            f"ac_aco.{field.name}", getattr(config.ac_aco_parameters, field.name),
            ParameterProvenance.EXISTING_SIMULATOR_PARAMETER, False,
            "Outside Pure-MRP calibration; retained unchanged for later comparison.",
        ))
    return tuple(entries)


def parameter_snapshot(config: ExperimentConfig) -> dict[str, object]:
    """Flatten all MRP values so frozen evaluation cannot inherit changed defaults."""

    parameters = config.mrp_parameters
    snapshot = {"mrp_pheromone_lifecycle": config.mrp_pheromone_lifecycle}
    for field in fields(parameters):
        if field.name not in {"config", "heuristic_bounds", *RUNTIME_ADAPTATION_FIELDS}:
            snapshot[f"mrp.{field.name}"] = getattr(parameters, field.name)
    for field in fields(parameters.heuristic_bounds):
        snapshot[f"mrp.heuristic_bounds.{field.name}"] = getattr(parameters.heuristic_bounds, field.name)
    for field in fields(parameters.config):
        snapshot[f"mrp.config.{field.name}"] = getattr(parameters.config, field.name)
    return snapshot


def _unreported_reason(name: str) -> str:
    reasons = {
        "lambda_coefficient": "Eq. (31) defines λ as a coefficient without a numeric value.",
        "ttl": "Paper defines TTL behavior but gives no initial TTL value.",
        "num_sants": "Paper relates SANT count to scale/application demand without fixing it.",
        "c0": "Eq. (33) names c0 as a coefficient without a numeric value.",
        "c": "Eq. (32) names c as a coefficient without a numeric value.",
        "c1": "Eq. (32) names c1 as a coefficient without a numeric value.",
    }
    return reasons[name]


def _c0_reason() -> str:
    return (
        "Fixed at 1.0 as a user-selected normalization convention: Eq. (33) scales route quality by c0, "
        "while the c1 term in Eq. (32) depends on the product c1*c0."
    )


def _configured_provenance(
    config: ExperimentConfig, path: str, fallback: ParameterProvenance,
) -> ParameterProvenance:
    """Read an explicit experiment provenance label without treating its detail as an enum value."""

    raw = config.parameter_provenance.get(path)
    if not raw:
        return fallback
    label = str(raw).split(":", 1)[0]
    try:
        return ParameterProvenance(label)
    except ValueError:
        return fallback
