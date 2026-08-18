"""Neutral raw-output and descriptive summaries for frozen final evaluation."""

from collections import defaultdict
from dataclasses import asdict
import csv
import gc
import json
import os
import statistics
from pathlib import Path
import tempfile

from .config import ExperimentMode
from .final_evaluation import validate_final_evaluation_result
from .final_evaluation_types import FinalEvaluationResult, NativeEvaluationTrial
from .io import csv_text, json_text, publish_output_files
from .io import _csv_value


class IncrementalFinalEvaluationWriter:
    """Stage canonical native-trial records without retaining round histories."""

    def __init__(self, context, output_directory: str | Path):
        self.context = context
        self.output = Path(output_directory)
        if self.output.exists():
            raise FileExistsError("experiment output directory must not already exist")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix=f".{self.output.name}.staging.", dir=self.output.parent))
        self._handles = {}
        self._writers = {}
        self._digests = []
        self._closed = False

    def append(self, item: NativeEvaluationTrial) -> None:
        """Persist a terminal trial immediately, then release its large history."""

        native = _native_row(item)
        self._write("native_trial_metrics.csv", native)
        for row in _round_rows(item):
            self._write("round_metrics.csv", row)
            if not row["success"]:
                self._write("failures.csv", row)
        self._digests.append(_digest(item, native))
        del item
        gc.collect()

    def finalize(self) -> Path:
        """Write small derived views, then atomically publish the staged directory."""

        if len(self._digests) != 180:
            raise ValueError("canonical final evaluation requires exactly 180 terminal native-trial records")
        self._close_streams()
        summaries = _digest_summaries(self._digests)
        files = {
            "evaluation_manifest.json": json_text(_manifest(self.context)),
            "seed_level_metrics.csv": csv_text(_digest_seed_rows(self._digests)),
            "summary.json": json_text({"mode_summaries": summaries, "seed_level_count": 30}),
            "summary.csv": csv_text(summaries),
            "pure_mrp_scenario_summary.csv": csv_text(_digest_pure_summaries(self._digests)),
        }
        for name, text in files.items():
            (self.stage / name).write_text(text, encoding="utf-8", newline="")
        for name in ("native_trial_metrics.csv", "round_metrics.csv", "failures.csv"):
            path = self.stage / name
            if not path.exists():
                path.write_text(csv_text(()), encoding="utf-8", newline="")
        self.stage.replace(self.output)
        self._closed = True
        return self.output

    def abort(self) -> None:
        """Leave only an explicitly non-published staging ledger for forensics."""

        if self._closed:
            return
        self._close_streams()
        (self.stage / "run_state.json").write_text(json_text({"status": "INCOMPLETE_NOT_PUBLISHED"}), encoding="utf-8")
        self._closed = True

    def _write(self, name, row) -> None:
        writer = self._writers.get(name)
        if writer is None:
            handle = (self.stage / name).open("w", encoding="utf-8", newline="")
            self._handles[name] = handle
            writer = self._writers[name] = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
        writer.writerow({key: _csv_value(value) for key, value in row.items()})
        self._handles[name].flush()

    def _close_streams(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._writers.clear()


def _digest(item, native):
    rounds = item.mode_result.rounds
    routing = [row for row in rounds if row.discovered_route_counts is not None]
    final = rounds[-1] if rounds else None
    energy = sum(row.actual_round_energy or 0.0 for row in rounds)
    payload = sum(row.delivered_payload_count or 0 for row in rounds)
    return {
        **native, "normalized_energy": None if payload <= 0 or item.fnd.algorithm_failure_before_fnd else energy / payload,
        "route_attempts": len(routing),
        "route_discoveries": sum(any(count > 0 for count in row.discovered_route_counts) for row in routing),
        "multipath_ready": sum(any(row.multipath_ready or ()) for row in routing),
        "final": final,
    }


def _digest_summaries(digests):
    return [_digest_summary([item for item in digests if item["algorithm"] == mode.value]) for mode in ExperimentMode]


def _digest_pure_summaries(digests):
    grouped = defaultdict(list)
    for item in digests:
        if item["algorithm"] == ExperimentMode.PURE_MRP.value:
            grouped[item["scenario_id"]].append(item)
    return [_digest_summary(items) for _, items in sorted(grouped.items())]


def _digest_summary(items):
    mode = items[0]["algorithm"] if items else None
    completed = [item for item in items if not item["algorithm_failure_before_fnd"]]
    finals = [item["final"] for item in completed if item["final"] is not None]
    normalized = [item["normalized_energy"] for item in completed if item["normalized_energy"] is not None]
    observed = [item["fnd_round_number_one_based"] for item in items if item["fnd_round_number_one_based"] is not None]
    attempts = sum(item["route_attempts"] for item in items)
    return {
        "mode": mode, "scenario_id": items[0]["scenario_id"] if mode == ExperimentMode.PURE_MRP.value else None,
        "native_trial_count": len(items), "successful_trial_count": len(completed),
        "algorithm_failure_count": len(items) - len(completed),
        "mean_cumulative_physical_energy_at_stop": _mean([row.cumulative_actual_energy for row in finals]),
        "mean_residual_energy": _mean([row.residual_mean for row in finals]),
        "mean_population_energy_std": _mean([row.residual_std_population for row in finals]),
        "energy_per_delivered_payload": _mean(normalized), "energy_per_delivered_payload_sample_count": len(normalized),
        "fnd_observed_count": len(observed), "fnd_censored_count": sum(item["fnd_censored_at_cap"] for item in items),
        "fnd_observed_mean_one_based": _mean(observed), "fnd_observed_stddev": _std(observed),
        "routing_round_attempt_count": attempts,
        "route_discovery_success_rate": _ratio(sum(item["route_discoveries"] for item in items), attempts),
        "multipath_ready_rate": _ratio(sum(item["multipath_ready"] for item in items), attempts),
        "raw_energy_fairness": "LIMITED / WORKLOAD-SENSITIVE",
        "fnd_interpretation": "NATIVE_SEMANTICS_ONLY / WORKLOAD-SENSITIVE",
    }


def _digest_seed_rows(digests):
    rows = []
    for seed in sorted({item["trial_seed"] for item in digests}):
        items = [item for item in digests if item["trial_seed"] == seed]
        row = {"trial_seed": seed, "pure_mrp_scenario_count": sum(item["algorithm"] == "PURE_MRP" for item in items)}
        for mode in (ExperimentMode.BASELINE, ExperimentMode.HYBRID):
            found = [item for item in items if item["algorithm"] == mode.value]
            item = found[0] if len(found) == 1 else None
            prefix = mode.value.lower()
            row.update({f"{prefix}_native_observation_count": len(found), f"{prefix}_normalized_energy": None if item is None else item["normalized_energy"], f"{prefix}_fnd_round_number_one_based": None if item is None else item["fnd_round_number_one_based"], f"{prefix}_fnd_censored_at_cap": None if item is None else item["fnd_censored_at_cap"]})
        pure = [item for item in items if item["algorithm"] == "PURE_MRP"]
        energy = [item["normalized_energy"] for item in pure if item["normalized_energy"] is not None]
        observed = [item["fnd_round_number_one_based"] for item in pure if item["fnd_round_number_one_based"] is not None]
        row.update({"pure_mrp_scenario_ids": tuple(item["scenario_id"] for item in pure), "pure_mrp_normalized_energy_mean": _mean(energy), "pure_mrp_normalized_energy_stddev": _std(energy), "pure_mrp_fnd_observed_count": len(observed), "pure_mrp_fnd_censored_count": sum(item["fnd_censored_at_cap"] for item in pure), "pure_mrp_fnd_observed_mean": _mean(observed), "pure_mrp_fnd_observed_min": min(observed) if observed else None, "pure_mrp_fnd_observed_max": max(observed) if observed else None})
        rows.append(row)
    return rows


def write_final_evaluation_outputs(result: FinalEvaluationResult, output_directory: str | Path) -> Path:
    """Publish all three requested result levels atomically into a new directory."""

    validate_final_evaluation_result(result)
    context = result.context
    config, existing_map, scenarios = context.configuration, context.existing_map, context.scenarios
    native_rows = [_native_row(item) for item in result.native_trials]
    round_rows = [row for item in result.native_trials for row in _round_rows(item)]
    failures = [row for row in round_rows if not row["success"]]
    seed_rows = _seed_rows(result.native_trials)
    summaries = _mode_summaries(result.native_trials)
    pure_by_scenario = [_mode_summary(items) for items in _group_pure_scenarios(result.native_trials).values()]
    manifest = _manifest(context)
    files = {
        "evaluation_manifest.json": json_text(manifest),
        "round_metrics.csv": csv_text(round_rows),
        "native_trial_metrics.csv": csv_text(native_rows),
        "seed_level_metrics.csv": csv_text(seed_rows),
        "failures.csv": csv_text(failures),
        "summary.json": json_text({"mode_summaries": summaries, "seed_level_count": len(seed_rows)}),
        "summary.csv": csv_text(summaries),
        "pure_mrp_scenario_summary.csv": csv_text(pure_by_scenario),
    }
    return publish_output_files(output_directory, files)


def _manifest(context):
    config, existing_map, scenarios = context.configuration, context.existing_map, context.scenarios
    return {
        "checkpoint": "10 — FROZEN CONTROLLED THREE-MODE EVALUATION",
        "selected_config_id": context.preflight.selected_config_id,
        "selected_config_hash": context.preflight.selected_config_hash,
        "scenario_set_id": "evaluation-v1",
        "scenario_hash": context.preflight.evaluation_scenario_hash,
        "calibration_scenario_hash": context.preflight.calibration_scenario_hash,
        "evaluation_seeds": sorted({item.trial_seed for item in result.native_trials}),
        "round_cap": context.preflight.max_rounds,
        "round_cap_provenance": "USER_SELECTED_EVALUATION_CAP / PAPER_SCALE_REFERENCED",
        "paper_prescribes_5000": False,
        "primary_stop": "FND",
        "round_indexing": "round_index_zero_based; round_number_one_based = index + 1",
        "seed_protocol": "derive_seed_streams(seed): separate Baseline AC-ACO, Pure-MRP, Hybrid AC-ACO, Hybrid MRP streams",
        "traffic_scope": {
            "BASELINE": "all-live-node membership / traffic",
            "HYBRID": "all-live-node membership / traffic",
            "PURE_MRP": "event-scoped source traffic",
            "raw_energy_fairness": "LIMITED / comparable only with workload normalization",
        },
        "mode_scenario_dependence": {
            "BASELINE": "scenario-independent; one native trial per seed",
            "HYBRID": "scenario-independent under frozen identical hop snapshots; one native trial per seed",
            "PURE_MRP": "scenario-conditioned; four native trials per seed",
        },
        "network": existing_map.network,
        "parameter_provenance": dict(config.parameter_provenance),
        "scenarios": {item.scenario_id: item.event_anchor_sensor for item in scenarios.evaluation},
        "multipath_denominator_audit": context.preflight.multipath_denominator_audit,
    }


def _round_rows(item):
    for row in item.mode_result.rounds:
        result = asdict(row)
        result.update({
            "algorithm": item.algorithm.value,
            "scenario_id": item.scenario_id,
            "scenario_anchor_sensor": item.scenario_anchor_sensor,
            "round_number_one_based": row.round_index + 1,
        })
        yield result


def _native_row(item):
    final = item.mode_result.rounds[-1] if item.mode_result.rounds else None
    return {
        "algorithm": item.algorithm.value, "trial_seed": item.trial_seed,
        "scenario_id": item.scenario_id, "scenario_anchor_sensor": item.scenario_anchor_sensor,
        "rounds_executed": item.fnd.rounds_executed,
        "algorithm_failure_before_fnd": item.fnd.algorithm_failure_before_fnd,
        "fnd_round_index_zero_based": item.fnd.round_index_zero_based,
        "fnd_round_number_one_based": item.fnd.round_number_one_based,
        "fnd_censored_at_cap": item.fnd.censored_at_cap,
        "hnd": "NOT_EVALUATED_PRIMARY_FND_STOP", "lnd": "NOT_EVALUATED_PRIMARY_FND_STOP",
        "success": None if final is None else final.success,
        "failure_stage": None if final is None else final.failure_stage,
        "failure_reason": None if final is None else final.failure_reason,
        "cumulative_actual_energy": None if final is None else final.cumulative_actual_energy,
        "residual_mean": None if final is None else final.residual_mean,
        "residual_std_population": None if final is None else final.residual_std_population,
    }


def _seed_rows(trials):
    grouped = defaultdict(list)
    for item in trials:
        grouped[item.trial_seed].append(item)
    return [_seed_row(seed, items) for seed, items in sorted(grouped.items())]


def _seed_row(seed, items):
    by_mode = {mode: [item for item in items if item.algorithm is mode] for mode in ExperimentMode}
    row = {"trial_seed": seed, "pure_mrp_scenario_count": len(by_mode[ExperimentMode.PURE_MRP])}
    for mode in (ExperimentMode.BASELINE, ExperimentMode.HYBRID):
        row.update(_single_seed_values(mode, by_mode[mode]))
    pure = by_mode[ExperimentMode.PURE_MRP]
    energy = [_normalized(item) for item in pure if _normalized(item) is not None]
    observed = [item.fnd.round_number_one_based for item in pure if item.fnd.round_number_one_based is not None]
    row.update({
        "pure_mrp_scenario_ids": tuple(item.scenario_id for item in pure),
        "pure_mrp_normalized_energy_mean": _mean(energy),
        "pure_mrp_normalized_energy_stddev": _std(energy),
        "pure_mrp_fnd_observed_count": len(observed),
        "pure_mrp_fnd_censored_count": sum(item.fnd.censored_at_cap for item in pure),
        "pure_mrp_fnd_observed_mean": _mean(observed),
        "pure_mrp_fnd_observed_min": min(observed) if observed else None,
        "pure_mrp_fnd_observed_max": max(observed) if observed else None,
    })
    return row


def _single_seed_values(mode, items):
    item = items[0] if len(items) == 1 else None
    prefix = mode.value.lower()
    return {
        f"{prefix}_native_observation_count": len(items),
        f"{prefix}_normalized_energy": None if item is None else _normalized(item),
        f"{prefix}_fnd_round_number_one_based": None if item is None else item.fnd.round_number_one_based,
        f"{prefix}_fnd_censored_at_cap": None if item is None else item.fnd.censored_at_cap,
    }


def _mode_summaries(trials):
    return [_mode_summary([item for item in trials if item.algorithm is mode]) for mode in ExperimentMode]


def _group_pure_scenarios(trials):
    result = defaultdict(list)
    for item in trials:
        if item.algorithm is ExperimentMode.PURE_MRP:
            result[item.scenario_id].append(item)
    return dict(sorted(result.items()))


def _mode_summary(items):
    mode = items[0].algorithm.value if items else None
    completed = [item for item in items if not item.fnd.algorithm_failure_before_fnd]
    finals = [item.mode_result.rounds[-1] for item in completed if item.mode_result.rounds]
    normalized = [_normalized(item) for item in completed if _normalized(item) is not None]
    observed = [item.fnd.round_number_one_based for item in items if item.fnd.round_number_one_based is not None]
    routing = [row for item in items for row in item.mode_result.rounds if row.discovered_route_counts is not None]
    route_attempts = len(routing)
    return {
        "mode": mode, "scenario_id": items[0].scenario_id if items and mode == ExperimentMode.PURE_MRP.value else None,
        "native_trial_count": len(items), "successful_trial_count": len(completed),
        "algorithm_failure_count": len(items) - len(completed),
        "mean_cumulative_physical_energy_at_stop": _mean([row.cumulative_actual_energy for row in finals]),
        "mean_residual_energy": _mean([row.residual_mean for row in finals]),
        "mean_population_energy_std": _mean([row.residual_std_population for row in finals]),
        "energy_per_delivered_payload": _mean(normalized),
        "energy_per_delivered_payload_sample_count": len(normalized),
        "fnd_observed_count": len(observed), "fnd_censored_count": sum(item.fnd.censored_at_cap for item in items),
        "fnd_observed_mean_one_based": _mean(observed), "fnd_observed_stddev": _std(observed),
        "routing_round_attempt_count": route_attempts,
        "route_discovery_success_rate": _ratio(sum(any(count > 0 for count in row.discovered_route_counts) for row in routing), route_attempts),
        "multipath_ready_rate": _ratio(sum(any(row.multipath_ready or ()) for row in routing), route_attempts),
        "raw_energy_fairness": "LIMITED / WORKLOAD-SENSITIVE",
        "fnd_interpretation": "NATIVE_SEMANTICS_ONLY / WORKLOAD-SENSITIVE",
    }


def _normalized(item):
    rows = item.mode_result.rounds
    if not rows or item.fnd.algorithm_failure_before_fnd:
        return None
    energy = sum(row.actual_round_energy for row in rows if row.actual_round_energy is not None)
    payload = sum(row.delivered_payload_count for row in rows if row.delivered_payload_count is not None)
    return None if payload <= 0 else energy / payload


def _ratio(numerator, denominator):
    return None if not denominator else numerator / denominator


def _mean(values):
    return statistics.fmean(values) if values else None


def _std(values):
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
