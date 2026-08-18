"""Run the approved TTL-only network-wide Pure/Hybrid calibration study."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from experiments.network_wide_ttl_reporting import aggregate_ttl_trials  # noqa: E402
from experiments.network_wide_ttl_trial import run_ttl_trial  # noqa: E402
from experiments.real_calibration import CALIBRATION_SEEDS, EVALUATION_SEEDS  # noqa: E402
from experiments.scenario_generation import load_frozen_scenario_sets  # noqa: E402


TTLS = (3, 4, 5, 6)
MODES = ("mrp_pure", "mrp_hybrid")
ROUND_BUDGET = 10
DEFAULT_OUTPUT = ROOT / "real_calibration_v2" / "network_wide_ttl_recalibration"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--ttl", type=int, choices=TTLS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scenario")
    args = parser.parse_args(argv)
    return _worker(args) if args.worker else _orchestrate(args.output.resolve())


def _worker(args):
    if None in (args.mode, args.ttl, args.seed, args.scenario):
        raise SystemExit("worker requires mode, ttl, seed, and scenario")
    if args.seed in EVALUATION_SEEDS or args.seed not in CALIBRATION_SEEDS:
        raise SystemExit("worker seed is not in the approved calibration set")
    destination = _trial_path(args.output.resolve(), args.ttl, args.mode, args.scenario, args.seed)
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    result = run_ttl_trial(
        ROOT, mode=args.mode, ttl=args.ttl, seed=args.seed,
        scenario_id=args.scenario, round_budget=ROUND_BUDGET,
    )
    result.update({
        "pid": os.getpid(), "type": "NETWORK_WIDE_TTL_CALIBRATION_TRIAL",
        "reserved_evaluation_seed_used": False,
    })
    _atomic_json(destination, result)
    print(json.dumps({
        "pid": result["pid"], "ttl": args.ttl, "mode": args.mode,
        "scenario": args.scenario, "seed": args.seed,
        "rounds": result["successful_complete_rounds"], "status": result["status"],
    }), flush=True)
    return 0


def _orchestrate(output):
    scenarios = load_frozen_scenario_sets(ROOT / "real_calibration_v2" / "scenario_manifest.json")
    scenario_ids = tuple(item.scenario_id for item in scenarios.calibration)
    if set(CALIBRATION_SEEDS) & set(EVALUATION_SEEDS):
        raise RuntimeError("calibration/evaluation seed overlap")
    output.mkdir(parents=True, exist_ok=True)
    plan = [
        (ttl, mode, scenario_id, seed)
        for ttl in TTLS for mode in MODES
        for scenario_id in scenario_ids for seed in CALIBRATION_SEEDS
    ]
    _atomic_json(output / "study_manifest.json", {
        "type": "NETWORK_WIDE_MRP_TTL_RECALIBRATION",
        "scientific_freeze_status": "STALE",
        "old_ttl": 3,
        "parameter_change_under_study": "mrp.ttl only",
        "status": "RUNNING", "created_at": _now(), "ttl_candidates": TTLS,
        "modes": MODES, "calibration_scenarios": scenario_ids,
        "calibration_seeds": CALIBRATION_SEEDS, "round_budget": ROUND_BUDGET,
        "planned_trial_count": len(plan), "reserved_evaluation_seeds_used": False,
    })
    for index, (ttl, mode, scenario_id, seed) in enumerate(plan, 1):
        destination = _trial_path(output, ttl, mode, scenario_id, seed)
        if destination.exists():
            _validate_existing(destination, ttl, mode, scenario_id, seed)
            print(f"[{index}/{len(plan)}] staged {destination.name}", flush=True)
            continue
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--output", str(output), "--ttl", str(ttl), "--mode", mode,
            "--scenario", scenario_id, "--seed", str(seed),
        ]
        print(f"[{index}/{len(plan)}] START ttl={ttl} mode={mode} scenario={scenario_id} seed={seed}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0 or not destination.is_file():
            raise RuntimeError(f"worker failed with exit code {completed.returncode}: {command}")
        print(f"[{index}/{len(plan)}] END exit=0 active_workers=0", flush=True)
    trials = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "trials").rglob("*.json"))]
    aggregate = aggregate_ttl_trials(trials)
    _atomic_json(output / "aggregate_results.json", {
        "type": "NETWORK_WIDE_MRP_TTL_RECALIBRATION_AGGREGATE",
        "status": "COMPLETE", "completed_at": _now(), "trial_count": len(trials),
        "reserved_evaluation_seeds_used": False, "results": aggregate,
    })
    manifest = json.loads((output / "study_manifest.json").read_text(encoding="utf-8"))
    manifest.update({"status": "COMPLETE", "completed_at": _now(), "completed_trial_count": len(trials)})
    _atomic_json(output / "study_manifest.json", manifest)
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


def _trial_path(output, ttl, mode, scenario, seed):
    return output / "trials" / f"ttl_{ttl}" / mode / f"{scenario}_seed{seed}.json"


def _validate_existing(path, ttl, mode, scenario, seed):
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = (ttl, mode, scenario, seed, False)
    actual = (value["ttl"], value["mode"], value["scenario_id"], value["seed"], value["reserved_evaluation_seed_used"])
    if actual != expected:
        raise RuntimeError(f"staged trial identity mismatch: {path}")


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _now():
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
