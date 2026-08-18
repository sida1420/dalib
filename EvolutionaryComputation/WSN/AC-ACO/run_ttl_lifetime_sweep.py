"""Sequential TTL-only lifetime sweep for network-wide Pure and Hybrid MRP."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from experiments.manual_runner import run_manual_route_exhaustion  # noqa: E402
from experiments.native_case import NATIVE_CASE_ID  # noqa: E402


BASE_TTLS = (3, 4, 5, 6, 7, 8)
MODES = ("mrp_pure", "mrp_hybrid")
WORKER_MODES = MODES + ("mrp_hybrid_lifetime",)
SEED = 33001
SAFETY_MAX_ROUNDS = 100000
EXPECTED_STATE_HASH = "fbfc3cb1eb57e2d72b6cf7eb69a05561e25ff641b2e540ce4597f39f56c5b428"
DEFAULT_OUTPUT = ROOT / "diagnostic_ttl_lifetime_sweep"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--ttl", type=int)
    parser.add_argument("--mode", choices=WORKER_MODES)
    args = parser.parse_args(argv)
    return _worker(args) if args.worker else _orchestrate(args.output.resolve())


def _worker(args):
    if args.ttl is None or args.mode is None:
        raise SystemExit("worker requires --ttl and --mode")
    destination = _trial_path(args.output.resolve(), args.ttl, args.mode)
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    result = run_manual_route_exhaustion(
        ROOT, mode=args.mode, seed=SEED, native_case=True,
        safety_max_rounds=SAFETY_MAX_ROUNDS, ttl_override=args.ttl,
        progress=_print_progress,
    )
    if result["initial_state_hash"] != EXPECTED_STATE_HASH:
        raise RuntimeError("native initial-state identity mismatch")
    result.update({
        "type": "TTL_LIFETIME_SWEEP_TRIAL", "ttl": args.ttl,
        "parameter_change": "mrp.ttl only", "reserved_evaluation_seed_used": False,
    })
    _atomic_json(destination, result)
    print(json.dumps({
        "pid": os.getpid(), "mode": args.mode, "ttl": args.ttl,
        "rounds": result["successful_rounds"], "status": result["status"],
        "peak_rss_bytes": result["peak_rss_bytes"],
    }), flush=True)
    return 0


def _orchestrate(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "MRP_TTL_LIFETIME_SWEEP", "status": "RUNNING",
        "created_at": _now(), "base_ttl_candidates": BASE_TTLS,
        "conditional_ttl_10": "TTL8 improves over TTL7 and remains TTL-limited",
        "modes": MODES, "seed": SEED, "case": NATIVE_CASE_ID,
        "initial_state_hash": EXPECTED_STATE_HASH,
        "safety_max_successful_rounds": SAFETY_MAX_ROUNDS,
        "parameter_change": "mrp.ttl only", "reserved_evaluation_seeds_used": False,
    }
    _atomic_json(output / "sweep_manifest.json", manifest)
    for ttl in BASE_TTLS:
        for mode in MODES:
            _run_one(output, ttl, mode)
    initial = _load_results(output, BASE_TTLS)
    tested_ttls = list(BASE_TTLS)
    if _should_test_ttl_10(initial):
        tested_ttls.append(10)
        for mode in MODES:
            _run_one(output, 10, mode)
    results = _load_results(output, tested_ttls)
    aggregate = _aggregate(results, tested_ttls)
    _atomic_json(output / "sweep_results.json", aggregate)
    manifest.update({
        "status": "COMPLETE", "completed_at": _now(),
        "ttl_candidates_tested": tested_ttls, "trial_count": len(results),
    })
    _atomic_json(output / "sweep_manifest.json", manifest)
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


def _run_one(output, ttl, mode):
    destination = _trial_path(output, ttl, mode)
    if destination.exists():
        _validate_existing(destination, ttl, mode)
        print(f"STAGED ttl={ttl} mode={mode} active_workers=0", flush=True)
        return
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--output", str(output), "--ttl", str(ttl), "--mode", mode,
    ]
    print(f"START ttl={ttl} mode={mode} time={_now()} active_workers=1", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"worker failed exit={completed.returncode}: {command}")
    print(f"END ttl={ttl} mode={mode} time={_now()} exit=0 active_workers=0", flush=True)


def _should_test_ttl_10(results):
    by_key = {(item["ttl"], item["mode"]): item for item in results}
    for mode in MODES:
        ttl8 = by_key[(8, mode)]
        failed_phase2 = (ttl8.get("failure_details") or {}).get("failed_ch_phase2") or {}
        if (
            ttl8["successful_rounds"] > by_key[(7, mode)]["successful_rounds"]
            and ttl8["status"] == "MRP_SEARCH_FAILURE"
            and failed_phase2.get("ttl_exhausted_sants", 0) > 0
        ):
            return True
    return False


def _aggregate(results, tested_ttls):
    by_key = {(item["ttl"], item["mode"]): item for item in results}
    best_pure = min(tested_ttls, key=lambda ttl: (-by_key[(ttl, "mrp_pure")]["successful_rounds"], ttl))
    best_hybrid = min(tested_ttls, key=lambda ttl: (-by_key[(ttl, "mrp_hybrid")]["successful_rounds"], ttl))
    best_common = min(tested_ttls, key=lambda ttl: (-min(
        by_key[(ttl, "mrp_pure")]["successful_rounds"],
        by_key[(ttl, "mrp_hybrid")]["successful_rounds"],
    ), ttl))
    return {
        "type": "MRP_TTL_LIFETIME_SWEEP_RESULTS", "status": "COMPLETE",
        "direct_reference_rounds": 2219, "ttl_candidates_tested": list(tested_ttls),
        "best_ttl_pure": best_pure, "best_ttl_hybrid": best_hybrid,
        "best_common_ttl": best_common, "trials": results,
    }


def _load_results(output, ttls):
    return [
        json.loads(_trial_path(output, ttl, mode).read_text(encoding="utf-8"))
        for ttl in ttls for mode in MODES
    ]


def _trial_path(output, ttl, mode):
    return output / "trials" / f"ttl_{ttl}" / f"{mode}_seed{SEED}.json"


def _validate_existing(path, ttl, mode):
    value = json.loads(path.read_text(encoding="utf-8"))
    actual = (value["ttl"], value["mode"], value["seed"], value["initial_state_hash"])
    expected = (ttl, mode, SEED, EXPECTED_STATE_HASH)
    if actual != expected or value.get("reserved_evaluation_seed_used") is not False:
        raise RuntimeError(f"staged trial identity mismatch: {path}")


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _print_progress(value):
    print(
        f"progress round={value['round']} live={value['live']} dead={value['dead']} "
        f"energy={value['cumulative_energy']:.12g} rss={value['current_rss_bytes']}",
        flush=True,
    )


def _now():
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
