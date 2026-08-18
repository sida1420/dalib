"""Run exactly one manual AC-ACO/MRP diagnostic mode per process invocation."""

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from experiments.manual_runner import (  # noqa: E402
    DEFAULT_ANCHOR,
    DEFAULT_SAFETY_MAX_ROUNDS,
    DEFAULT_SCENARIO_ID,
    DEFAULT_SEED,
    ManualRunnerError,
    normalize_manual_mode,
    run_manual_route_exhaustion,
)
from experiments.native_case import NATIVE_CASE_ID  # noqa: E402
from experiments.round_state_output import (  # noqa: E402
    ROUND_STATE_FIELDS,
    RoundStateWriter,
    available_output_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exactly one manual route-exhaustion diagnostic method.",
        epilog=(
            "Examples:\n"
            "  python run.py --mode direct\n"
            "  python run.py --mode mrp_pure --seed 33001 --scenario calibration-v1-c1 --anchor 49\n"
            "  python run.py --mode mrp_hybrid --seed 33001 --scenario calibration-v1-c1 --anchor 49\n"
            "  python run.py --mode direct_mrp_topk --native-case --seed 33001 --ttl 3 --phase3-top-k 3\n"
            "  python run.py --mode mrp_hybrid_lifetime --native-case --seed 33001"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True,
        help="direct, mrp_pure, mrp_hybrid, direct_mrp_topk, mrp_hybrid_lifetime (experimental), or mrp_event",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"diagnostic seed (default: {DEFAULT_SEED})")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO_ID, help=f"frozen calibration scenario (default: {DEFAULT_SCENARIO_ID})")
    parser.add_argument("--anchor", type=int, default=DEFAULT_ANCHOR, help=f"frozen scenario anchor (default: {DEFAULT_ANCHOR})")
    parser.add_argument(
        "--ttl", type=int, default=None,
        help="runtime-only MRP TTL override; canonical configuration is not modified",
    )
    parser.add_argument(
        "--native-case", action="store_true",
        help="use the original map.pkl network-wide case with no event scenario/anchor",
    )
    parser.add_argument(
        "--output-directory", type=Path, default=None,
        help="diagnostic output root (default: run_outputs under repository root)",
    )
    parser.add_argument(
        "--safety-max-rounds", type=int, default=DEFAULT_SAFETY_MAX_ROUNDS,
        help=f"operational guard, not a routing stop policy (default: {DEFAULT_SAFETY_MAX_ROUNDS})",
    )
    parser.add_argument(
        "--round-state-file", type=Path, default=None,
        help="live per-round CSV (default: round_state.csv under repository root)",
    )
    parser.add_argument(
        "--mrp-search-retries", type=int, default=3,
        help="extra attempts for transient MRP search failure in the same round (default: 3)",
    )
    parser.add_argument(
        "--phase3-top-k", type=int, default=None,
        help="Phase-III candidate limit for Hybrid or direct_mrp_topk; the new mode requires K > 0",
    )
    parser.add_argument(
        "--charge-ant-energy", action="store_true",
        help="charge optional physical SANT/BANT/AANT radio energy (default: off)",
    )
    parser.add_argument(
        "--ant-control-packet-bits", type=int, default=None,
        help="explicit control-ant packet size in bits; required when charging ants",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        mode = normalize_manual_mode(args.mode)
    except ManualRunnerError:
        parser.error("Invalid mode.\n\nValid modes:\n    direct\n    mrp_pure\n    mrp_hybrid\n    direct_mrp_topk\n    mrp_hybrid_lifetime\n    mrp_event")
    _validate_ant_energy_cli(parser, args, mode)
    scenario_id = NATIVE_CASE_ID if args.native_case else args.scenario
    output_root = (
        args.output_directory.resolve()
        if args.output_directory is not None else ROOT / "run_outputs"
    )
    output_path = _output_path(output_root, mode, args.seed, scenario_id)
    output_path = available_output_path(output_path)
    round_state_path = (
        args.round_state_file.resolve()
        if args.round_state_file is not None else ROOT / "round_state.csv"
    )
    _print_header(mode, args)
    try:
        print(",".join(ROUND_STATE_FIELDS), flush=True)
        with RoundStateWriter(round_state_path) as round_state:
            result = run_manual_route_exhaustion(
                ROOT,
                mode=mode,
                seed=args.seed,
                scenario_id=args.scenario,
                anchor=args.anchor,
                safety_max_rounds=args.safety_max_rounds,
                native_case=args.native_case,
                ttl_override=args.ttl,
                phase3_top_k=args.phase3_top_k,
                charge_ant_energy=args.charge_ant_energy,
                ant_control_packet_bits=args.ant_control_packet_bits,
                mrp_search_retries=args.mrp_search_retries,
                progress=lambda record: _record_progress(round_state, record),
            )
    except ManualRunnerError as error:
        parser.error(str(error))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _print_result(result, output_path)
    return 0 if result["status"] in {
        "DIRECT_ROUTE_EXHAUSTED", "MRP_EVENT_ROUTE_EXHAUSTED", "MRP_HYBRID_ROUTE_EXHAUSTED",
        "DIRECT_MRP_TOPK_ROUTE_EXHAUSTED",
        "AC_ACO_SELECTION_FAILURE", "GREEDY_ROUTING_FAILURE",
        "MRP_CLUSTERING_EXHAUSTED", "MRP_SEARCH_FAILURE",
        "PHYSICAL_CONNECTIVITY_EXHAUSTED", "SAFETY_CAP_REACHED",
    } else 1


def _validate_ant_energy_cli(parser, args, mode) -> None:
    """Reject invalid ant accounting before opening any output artifact."""

    bits = args.ant_control_packet_bits
    if bits is not None and bits <= 0:
        parser.error("--ant-control-packet-bits must be a positive integer")
    if args.charge_ant_energy and bits is None:
        parser.error("--ant-control-packet-bits is required with --charge-ant-energy")
    if bits is not None and not args.charge_ant_energy:
        parser.error("--ant-control-packet-bits requires --charge-ant-energy")
    if args.charge_ant_energy and mode == "mrp_hybrid_lifetime":
        parser.error("ant energy is not supported by the experimental lifetime selector")


def _output_path(output_root: Path, mode: str, seed: int, scenario: str) -> Path:
    scenario_token = "".join(character if character.isalnum() or character in "._-" else "_" for character in scenario)
    suffix = f"seed{seed}" if mode == "direct" else f"seed{seed}_{scenario_token}"
    return output_root / mode / f"{mode}_{suffix}.json"


def _print_header(mode: str, args) -> None:
    labels = {
        "direct": "DIRECT", "mrp_pure": "MRP PURE — NETWORK WIDE",
        "mrp_hybrid": "MRP HYBRID", "mrp_event": "MRP EVENT — PAPER REFERENCE",
        "direct_mrp_topk": "AC-ACO + DIRECT-FIRST + MRP-FALLBACK + TOP-K",
        "mrp_hybrid_lifetime": "MRP HYBRID LIFETIME — EXPERIMENTAL",
    }
    print("=" * 46)
    print(f"METHOD: {labels[mode]}")
    print(f"PID: {os.getpid()}")
    if args.native_case:
        print(f"CASE: {NATIVE_CASE_ID}")
        print("SOURCE: repository map.pkl / legacy AC-ACO configuration")
        print("EVENT INPUT: NONE")
    elif mode != "direct":
        print(f"SCENARIO: {args.scenario}")
        print(f"ANCHOR: {args.anchor}")
    print(f"SEED: {args.seed}")
    if args.ttl is not None:
        print(f"TTL OVERRIDE: {args.ttl} (runtime only)")
    if args.phase3_top_k is not None:
        print(f"PHASE-III TOP-K: {args.phase3_top_k or 'ALL'}")
    print(f"ANT ENERGY: {'ON' if args.charge_ant_energy else 'OFF'}")
    if args.ant_control_packet_bits is not None:
        print(f"ANT CONTROL PACKET: {args.ant_control_packet_bits} bits")
    print("STOP POLICY: ROUTE EXHAUSTION")
    print(f"SAFETY CAP: {args.safety_max_rounds}")
    print("=" * 46, flush=True)


def _record_progress(writer: RoundStateWriter, progress: dict[str, object]) -> None:
    writer.write(progress)
    print(
        f"{progress['round']},{progress['alive_nodes']},"
        f"{progress['round_energy']:.12g}", flush=True,
    )


def _print_result(result: dict[str, object], output_path: Path) -> None:
    print("\nRESULT")
    for label, key in (
        ("METHOD", "mode"), ("SEED", "seed"), ("SCENARIO", "scenario_id"), ("STATUS", "status"),
        ("SUCCESSFUL ROUNDS", "successful_rounds"), ("FND ROUND (0/1)", None),
        ("HND ROUND (0/1)", None), ("LND ROUND (0/1)", None),
        ("LAST SUCCESSFUL ROUND (0/1)", None), ("ROUTING FAILURE ROUND (0/1)", None),
        ("ROUTING FAILURE REASON", "failure_reason"), ("FINAL LIVE NODES", "final_live_nodes"),
        ("FINAL DEAD NODES", "final_dead_nodes"), ("CUMULATIVE PHYSICAL ENERGY", "physical_energy"),
        ("DATA ENERGY", "data_energy"),
        ("ANT CONTROL ENERGY", "total_ant_control_energy"),
        ("SANT ENERGY", "sant_energy"), ("BANT ENERGY", "bant_energy"),
        ("AANT ENERGY", "aant_energy"),
        ("SANT SENSOR TX", "sant_tx_count"), ("SANT SENSOR RX", "sant_rx_count"),
        ("BANT SENSOR TX", "bant_tx_count"), ("BANT SENSOR RX", "bant_rx_count"),
        ("AANT SENSOR TX", "aant_tx_count"), ("AANT SENSOR RX", "aant_rx_count"),
        ("DELIVERED PAYLOADS", "delivered_payloads"), ("ENERGY/PAYLOAD", "energy_per_delivered_payload"),
        ("MRP SEARCH RETRIES", "mrp_search_retries"),
        ("CONFIGURED TOP-K", "configured_top_k"),
        ("MEAN ROUTES BEFORE PRUNING", "mean_routes_before_pruning"),
        ("MEAN ROUTES AFTER PRUNING", "mean_routes_after_pruning"),
        ("DIRECT CH DECISIONS", "direct_ch_count"),
        ("FALLBACK CH DECISIONS", "fallback_ch_count"),
        ("DIRECT TRANSMISSIONS", "direct_transmissions"),
        ("FALLBACK TRANSMISSIONS", "fallback_transmissions"),
        ("DIRECT SUCCESSES", "direct_successes"),
        ("FALLBACK SUCCESSES", "fallback_successes"),
        ("FALLBACK FAILURES", "fallback_failures"),
        ("DIRECT RATIO", "direct_ratio"),
        ("FALLBACK RATIO", "fallback_ratio"),
        ("MRP DISCOVERIES", "mrp_discoveries"),
        ("CACHED ROUTE USES", "cached_route_uses"),
        ("ROUTES BEFORE TOP-K", "routes_before_topk"),
        ("ROUTES AFTER TOP-K", "routes_after_topk"),
        ("ROUTES PRUNED", "routes_pruned"),
        ("SANT / BANT / AANT", None),
        ("SANT COUNT", "sant_count"),
        ("BANT COUNT", "bant_count"),
        ("AANT COUNT", "aant_count"),
        ("ROUTING FAILURES", "routing_failures"),
        ("DROPPED PACKETS", "dropped_packets"),
        ("PEAK RSS BYTES", "peak_rss_bytes"),
    ):
        if key is not None and key in result:
            print(f"{label}: {result.get(key)}")
    for milestone in ("fnd", "hnd", "lnd"):
        print(f"{milestone.upper()} ROUND (0/1): {result[f'{milestone}_round_zero_based']} / {result[f'{milestone}_round_one_based']}")
    print(
        "LAST SUCCESSFUL ROUND (0/1): "
        f"{result['last_successful_round_zero_based']} / {result['last_successful_round_one_based']}"
    )
    print(
        "ROUTING FAILURE ROUND (0/1): "
        f"{result['routing_failure_round_zero_based']} / {result['routing_failure_round_one_based']}"
    )
    if result["last_success_details"] is not None:
        print("LAST SUCCESS DETAILS:")
        print(json.dumps(result["last_success_details"], indent=2, sort_keys=True))
    if result["failure_details"] is not None:
        print("FAILURE DETAILS:")
        print(json.dumps(result["failure_details"], indent=2, sort_keys=True))
    print(f"RESULT FILE: {output_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
