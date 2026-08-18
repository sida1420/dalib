import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _module():
    spec = importlib.util.spec_from_file_location("ttl_sweep_under_test", ROOT / "run_ttl_lifetime_sweep.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _result(ttl, mode, rounds, *, status="MRP_SEARCH_FAILURE", exhausted=0):
    return {
        "ttl": ttl, "mode": mode, "successful_rounds": rounds, "status": status,
        "failure_details": {"failed_ch_phase2": {"ttl_exhausted_sants": exhausted}},
    }


def test_best_ttls_rank_lifetime_first_and_smaller_ttl_on_tie():
    module = _module()
    results = []
    for ttl, pure, hybrid in ((3, 10, 8), (4, 20, 12), (5, 20, 11)):
        results.extend((_result(ttl, "mrp_pure", pure), _result(ttl, "mrp_hybrid", hybrid)))
    aggregate = module._aggregate(results, (3, 4, 5))
    assert aggregate["best_ttl_pure"] == 4
    assert aggregate["best_ttl_hybrid"] == 4
    assert aggregate["best_common_ttl"] == 4


def test_ttl10_runs_only_when_ttl8_improves_same_mode_and_is_ttl_limited():
    module = _module()
    improved = [
        _result(7, "mrp_pure", 10), _result(7, "mrp_hybrid", 10),
        _result(8, "mrp_pure", 11, exhausted=3), _result(8, "mrp_hybrid", 9, exhausted=4),
    ]
    plateau = [
        _result(7, "mrp_pure", 10), _result(7, "mrp_hybrid", 10),
        _result(8, "mrp_pure", 10, exhausted=3), _result(8, "mrp_hybrid", 10, exhausted=4),
    ]
    assert module._should_test_ttl_10(improved) is True
    assert module._should_test_ttl_10(plateau) is False
