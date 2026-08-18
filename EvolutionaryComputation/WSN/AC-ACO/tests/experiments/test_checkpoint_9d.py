from pathlib import Path
import sys

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments import load_checkpoint_9d_freeze
from experiments.checkpoint_9d import Checkpoint9DError
from experiments.config import ExperimentMode
from experiments.final_evaluation_reporting import IncrementalFinalEvaluationWriter
from experiments.final_evaluation_types import FNDObservation, NativeEvaluationTrial
from experiments.types import LifetimeMetrics, ModeTrialResult, RoundMetrics


ROOT = Path(__file__).resolve().parents[2]
SELECTED = ROOT / "real_calibration_v2" / "calibration_outputs" / "selected_config.json"
SCENARIOS = ROOT / "real_calibration_v2" / "scenario_manifest.json"


def test_checkpoint_9d_is_stale_after_network_wide_mrp_semantics_change():
    with pytest.raises(Checkpoint9DError, match="STALE_BY_NETWORK_WIDE_MRP_ADAPTATION"):
        load_checkpoint_9d_freeze(ROOT, SELECTED, SCENARIOS)


def test_incremental_writer_persists_one_trial_then_releases_its_history(tmp_path):
    round_row = _round_row()
    trial = NativeEvaluationTrial(
        ExperimentMode.BASELINE, 33001, None, None, None,
        ModeTrialResult(ExperimentMode.BASELINE, (round_row,), (object(),), LifetimeMetrics(None, None, None, False, False, False)),
        FNDObservation(None, None, False, 1, False),
    )
    writer = IncrementalFinalEvaluationWriter(None, tmp_path / "final")
    writer.append(trial)
    assert len(writer._digests) == 1
    assert writer._digests[0]["final"] is round_row
    assert (writer.stage / "round_metrics.csv").is_file()
    writer.abort()
    assert not (tmp_path / "final").exists()
    assert (writer.stage / "run_state.json").is_file()


def _round_row():
    return RoundMetrics(
        ExperimentMode.BASELINE, 0, 33001, 0, (0,), (0,), 0, (), None, 1, 1, 1.0,
        (0,), ((0, -1),), {0: 1}, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.0, 1.0, 1.0,
        (1,), (1.0,), None, None, True, None, None,
    )
