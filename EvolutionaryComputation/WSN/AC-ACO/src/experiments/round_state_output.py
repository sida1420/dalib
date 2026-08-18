"""Immediate console and CSV output for manual simulation rounds."""

import csv
from pathlib import Path


ROUND_STATE_FIELDS = ("round", "alive_nodes", "round_energy")


def available_output_path(path: Path) -> Path:
    """Keep earlier diagnostic JSON while allowing README commands to be rerun."""

    if not path.exists():
        return path
    run_number = 2
    while True:
        candidate = path.with_name(f"{path.stem}_run{run_number}{path.suffix}")
        if not candidate.exists():
            return candidate
        run_number += 1


class RoundStateWriter:
    """Replace one run's CSV and flush every completed round immediately."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None
        self._writer = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=ROUND_STATE_FIELDS)
        self._writer.writeheader()
        self._handle.flush()
        return self

    def write(self, progress: dict[str, object]) -> None:
        row = {field: progress[field] for field in ROUND_STATE_FIELDS}
        self._writer.writerow(row)
        self._handle.flush()

    def __exit__(self, *_error_info):
        if self._handle is not None:
            self._handle.close()
