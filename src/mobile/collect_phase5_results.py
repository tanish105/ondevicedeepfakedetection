"""Phase 5 post-processing: move pulled CSVs and compute summary stats.

Run after adb_pull_results.py has populated artifacts/predictions/results/.

Usage
-----
  python -m src.mobile.collect_phase5_results
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

# Import directly from src.common.constants (not the src.common package)
# to avoid pulling in src.common.runtime, which requires PyTorch at import time.
from src.common.constants import PROJECT_ROOT, TASK_IDS, QUANT_DYNAMIC_RANGE, QUANT_FLOAT16, QUANT_INT8_STATIC

TFLITE_QUANT_IDS = (QUANT_DYNAMIC_RANGE, QUANT_FLOAT16, QUANT_INT8_STATIC)

_DEFAULT_RESULTS_DIR = PROJECT_ROOT / "artifacts" / "predictions" / "results"
_DEFAULT_PREDS_DIR   = PROJECT_ROOT / "artifacts" / "predictions"
_DEFAULT_METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"


def collect_csvs(results_dir: Path, preds_dir: Path) -> None:
    """Move all CSVs from results_dir into preds_dir (one level up)."""
    for csv_file in results_dir.glob("*.csv"):
        dest = preds_dir / csv_file.name
        shutil.move(csv_file, dest)


def collect_device_specs(results_dir: Path, metrics_dir: Path) -> None:
    """Copy device_specs.json from results_dir to metrics_dir."""
    src = results_dir / "device_specs.json"
    if not src.exists():
        print("WARNING: device_specs.json not found in results dir — skipping.")
        return
    metrics_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, metrics_dir / "device_specs.json")


def compute_summary(preds_dir: Path) -> dict[str, dict[str, Any]]:
    """
    For each (task_id, quant_id) pair, compute mean_latency_ms, fps, num_frames
    from the frame-level CSV.

    Returns nested dict: summary[task_id][quant_id] = {mean_latency_ms, fps, num_frames}
    """
    summary: dict[str, dict[str, Any]] = {}

    for task_id in TASK_IDS:
        summary[task_id] = {}
        for quant_id in TFLITE_QUANT_IDS:
            csv_path = preds_dir / f"{task_id}_{quant_id}_test_frames.csv"
            if not csv_path.exists():
                print(f"  MISSING: {csv_path.name}")
                continue

            latencies: list[float] = []
            with csv_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    latencies.append(float(row["latency_ms"]))

            if not latencies:
                print(f"  EMPTY: {csv_path.name} (no data rows)")
                continue

            mean_lat = sum(latencies) / len(latencies)
            fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
            summary[task_id][quant_id] = {
                "mean_latency_ms": mean_lat,
                "fps": fps,
                "num_frames": len(latencies),
            }

    return summary


def main() -> None:
    results_dir = _DEFAULT_RESULTS_DIR
    preds_dir   = _DEFAULT_PREDS_DIR
    metrics_dir = _DEFAULT_METRICS_DIR

    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist.")
        print("Run `python scripts/adb_pull_results.py` first.")
        raise SystemExit(1)

    print("[1/3] Moving result CSVs to artifacts/predictions/ ...")
    collect_csvs(results_dir, preds_dir)
    print("  Done.")

    print("[2/3] Copying device_specs.json to artifacts/metrics/ ...")
    collect_device_specs(results_dir, metrics_dir)
    print("  Done.")

    print("[3/3] Computing per-variant latency summary ...")
    summary = compute_summary(preds_dir)

    summary_path = metrics_dir / "phase5_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Written: {summary_path}")

    print("\nPhase 5 post-processing complete.")
    print(f"  Predictions: {preds_dir}")
    print(f"  Metrics:     {metrics_dir}")


if __name__ == "__main__":
    main()
