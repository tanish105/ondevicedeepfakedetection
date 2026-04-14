"""Phase 6 orchestrator: compute metrics, build tables, mine failures, write artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.analysis.failures import mine_nt_failures, render_failure_grid
from src.analysis.metrics import fp32_video_accuracy, frame_metrics
from src.analysis.tables import build_experiment_1, build_experiment_2, build_experiment_3
from src.common.constants import (
    PROJECT_ROOT,
    QUANT_DYNAMIC_RANGE,
    QUANT_FLOAT16,
    QUANT_FP32_GPU,
    QUANT_INT8_STATIC,
    TASK_DF_VS_REAL,
    TASK_NT_VS_REAL,
)
from src.common.csv_io import read_rows, write_rows
from src.common.schemas import (
    FailureMetadataRow,
    FramePredictionRow,
    VideoPredictionRow,
)

PREDS_DIR = PROJECT_ROOT / "artifacts" / "predictions"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
ANALYSIS_DIR = PROJECT_ROOT / "artifacts" / "analysis"

_TFLITE_QUANT_IDS = [QUANT_DYNAMIC_RANGE, QUANT_FLOAT16, QUANT_INT8_STATIC]


def _frame_csv(task_id: str, quant_id: str) -> Path:
    if quant_id == QUANT_FP32_GPU:
        return PREDS_DIR / f"{task_id}_fp32_test_frames.csv"
    return PREDS_DIR / f"{task_id}_{quant_id}_test_frames.csv"


def _video_csv(task_id: str, quant_id: str) -> Path:
    return PREDS_DIR / f"{task_id}_{quant_id}_test_videos.csv"


def _load_model_sizes() -> Dict[str, Dict[str, float]]:
    path = METRICS_DIR / "phase4_export_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    sizes: Dict[str, Dict[str, float]] = {}
    for task_id, info in data.items():
        try:
            sizes[task_id] = {QUANT_FP32_GPU: info["onnx_size_mb"]}
            for quant_id, tflite_info in info["tflite"].items():
                sizes[task_id][quant_id] = tflite_info["size_mb"]
        except KeyError as exc:
            raise KeyError(
                f"Unexpected structure in {path} for task '{task_id}': missing key {exc}"
            ) from exc
    return sizes


def _tflite_mean_fps(video_rows: List[VideoPredictionRow]) -> float:
    values = [r.fps for r in video_rows if r.fps > 0]
    return sum(values) / len(values) if values else 0.0


def build_all_metrics(model_sizes: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """Load all prediction CSVs and compute per-condition metrics dict."""
    all_metrics: Dict[str, Dict[str, Any]] = {}

    for task_id in (TASK_DF_VS_REAL, TASK_NT_VS_REAL):
        all_metrics[task_id] = {}

        # FP32 baseline (frame only, no video CSV)
        fp32_frame_rows = read_rows(_frame_csv(task_id, QUANT_FP32_GPU), FramePredictionRow)
        fm = frame_metrics(fp32_frame_rows)
        vid_acc, _ = fp32_video_accuracy(fp32_frame_rows)
        all_metrics[task_id][QUANT_FP32_GPU] = {
            **fm,
            "fps": 0.0,
            "model_size_mb": model_sizes[task_id][QUANT_FP32_GPU],
            "video_accuracy": vid_acc,
        }

        # TFLite variants
        for quant_id in _TFLITE_QUANT_IDS:
            frame_rows = read_rows(_frame_csv(task_id, quant_id), FramePredictionRow)
            video_rows = read_rows(_video_csv(task_id, quant_id), VideoPredictionRow)
            fm = frame_metrics(frame_rows)
            fps = _tflite_mean_fps(video_rows)
            vid_correct = sum(r.majority_vote_pred == r.binary_label for r in video_rows)
            vid_acc = vid_correct / len(video_rows) if video_rows else 0.0
            all_metrics[task_id][quant_id] = {
                **fm,
                "fps": fps,
                "model_size_mb": model_sizes[task_id][quant_id],
                "video_accuracy": vid_acc,
            }

    return all_metrics


def _print_summary(all_metrics: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("PHASE 6 METRICS SUMMARY")
    print("=" * 70)
    header = f"{'Task':<30} {'Quant':<25} {'AUC':>7} {'Acc':>7} {'FPS':>7} {'Gap':>7}"
    print(header)
    print("-" * 70)
    fp32_aucs = {task_id: all_metrics[task_id][QUANT_FP32_GPU]["auc"]
                 for task_id in (TASK_DF_VS_REAL, TASK_NT_VS_REAL)}
    for task_id in (TASK_DF_VS_REAL, TASK_NT_VS_REAL):
        fp32_auc = fp32_aucs[task_id]
        for quant_id in [QUANT_FP32_GPU] + _TFLITE_QUANT_IDS:
            m = all_metrics[task_id][quant_id]
            gap = fp32_auc - m["auc"] if quant_id != QUANT_FP32_GPU else 0.0
            print(f"{task_id:<30} {quant_id:<25} {m['auc']:>7.4f} {m['accuracy']:>7.4f} {m['fps']:>7.1f} {gap:>7.4f}")
    print("=" * 70)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    model_sizes = _load_model_sizes()

    print("Loading CSVs and computing metrics...")
    all_metrics = build_all_metrics(model_sizes)

    print("Building experiment tables...")
    exp1 = build_experiment_1(all_metrics)
    exp2 = build_experiment_2(all_metrics)
    exp3 = build_experiment_3(all_metrics)

    print("Mining NT failure frames...")
    # NT frame rows are re-loaded here because build_all_metrics discards raw rows
    # after computing metrics. Acceptable for a research script (7k rows, ~1s).
    fp32_nt_rows = read_rows(_frame_csv(TASK_NT_VS_REAL, QUANT_FP32_GPU), FramePredictionRow)
    int8_nt_rows = read_rows(_frame_csv(TASK_NT_VS_REAL, QUANT_INT8_STATIC), FramePredictionRow)
    failures = mine_nt_failures(fp32_nt_rows, int8_nt_rows, n=20)

    print(f"Found {len(failures)} NT failure candidates. Saving top 6 to grid...")
    write_rows(ANALYSIS_DIR / "nt_failure_frames.csv", failures, FailureMetadataRow)
    render_failure_grid(failures, frames_root=PROJECT_ROOT, output_path=ANALYSIS_DIR / "nt_failure_grid.png", n=6)

    # Write JSON outputs
    (ANALYSIS_DIR / "experiment_1.json").write_text(json.dumps(exp1, indent=2), encoding="utf-8")
    (ANALYSIS_DIR / "experiment_2.json").write_text(json.dumps(exp2, indent=2), encoding="utf-8")
    (ANALYSIS_DIR / "experiment_3.json").write_text(json.dumps(exp3, indent=2), encoding="utf-8")

    # Flatten all_metrics into phase6_summary.json
    summary: dict = {"conditions": {}}
    for task_id, quant_map in all_metrics.items():
        for quant_id, m in quant_map.items():
            summary["conditions"][f"{task_id}__{quant_id}"] = {"task_id": task_id, "quant_id": quant_id, **m}

    (ANALYSIS_DIR / "phase6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print_summary(all_metrics)
    print(f"\nAll outputs written to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
