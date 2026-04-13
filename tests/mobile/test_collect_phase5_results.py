"""Tests for src/mobile/collect_phase5_results.py"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.mobile.collect_phase5_results import (
    collect_csvs,
    collect_device_specs,
    compute_summary,
)


@pytest.fixture()
def tmp_layout(tmp_path: Path):
    """Create a minimal artifacts/ layout inside tmp_path."""
    results_dir = tmp_path / "artifacts" / "predictions" / "results"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    preds_dir   = tmp_path / "artifacts" / "predictions"
    results_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    return tmp_path, results_dir, metrics_dir, preds_dir


def _write_frame_csv(path: Path, rows: list[dict]) -> None:
    header = [
        "task_id", "quant_id", "split", "video_id", "frame_idx",
        "image_path", "manipulation_type", "binary_label",
        "score_fake", "pred_label", "latency_ms", "device_name", "runtime",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def test_collect_csvs_moves_files(tmp_layout):
    tmp, results_dir, _, preds_dir = tmp_layout
    for name in [
        "mobilenetv2_df_vs_real_dynamic_range_tflite_test_frames.csv",
        "mobilenetv2_df_vs_real_dynamic_range_tflite_test_videos.csv",
    ]:
        (results_dir / name).write_text("task_id\nfoo")

    collect_csvs(results_dir, preds_dir)

    assert (preds_dir / "mobilenetv2_df_vs_real_dynamic_range_tflite_test_frames.csv").exists()
    assert (preds_dir / "mobilenetv2_df_vs_real_dynamic_range_tflite_test_videos.csv").exists()
    assert not (results_dir / "mobilenetv2_df_vs_real_dynamic_range_tflite_test_frames.csv").exists()


def test_collect_device_specs_copies_json(tmp_layout):
    tmp, results_dir, metrics_dir, _ = tmp_layout
    specs = {"device_name": "Pixel 6", "android_api_level": 33}
    (results_dir / "device_specs.json").write_text(json.dumps(specs))

    collect_device_specs(results_dir, metrics_dir)

    dest = metrics_dir / "device_specs.json"
    assert dest.exists()
    assert json.loads(dest.read_text())["device_name"] == "Pixel 6"


def test_compute_summary_mean_latency(tmp_layout):
    tmp, results_dir, metrics_dir, preds_dir = tmp_layout
    rows = [
        {
            "task_id": "mobilenetv2_df_vs_real",
            "quant_id": "dynamic_range_tflite",
            "split": "test",
            "video_id": "1",
            "frame_idx": 0,
            "image_path": "p",
            "manipulation_type": "real",
            "binary_label": 0,
            "score_fake": 0.1,
            "pred_label": 0,
            "latency_ms": 20.0,
            "device_name": "Pixel 6",
            "runtime": "tflite_android",
        },
        {
            "task_id": "mobilenetv2_df_vs_real",
            "quant_id": "dynamic_range_tflite",
            "split": "test",
            "video_id": "1",
            "frame_idx": 1,
            "image_path": "p2",
            "manipulation_type": "real",
            "binary_label": 0,
            "score_fake": 0.2,
            "pred_label": 0,
            "latency_ms": 40.0,
            "device_name": "Pixel 6",
            "runtime": "tflite_android",
        },
    ]
    csv_path = preds_dir / "mobilenetv2_df_vs_real_dynamic_range_tflite_test_frames.csv"
    _write_frame_csv(csv_path, rows)

    summary = compute_summary(preds_dir)

    entry = summary["mobilenetv2_df_vs_real"]["dynamic_range_tflite"]
    assert entry["mean_latency_ms"] == pytest.approx(30.0)
    assert entry["fps"] == pytest.approx(1000.0 / 30.0)
    assert entry["num_frames"] == 2


def test_compute_summary_missing_csv(tmp_layout):
    tmp, results_dir, metrics_dir, preds_dir = tmp_layout
    # preds_dir has no CSVs at all

    summary = compute_summary(preds_dir)

    # The returned dict has both task_id keys but no quant_id sub-entries
    assert "mobilenetv2_df_vs_real" in summary
    assert "mobilenetv2_nt_vs_real" in summary
    assert summary["mobilenetv2_df_vs_real"] == {}
    assert summary["mobilenetv2_nt_vs_real"] == {}
