"""Tests for src/analysis/failures.py"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.failures import mine_nt_failures, render_failure_grid
from src.common.schemas import FailureMetadataRow, FramePredictionRow


def _make_nt_frame_row(
    video_id: str,
    frame_idx: int,
    binary_label: int,
    score_fake: float,
    pred_label: int,
    quant_id: str,
) -> FramePredictionRow:
    return FramePredictionRow(
        task_id="mobilenetv2_nt_vs_real",
        quant_id=quant_id,
        split="test",
        video_id=video_id,
        frame_idx=frame_idx,
        image_path=f"data/processed/frames/nt_vs_real/test/{video_id}/{frame_idx:04d}.png",
        manipulation_type="nt" if binary_label == 1 else "real",
        binary_label=binary_label,
        score_fake=score_fake,
        pred_label=pred_label,
        latency_ms=0.0,
        device_name="unknown",
        runtime="pytorch_fp32" if quant_id == "fp32_gpu" else "tflite_android",
    )


def test_mine_nt_failures_finds_disagreements():
    # frame 0: fp32 correct (fake, pred=1), int8 wrong (pred=0) → failure
    # frame 1: both correct → not a failure
    fp32_rows = [
        _make_nt_frame_row("v1", 0, 1, 0.95, 1, "fp32_gpu"),  # correct
        _make_nt_frame_row("v1", 1, 1, 0.90, 1, "fp32_gpu"),  # correct
    ]
    int8_rows = [
        _make_nt_frame_row("v1", 0, 1, 0.30, 0, "int8_static_tflite"),  # wrong → failure
        _make_nt_frame_row("v1", 1, 1, 0.85, 1, "int8_static_tflite"),  # correct
    ]
    failures = mine_nt_failures(fp32_rows, int8_rows)
    assert len(failures) == 1
    assert failures[0].video_id == "v1"
    assert failures[0].frame_idx == 0


def test_mine_nt_failures_sorted_by_delta():
    # Two failures: frame 0 has larger delta than frame 1
    fp32_rows = [
        _make_nt_frame_row("v1", 0, 1, 0.95, 1, "fp32_gpu"),
        _make_nt_frame_row("v1", 1, 1, 0.70, 1, "fp32_gpu"),
    ]
    int8_rows = [
        _make_nt_frame_row("v1", 0, 1, 0.10, 0, "int8_static_tflite"),
        _make_nt_frame_row("v1", 1, 1, 0.20, 0, "int8_static_tflite"),
    ]
    failures = mine_nt_failures(fp32_rows, int8_rows)
    assert len(failures) == 2
    # delta[0] = 0.95 - 0.10 = 0.85 > delta[1] = 0.70 - 0.20 = 0.50
    assert failures[0].frame_idx == 0
    assert failures[1].frame_idx == 1


def test_mine_nt_failures_skips_real_frames():
    # real frames (binary_label=0) should never appear as failures
    fp32_rows = [_make_nt_frame_row("v1", 0, 0, 0.05, 0, "fp32_gpu")]
    int8_rows = [_make_nt_frame_row("v1", 0, 0, 0.60, 1, "int8_static_tflite")]
    failures = mine_nt_failures(fp32_rows, int8_rows)
    assert failures == []


def test_mine_nt_failures_respects_n_limit():
    fp32_rows = [_make_nt_frame_row("v1", i, 1, 0.9, 1, "fp32_gpu") for i in range(10)]
    int8_rows = [_make_nt_frame_row("v1", i, 1, 0.1, 0, "int8_static_tflite") for i in range(10)]
    failures = mine_nt_failures(fp32_rows, int8_rows, n=4)
    assert len(failures) == 4


def test_mine_nt_failures_returns_failure_metadata_rows():
    fp32_rows = [_make_nt_frame_row("v1", 0, 1, 0.95, 1, "fp32_gpu")]
    int8_rows = [_make_nt_frame_row("v1", 0, 1, 0.10, 0, "int8_static_tflite")]
    failures = mine_nt_failures(fp32_rows, int8_rows)
    assert isinstance(failures[0], FailureMetadataRow)
    assert failures[0].fp32_score_fake == pytest.approx(0.95)
    assert failures[0].int8_score_fake == pytest.approx(0.10)
    assert failures[0].selection_reason == "fp32_correct_int8_wrong"


def test_mine_nt_failures_excludes_fp32_also_wrong():
    # fp32 wrong (pred_label=0) → frame should not appear in failures even if int8 is also wrong
    fp32_rows = [_make_nt_frame_row("v1", 0, 1, 0.40, 0, "fp32_gpu")]
    int8_rows = [_make_nt_frame_row("v1", 0, 1, 0.20, 0, "int8_static_tflite")]
    failures = mine_nt_failures(fp32_rows, int8_rows)
    assert failures == []


def test_render_failure_grid_creates_png(tmp_path: Path):
    # Create a tiny 224×224 black PNG at the expected image path
    Image = pytest.importorskip("PIL.Image")
    img_dir = tmp_path / "data" / "processed" / "frames" / "nt_vs_real" / "test" / "v1"
    img_dir.mkdir(parents=True)
    for i in range(6):
        Image.new("RGB", (224, 224), (0, 0, 0)).save(img_dir / f"{i:04d}.png")  # type: ignore[attr-defined]

    failures = [
        FailureMetadataRow(
            video_id="v1",
            frame_idx=i,
            image_path=f"data/processed/frames/nt_vs_real/test/v1/{i:04d}.png",
            fp32_score_fake=0.9 - i * 0.05,
            int8_score_fake=0.1 + i * 0.02,
            binary_label=1,
            selection_reason="fp32_correct_int8_wrong",
        )
        for i in range(6)
    ]
    output_path = tmp_path / "grid.png"
    render_failure_grid(failures, frames_root=tmp_path, output_path=output_path, n=6)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
