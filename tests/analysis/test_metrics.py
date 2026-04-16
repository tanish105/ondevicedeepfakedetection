"""Tests for src/analysis/metrics.py"""
from __future__ import annotations

import pytest

from src.analysis.metrics import frame_metrics, fp32_video_accuracy
from src.common.schemas import FramePredictionRow


def _make_frame_row(
    video_id: str,
    frame_idx: int,
    manipulation_type: str,
    binary_label: int,
    score_fake: float,
    pred_label: int,
    quant_id: str = "fp32_gpu",
    task_id: str = "mobilenetv2_df_vs_real",
) -> FramePredictionRow:
    return FramePredictionRow(
        task_id=task_id,
        quant_id=quant_id,
        split="test",
        video_id=video_id,
        frame_idx=frame_idx,
        image_path=f"data/processed/frames/df_vs_real/test/{video_id}/{frame_idx:04d}.png",
        manipulation_type=manipulation_type,
        binary_label=binary_label,
        score_fake=score_fake,
        pred_label=pred_label,
        latency_ms=0.0,
        device_name="unknown",
        runtime="pytorch_fp32",
    )


def test_frame_metrics_perfect_predictions():
    rows = [
        _make_frame_row("v1", 0, "real", 0, 0.05, 0),
        _make_frame_row("v1", 1, "df", 1, 0.95, 1),
        _make_frame_row("v2", 0, "real", 0, 0.10, 0),
        _make_frame_row("v2", 1, "df", 1, 0.90, 1),
    ]
    result = frame_metrics(rows)
    assert result["auc"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["num_frames"] == 4


def test_frame_metrics_imperfect_predictions():
    # 3 correct, 1 wrong → accuracy = 0.75
    rows = [
        _make_frame_row("v1", 0, "real", 0, 0.1, 0),
        _make_frame_row("v1", 1, "df", 1, 0.9, 1),
        _make_frame_row("v2", 0, "real", 0, 0.8, 1),  # wrong pred
        _make_frame_row("v2", 1, "df", 1, 0.6, 1),
    ]
    result = frame_metrics(rows)
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["num_frames"] == 4
    assert 0.0 < result["auc"] < 1.0


def test_frame_metrics_num_frames():
    rows = [_make_frame_row("v1", i, "real", 0, 0.1, 0) for i in range(5)]
    rows.append(_make_frame_row("v1", 5, "df", 1, 0.9, 1))
    result = frame_metrics(rows)
    assert result["num_frames"] == 6


def test_frame_metrics_raises_on_single_class():
    rows = [_make_frame_row("v1", i, "real", 0, 0.1, 0) for i in range(4)]
    with pytest.raises(ValueError, match="both classes present"):
        frame_metrics(rows)


def test_fp32_video_accuracy_all_correct():
    rows = (
        [_make_frame_row("v1", i, "df", 1, 0.9, 1) for i in range(3)]
        + [_make_frame_row("v2", i, "real", 0, 0.1, 0) for i in range(3)]
    )
    accuracy, num_videos = fp32_video_accuracy(rows)
    assert accuracy == pytest.approx(1.0)
    assert num_videos == 2


def test_fp32_video_accuracy_tie_break_predicts_real():
    # Equal votes (1 fake, 1 real pred) → tie-break resolves to real (0)
    # binary_label=1 (fake), so predicting real is wrong → accuracy=0.0
    rows = [
        _make_frame_row("v1", 0, "df", 1, 0.9, 1),  # pred=1
        _make_frame_row("v1", 1, "df", 1, 0.2, 0),  # pred=0 — tie
    ]
    accuracy, num_videos = fp32_video_accuracy(rows)
    assert accuracy == pytest.approx(0.0)
    assert num_videos == 1


def test_fp32_video_accuracy_majority_vote():
    # video v1: pred=[1,1,0], label=1 → majority=1 → correct
    # video v2: pred=[0,0,1], label=1 → majority=0 → wrong
    rows = (
        [_make_frame_row("v1", 0, "df", 1, 0.9, 1),
         _make_frame_row("v1", 1, "df", 1, 0.8, 1),
         _make_frame_row("v1", 2, "df", 1, 0.3, 0)]
        + [_make_frame_row("v2", 0, "df", 1, 0.4, 0),
           _make_frame_row("v2", 1, "df", 1, 0.3, 0),
           _make_frame_row("v2", 2, "df", 1, 0.8, 1)]
    )
    accuracy, num_videos = fp32_video_accuracy(rows)
    assert accuracy == pytest.approx(0.5)
    assert num_videos == 2
