"""Tests for src/analysis/run_phase6.py — _tflite_mean_fps and build_all_metrics."""
from __future__ import annotations

from typing import List
from unittest.mock import patch

import pytest

from src.analysis.run_phase6 import _tflite_mean_fps, build_all_metrics
from src.common.schemas import FramePredictionRow, VideoPredictionRow


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_video_row(video_id: str, fps: float, majority_vote_pred: int, binary_label: int,
                    task_id: str = "mobilenetv2_df_vs_real",
                    quant_id: str = "int8_static_tflite") -> VideoPredictionRow:
    return VideoPredictionRow(
        task_id=task_id,
        quant_id=quant_id,
        split="test",
        video_id=video_id,
        manipulation_type="df" if binary_label == 1 else "real",
        binary_label=binary_label,
        num_frames_used=25,
        mean_score_fake=0.8 if majority_vote_pred == 1 else 0.2,
        majority_vote_pred=majority_vote_pred,
        fps=fps,
        device_name="unknown",
        runtime="tflite_android",
    )


def _make_frame_row(video_id: str, frame_idx: int, binary_label: int, score_fake: float,
                    pred_label: int, task_id: str = "mobilenetv2_df_vs_real",
                    quant_id: str = "fp32_gpu") -> FramePredictionRow:
    return FramePredictionRow(
        task_id=task_id,
        quant_id=quant_id,
        split="test",
        video_id=video_id,
        frame_idx=frame_idx,
        image_path=f"data/processed/frames/df_vs_real/test/{video_id}/{frame_idx:04d}.png",
        manipulation_type="df" if binary_label == 1 else "real",
        binary_label=binary_label,
        score_fake=score_fake,
        pred_label=pred_label,
        latency_ms=0.0,
        device_name="unknown",
        runtime="pytorch_fp32" if quant_id == "fp32_gpu" else "tflite_android",
    )


# ── _tflite_mean_fps ─────────────────────────────────────────────────────────

def test_tflite_mean_fps_basic():
    rows = [
        _make_video_row("v1", 10.0, 1, 1),
        _make_video_row("v2", 20.0, 0, 0),
    ]
    assert _tflite_mean_fps(rows) == pytest.approx(15.0)


def test_tflite_mean_fps_skips_zero_fps():
    rows = [
        _make_video_row("v1", 0.0, 1, 1),  # fps=0 → excluded
        _make_video_row("v2", 12.0, 0, 0),
    ]
    assert _tflite_mean_fps(rows) == pytest.approx(12.0)


def test_tflite_mean_fps_empty_returns_zero():
    assert _tflite_mean_fps([]) == pytest.approx(0.0)


def test_tflite_mean_fps_all_zero_returns_zero():
    rows = [_make_video_row("v1", 0.0, 1, 1), _make_video_row("v2", 0.0, 0, 0)]
    assert _tflite_mean_fps(rows) == pytest.approx(0.0)


# ── build_all_metrics ─────────────────────────────────────────────────────────

_TASKS = ["mobilenetv2_df_vs_real", "mobilenetv2_nt_vs_real"]
_TFLITE_QUANTS = ["dynamic_range_tflite", "float16_tflite", "int8_static_tflite"]


def _minimal_model_sizes() -> dict:
    return {
        task: {"fp32_gpu": 8.47, "dynamic_range_tflite": 2.38,
               "float16_tflite": 4.26, "int8_static_tflite": 2.58}
        for task in _TASKS
    }


def _make_frame_rows_two_class(task_id: str, quant_id: str) -> List[FramePredictionRow]:
    """Minimal frame rows: 2 fake + 2 real, all correct predictions."""
    rows = []
    for i in range(2):
        rows.append(_make_frame_row(f"v{i}", 0, 1, 0.9, 1, task_id, quant_id))
    for i in range(2, 4):
        rows.append(_make_frame_row(f"v{i}", 0, 0, 0.1, 0, task_id, quant_id))
    return rows


def _make_video_rows(task_id: str, quant_id: str) -> List[VideoPredictionRow]:
    return [
        _make_video_row("v0", 10.0, 1, 1, task_id, quant_id),
        _make_video_row("v1", 12.0, 1, 1, task_id, quant_id),
        _make_video_row("v2", 11.0, 0, 0, task_id, quant_id),
        _make_video_row("v3", 13.0, 0, 0, task_id, quant_id),
    ]


def test_build_all_metrics_returns_expected_structure():
    def _side_effect(path, row_cls):
        path_str = str(path)
        task_id = next((t for t in _TASKS if t in path_str), _TASKS[0])
        if row_cls is FramePredictionRow:
            quant_id = "fp32_gpu" if "fp32" in path_str else next(
                (q for q in _TFLITE_QUANTS if q in path_str), _TFLITE_QUANTS[0]
            )
            return _make_frame_rows_two_class(task_id, quant_id)
        # VideoPredictionRow
        quant_id = next((q for q in _TFLITE_QUANTS if q in path_str), _TFLITE_QUANTS[0])
        return _make_video_rows(task_id, quant_id)

    with patch("src.analysis.run_phase6.read_rows", side_effect=_side_effect):
        result = build_all_metrics(_minimal_model_sizes())

    assert set(result.keys()) == set(_TASKS)
    for task_id in _TASKS:
        assert set(result[task_id].keys()) == {"fp32_gpu"} | set(_TFLITE_QUANTS)
        for condition in result[task_id].values():
            assert {"auc", "accuracy", "num_frames", "fps", "model_size_mb", "video_accuracy"} <= set(condition.keys())


def test_build_all_metrics_fp32_fps_is_zero():
    def _side_effect(path, row_cls):
        path_str = str(path)
        task_id = next((t for t in _TASKS if t in path_str), _TASKS[0])
        if row_cls is FramePredictionRow:
            quant_id = "fp32_gpu" if "fp32" in path_str else next(
                (q for q in _TFLITE_QUANTS if q in path_str), _TFLITE_QUANTS[0]
            )
            return _make_frame_rows_two_class(task_id, quant_id)
        quant_id = next((q for q in _TFLITE_QUANTS if q in path_str), _TFLITE_QUANTS[0])
        return _make_video_rows(task_id, quant_id)

    with patch("src.analysis.run_phase6.read_rows", side_effect=_side_effect):
        result = build_all_metrics(_minimal_model_sizes())

    for task_id in _TASKS:
        assert result[task_id]["fp32_gpu"]["fps"] == 0.0


def test_build_all_metrics_model_sizes_wired():
    def _side_effect(path, row_cls):
        path_str = str(path)
        task_id = next((t for t in _TASKS if t in path_str), _TASKS[0])
        if row_cls is FramePredictionRow:
            quant_id = "fp32_gpu" if "fp32" in path_str else next(
                (q for q in _TFLITE_QUANTS if q in path_str), _TFLITE_QUANTS[0]
            )
            return _make_frame_rows_two_class(task_id, quant_id)
        quant_id = next((q for q in _TFLITE_QUANTS if q in path_str), _TFLITE_QUANTS[0])
        return _make_video_rows(task_id, quant_id)

    sizes = _minimal_model_sizes()
    with patch("src.analysis.run_phase6.read_rows", side_effect=_side_effect):
        result = build_all_metrics(sizes)

    for task_id in _TASKS:
        for quant_id in ["fp32_gpu"] + _TFLITE_QUANTS:
            assert result[task_id][quant_id]["model_size_mb"] == sizes[task_id][quant_id]
