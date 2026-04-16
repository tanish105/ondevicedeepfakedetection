"""Tests for src/analysis/tables.py"""
from __future__ import annotations

import pytest

from src.analysis.tables import build_experiment_1, build_experiment_2, build_experiment_3


def _make_metrics() -> dict:
    """Minimal all_metrics fixture with two tasks and relevant quant_ids."""
    return {
        "mobilenetv2_df_vs_real": {
            "fp32_gpu":           {"auc": 0.999, "accuracy": 0.989, "fps": 0.0,  "model_size_mb": 8.47, "video_accuracy": 0.991, "num_frames": 7000},
            "dynamic_range_tflite": {"auc": 0.990, "accuracy": 0.975, "fps": 1.2,  "model_size_mb": 2.38, "video_accuracy": 0.980, "num_frames": 6727},
            "float16_tflite":     {"auc": 0.996, "accuracy": 0.985, "fps": 5.9,  "model_size_mb": 4.26, "video_accuracy": 0.988, "num_frames": 7000},
            "int8_static_tflite": {"auc": 0.997, "accuracy": 0.986, "fps": 11.6, "model_size_mb": 2.58, "video_accuracy": 0.989, "num_frames": 7000},
        },
        "mobilenetv2_nt_vs_real": {
            "fp32_gpu":           {"auc": 0.973, "accuracy": 0.924, "fps": 0.0,  "model_size_mb": 8.47, "video_accuracy": 0.930, "num_frames": 7000},
            "dynamic_range_tflite": {"auc": 0.950, "accuracy": 0.900, "fps": 1.5,  "model_size_mb": 2.39, "video_accuracy": 0.910, "num_frames": 7000},
            "float16_tflite":     {"auc": 0.960, "accuracy": 0.910, "fps": 5.3,  "model_size_mb": 4.26, "video_accuracy": 0.918, "num_frames": 7000},
            "int8_static_tflite": {"auc": 0.963, "accuracy": 0.912, "fps": 11.3, "model_size_mb": 2.58, "video_accuracy": 0.920, "num_frames": 7000},
        },
    }


# ── Experiment 1 ─────────────────────────────────────────────────────────────

def test_experiment_1_row_count():
    rows = build_experiment_1(_make_metrics())
    # 2 tasks × 2 quant_ids (fp32_gpu, int8_static_tflite) = 4 rows
    assert len(rows) == 4


def test_experiment_1_only_fp32_and_int8():
    rows = build_experiment_1(_make_metrics())
    quant_ids = {r["quant_id"] for r in rows}
    assert quant_ids == {"fp32_gpu", "int8_static_tflite"}


def test_experiment_1_quant_gap_fp32_is_zero():
    rows = build_experiment_1(_make_metrics())
    fp32_rows = [r for r in rows if r["quant_id"] == "fp32_gpu"]
    for r in fp32_rows:
        assert r["quant_gap"] == pytest.approx(0.0)


def test_experiment_1_quant_gap_int8():
    rows = build_experiment_1(_make_metrics())
    df_int8 = next(r for r in rows if r["task_id"] == "mobilenetv2_df_vs_real" and r["quant_id"] == "int8_static_tflite")
    # gap = fp32_auc - int8_auc = 0.999 - 0.997
    assert df_int8["quant_gap"] == pytest.approx(0.999 - 0.997)


def test_experiment_1_has_required_keys():
    rows = build_experiment_1(_make_metrics())
    required = {"task_id", "quant_id", "auc", "accuracy", "fps", "model_size_mb", "quant_gap", "num_frames", "video_accuracy"}
    for r in rows:
        assert required <= set(r.keys())


# ── Experiment 2 ─────────────────────────────────────────────────────────────

def test_experiment_2_row_count():
    rows = build_experiment_2(_make_metrics())
    # 2 classes (df, nt) × 2 quant_ids (fp32, int8) = 4 rows
    assert len(rows) == 4


def test_experiment_2_has_manipulation_class():
    rows = build_experiment_2(_make_metrics())
    classes = {r["manipulation_class"] for r in rows}
    assert classes == {"df", "nt"}


def test_experiment_2_nt_gap_larger_than_df_gap():
    rows = build_experiment_2(_make_metrics())
    df_gap = next(r["quant_gap"] for r in rows if r["manipulation_class"] == "df" and r["quant_id"] == "int8_static_tflite")
    nt_gap = next(r["quant_gap"] for r in rows if r["manipulation_class"] == "nt" and r["quant_id"] == "int8_static_tflite")
    assert nt_gap > df_gap


# ── Experiment 3 ─────────────────────────────────────────────────────────────

def test_experiment_3_row_count():
    rows = build_experiment_3(_make_metrics())
    # 3 TFLite quant variants
    assert len(rows) == 3


def test_experiment_3_only_tflite_variants():
    rows = build_experiment_3(_make_metrics())
    quant_ids = {r["quant_id"] for r in rows}
    assert quant_ids == {"dynamic_range_tflite", "float16_tflite", "int8_static_tflite"}


def test_experiment_3_has_per_task_columns():
    rows = build_experiment_3(_make_metrics())
    required = {"quant_id", "overall_auc", "df_auc", "nt_auc", "df_accuracy", "nt_accuracy",
                "df_fps", "nt_fps", "model_size_mb", "df_quant_gap", "nt_quant_gap"}
    for r in rows:
        assert required <= set(r.keys())


def test_experiment_3_overall_auc_is_mean_of_df_and_nt():
    rows = build_experiment_3(_make_metrics())
    for r in rows:
        assert r["overall_auc"] == pytest.approx((r["df_auc"] + r["nt_auc"]) / 2, abs=1e-6)


def test_experiment_3_quant_gap_is_fp32_minus_quant():
    # gap = fp32_auc - quant_auc; can be negative (e.g. float16 marginally beats fp32)
    metrics = _make_metrics()
    rows = build_experiment_3(metrics)
    df_fp32_auc = metrics["mobilenetv2_df_vs_real"]["fp32_gpu"]["auc"]
    nt_fp32_auc = metrics["mobilenetv2_nt_vs_real"]["fp32_gpu"]["auc"]
    for r in rows:
        q = r["quant_id"]
        assert r["df_quant_gap"] == pytest.approx(
            round(df_fp32_auc - metrics["mobilenetv2_df_vs_real"][q]["auc"], 6)
        )
        assert r["nt_quant_gap"] == pytest.approx(
            round(nt_fp32_auc - metrics["mobilenetv2_nt_vs_real"][q]["auc"], 6)
        )
