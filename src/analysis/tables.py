"""Build experiment comparison tables from computed per-condition metrics."""
from __future__ import annotations

from typing import Any, Dict, List

from src.common.constants import (
    QUANT_DYNAMIC_RANGE,
    QUANT_FLOAT16,
    QUANT_FP32_GPU,
    QUANT_INT8_STATIC,
    TASK_DF_VS_REAL,
    TASK_NT_VS_REAL,
    TASK_TO_FAKE_CLASS,
)

# all_metrics shape: {task_id: {quant_id: {auc, accuracy, fps, model_size_mb, video_accuracy, num_frames}}}
AllMetrics = Dict[str, Dict[str, Dict[str, Any]]]

_TFLITE_QUANT_IDS = [QUANT_DYNAMIC_RANGE, QUANT_FLOAT16, QUANT_INT8_STATIC]


def _gap(fp32_auc: float, quant_auc: float) -> float:
    return round(fp32_auc - quant_auc, 6)


def build_experiment_1(all_metrics: AllMetrics) -> List[dict]:
    """FP32-GPU vs INT8-Mobile per task.

    Returns rows keyed by (task_id, quant_id) for fp32_gpu and int8_static_tflite.
    """
    rows = []
    for task_id in (TASK_DF_VS_REAL, TASK_NT_VS_REAL):
        task_m = all_metrics[task_id]
        fp32_auc = task_m[QUANT_FP32_GPU]["auc"]
        for quant_id in (QUANT_FP32_GPU, QUANT_INT8_STATIC):
            m = task_m[quant_id]
            rows.append({
                "task_id": task_id,
                "quant_id": quant_id,
                "auc": m["auc"],
                "accuracy": m["accuracy"],
                "fps": m["fps"],
                "model_size_mb": m["model_size_mb"],
                "video_accuracy": m["video_accuracy"],
                "num_frames": m["num_frames"],
                "quant_gap": _gap(fp32_auc, m["auc"]) if quant_id != QUANT_FP32_GPU else 0.0,
            })
    return rows


def build_experiment_2(all_metrics: AllMetrics) -> List[dict]:
    """Per-class (DF vs NT) AUC comparison under FP32-GPU and INT8-Mobile.

    Reshapes Experiment 1 rows by adding manipulation_class to show gap
    is larger for NT than DF.
    """
    # Experiment 2 is Experiment 1 with manipulation_class projected in.
    # Any change to build_experiment_1 columns will propagate here.
    return [
        {**row, "manipulation_class": TASK_TO_FAKE_CLASS[row["task_id"]]}
        for row in build_experiment_1(all_metrics)
    ]


def build_experiment_3(all_metrics: AllMetrics) -> List[dict]:
    """All three TFLite quant methods — per-quant rows with both task columns.

    One row per quant variant; columns prefixed df_/nt_ for per-task values.
    """
    rows = []
    df_fp32_auc = all_metrics[TASK_DF_VS_REAL][QUANT_FP32_GPU]["auc"]
    nt_fp32_auc = all_metrics[TASK_NT_VS_REAL][QUANT_FP32_GPU]["auc"]

    for quant_id in _TFLITE_QUANT_IDS:
        df_m = all_metrics[TASK_DF_VS_REAL][quant_id]
        nt_m = all_metrics[TASK_NT_VS_REAL][quant_id]
        rows.append({
            "quant_id": quant_id,
            "df_auc": df_m["auc"],
            "nt_auc": nt_m["auc"],
            "df_accuracy": df_m["accuracy"],
            "nt_accuracy": nt_m["accuracy"],
            "df_fps": df_m["fps"],
            "nt_fps": nt_m["fps"],
            "model_size_mb": df_m["model_size_mb"],  # same architecture, same size
            "df_quant_gap": _gap(df_fp32_auc, df_m["auc"]),
            "nt_quant_gap": _gap(nt_fp32_auc, nt_m["auc"]),
        })
    return rows
