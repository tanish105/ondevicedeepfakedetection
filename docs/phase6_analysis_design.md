# Phase 6 Analysis — Design Spec

**Date:** 2026-04-14  
**Branch:** scientific-salsa  
**Status:** Approved

---

## Context

Phase 5 produced 14 prediction CSVs in `artifacts/predictions/`:
- 6 TFLite frame CSVs (2 tasks × 3 quant variants: dynamic_range, float16, int8_static)
- 6 TFLite video CSVs (same 6 conditions)
- 2 FP32 GPU frame CSVs (baseline, pytorch_fp32 runtime, no video CSVs)

Phase 6 computes accuracy metrics, builds the three experiment tables, mines NT failure frames, and produces the qualitative failure grid. No additional data collection or device work is needed.

---

## Module Structure

```
src/analysis/
  __init__.py
  metrics.py      # pure functions: AUC, accuracy, gap, video-level aggregates
  tables.py       # builds the 3 experiment tables as dicts
  failures.py     # mines NT failures, renders the failure grid PNG
  run_phase6.py   # orchestrator: loads CSVs → calls modules → writes artifacts
```

---

## Data Flow

1. `run_phase6.py` loads all 14 CSVs via `csv_io.read_rows()` into typed row lists.
2. FP32 video-level aggregates are computed on-the-fly from FP32 frame rows via majority vote by `video_id` (no FP32 video CSVs exist).
3. `metrics.py` receives plain row lists and returns plain dicts — no I/O, testable in isolation.
4. `tables.py` calls `metrics.py` and assembles the three experiment dicts.
5. `failures.py` takes NT frame rows for fp32_gpu and int8_static_tflite, finds disagreements, saves `nt_failure_frames.csv`, then renders a 2×3 PNG grid from `data/processed/frames/`.
6. `run_phase6.py` writes all outputs to `artifacts/analysis/` and prints a summary to stdout.

---

## Metrics

Computed per condition (task_id × quant_id):

- **Frame-level overall:** AUC (`sklearn.metrics.roc_auc_score`), Binary Accuracy at threshold 0.5
- **Per-class frame-level:** AUC and Accuracy separately for DF frames and NT frames
- **Quantization gap:** `AUC(fp32_gpu) − AUC(quantized)` for each TFLite variant
- **Video-level:** Accuracy from `majority_vote_pred`, mean FPS from `fps` field

---

## Experiment Tables

### Experiment 1
FP32-GPU vs INT8-Mobile, per task.

Columns: `task_id`, `quant_id`, `auc`, `accuracy`, `fps`, `model_size_mb`, `quant_gap`

### Experiment 2
Per-class (DF vs NT) comparison under FP32-GPU and INT8-Mobile.

Columns: `task_id`, `quant_id`, `class`, `auc`, `accuracy`, `quant_gap`

Expected finding: gap is larger for NT than DF.

### Experiment 3
All three TFLite quantization methods, per task.

Columns: `task_id`, `quant_id`, `auc`, `auc_df_class`, `auc_nt_class`, `accuracy`, `fps`, `model_size_mb`, `quant_gap`

---

## Failure Analysis

**Mining criteria:** NT test frames where:
- `binary_label == 1` (fake)
- FP32 `pred_label == 1` (correct)
- INT8 `pred_label == 0` (wrong)

Sorted descending by `fp32_score_fake − int8_score_fake`. Top candidates saved as `nt_failure_frames.csv` using the existing `FailureMetadataRow` schema.

**Failure grid:** 2×3 PNG showing 6 frames. Each cell: frame image + FP32 score + INT8 score as caption. Rendered via matplotlib + Pillow from `data/processed/frames/nt_vs_real/test/`.

---

## Outputs

All written to `artifacts/analysis/`:

```
experiment_1.json
experiment_2.json
experiment_3.json
nt_failure_frames.csv        # FailureMetadataRow schema
nt_failure_grid.png          # 2×3 qualitative failure grid
phase6_summary.json          # all metrics in one flat structure
```

---

## Model Size Reference

From `artifacts/metrics/phase4_export_summary.json`:

| quant_id | size_mb |
|---|---|
| fp32_gpu (ONNX) | 8.47 |
| dynamic_range_tflite | 2.38 |
| float16_tflite | 4.26 |
| int8_static_tflite | 2.58 |

---

## 273-Frame Discrepancy

`df_vs_real / dynamic_range_tflite` processed 6727 frames instead of 7000 (emulator data transfer artifact). Metrics for this condition are computed on its actual sample. All output tables include a `num_frames` field per condition so the discrepancy is visible without silently skewing other conditions.

---

## Dependencies

- `scikit-learn` — `roc_auc_score`
- `matplotlib` + `Pillow` — failure grid PNG
- `numpy` — already in project stack

No new schemas, constants, or CSV I/O changes needed. `FailureMetadataRow` already defined in `src/common/schemas.py`.

---

## Out of Scope

- No training, export, or device code
- No changes to existing `src/common/`, `src/mobile/`, or `src/eval/`
- Frame images in `data/processed/frames/` are not committed (covered by `.gitignore`)
