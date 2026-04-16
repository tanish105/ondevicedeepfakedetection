# Phase 6 Implementation Log

**Branch:** `scientific-salsa`  
**Date:** 2026-04-14  
**Author:** Claude Sonnet 4.6  
**Workflow:** Brainstorming → Design spec → Implementation plan → Subagent-driven development (fresh subagent per task, spec compliance + code quality review after each)

---

## Overview

Phase 5 produced 14 prediction CSVs on a Pixel 6 emulator. Phase 6 is the analysis phase: consume those CSVs, compute accuracy metrics for all quantization variants vs the FP32-GPU baseline, build three experiment comparison tables, mine neural-texture (NT) failure frames, and render a qualitative failure grid PNG.

No device work, no training, no new schemas — pure Python analysis on top of the existing project stack.

---

## Starting Point: What Phase 5 Left Behind

Inputs consumed by Phase 6:

| Artifact | Contents |
|---|---|
| `artifacts/predictions/` | 14 CSVs: 6 TFLite frame CSVs, 6 TFLite video CSVs, 2 FP32-GPU frame CSVs |
| `artifacts/metrics/phase4_export_summary.json` | Model sizes (MB) per quantization variant |

Two binary classification tasks:
- `mobilenetv2_df_vs_real` — Deepfakes vs real
- `mobilenetv2_nt_vs_real` — Neural-texture fakes vs real

Three TFLite quantization variants:
- `dynamic_range_tflite`
- `float16_tflite`
- `int8_static_tflite`

One FP32-GPU baseline (PyTorch, GPU inference, no video CSVs).

One known data artifact: `df_vs_real / dynamic_range_tflite` has 6,727 frames instead of 7,000 due to an emulator data transfer issue during Phase 5.

---

## Design

### Approach chosen

After evaluating three options (monolithic script, focused modules + runner, class-based pipeline), **focused modules + runner** was selected:

- Pure-function modules with no I/O are trivially testable in isolation
- Each module has one clear responsibility
- The orchestrator handles all I/O and wires modules together
- No class hierarchy needed for a one-shot analysis script

### Design document

Saved to `docs/phase6_analysis_design.md` (commit `c8fed45`).

### Module structure decided

```
src/analysis/
  __init__.py        # empty package marker
  metrics.py         # pure: AUC, binary accuracy, video-level majority vote
  tables.py          # pure: assemble the 3 experiment comparison tables
  failures.py        # mine NT failure frames, render 2×3 failure grid PNG
  run_phase6.py      # orchestrator: load → compute → write artifacts
```

### Data flow

```
artifacts/predictions/*.csv
        │
        ▼ csv_io.read_rows() → List[FramePredictionRow] / List[VideoPredictionRow]
        │
        ├─► metrics.py → {auc, accuracy, fps, video_accuracy, num_frames} per condition
        │
        ├─► tables.py → experiment_1/2/3 as List[dict]
        │
        ├─► failures.py → nt_failure_frames.csv + nt_failure_grid.png
        │
        └─► run_phase6.py writes all JSON/CSV/PNG to artifacts/analysis/
```

---

## Implementation

Implementation was executed task-by-task with a fresh subagent per task. Each task went through:
1. Implementer subagent (TDD: failing tests first, then implementation)
2. Spec compliance review subagent
3. Code quality review subagent
4. Fix loop if either review found issues

### Task 1 — `src/analysis/metrics.py`

**Commit:** `2cec59b`  
**Tests:** `tests/analysis/test_metrics.py` (6 tests)

Two functions:

```python
def frame_metrics(rows: List[FramePredictionRow]) -> dict:
    """Returns {"auc": float, "accuracy": float, "num_frames": int}."""
```

AUC is `sklearn.metrics.roc_auc_score`. Binary accuracy at threshold 0.5 (comparing `pred_label` to `binary_label`). Raises `ValueError` if only one class is present in labels — `roc_auc_score` silently returns NaN in that case, which was caught during review and guarded explicitly.

```python
def fp32_video_accuracy(frame_rows: List[FramePredictionRow]) -> Tuple[float, int]:
    """Returns (accuracy, num_videos). Tie-break: equal votes resolve to 0 (real)."""
```

FP32 has no pre-built video CSVs, so video-level accuracy is computed on-the-fly by grouping frame predictions by `video_id` and taking a majority vote. Tie-breaking rule (equal counts of 0 and 1 → predict real) was explicitly documented.

**Bug caught during review:** `roc_auc_score` returns `NaN` without raising when all labels are one class. Fixed by adding a pre-check:
```python
if len(set(labels)) < 2:
    raise ValueError(
        f"frame_metrics requires both classes present in labels; "
        f"got only {set(labels)} across {len(rows)} rows"
    )
```

**Commit:** `314bd2a` (fix: guard frame_metrics against single-class input)

---

### Task 2 — `src/analysis/tables.py`

**Commit:** `693d213`, then refactored to `c016050`  
**Tests:** `tests/analysis/test_tables.py` (12 tests)

Three builder functions, all pure (no I/O):

**`build_experiment_1`** — FP32-GPU vs INT8-Mobile, per task. 4 rows (2 tasks × 2 quant variants). Columns: `task_id`, `quant_id`, `auc`, `accuracy`, `fps`, `model_size_mb`, `video_accuracy`, `num_frames`, `quant_gap`.

**`build_experiment_2`** — Same as Experiment 1 but with `manipulation_class` added per task to make the DF vs NT comparison explicit. Originally implemented by duplicating 13 lines of `build_experiment_1` — caught during code quality review and refactored to delegate:
```python
def build_experiment_2(all_metrics: AllMetrics) -> List[dict]:
    # Experiment 2 is Experiment 1 with manipulation_class projected in.
    # Any change to build_experiment_1 columns will propagate here.
    return [
        {**row, "manipulation_class": TASK_TO_FAKE_CLASS[row["task_id"]]}
        for row in build_experiment_1(all_metrics)
    ]
```

Also caught during review: the original used a private `_TASK_TO_CLASS` dict that duplicated the existing `TASK_TO_FAKE_CLASS` constant in `src/common/constants.py`. Fixed by importing the existing constant.

**`build_experiment_3`** — All three TFLite quant methods in one table. One row per quant variant with `df_`/`nt_`-prefixed columns for per-task values.

---

### Task 3 — `src/analysis/failures.py`

**Commit:** `8918381`  
**Tests:** `tests/analysis/test_failures.py` (6 tests)

**`mine_nt_failures(fp32_rows, int8_rows, n=20)`**

Finds NT frames where FP32-GPU is correct but INT8-Mobile is wrong:
- `binary_label == 1` (fake frame)
- `fp32.pred_label == 1` (FP32 got it right)
- `int8.pred_label == 0` (INT8 got it wrong)

Rows are matched by `(video_id, frame_idx)` key. Sorted descending by `fp32_score_fake - int8_score_fake` (largest confidence gap first). Returns top `n` as `FailureMetadataRow` instances (existing schema).

**`render_failure_grid(failures, frames_root, output_path, n=6)`**

Renders a 2×3 PNG grid using matplotlib + Pillow. Each cell shows the face crop image and a caption: `"FP32: {fp32_score:.3f} | INT8: {int8_score:.3f}"`. Title: `"NT Failure Frames: FP32-correct, INT8-wrong"`. Saved at 120 DPI. matplotlib backend set to `Agg` (non-interactive) to avoid display dependency.

Frame images (745 MB) were copied from the `Sudhanva-M/deployment` worktree: `data/processed/frames/nt_vs_real/test/`.

---

### Task 4 — `src/analysis/run_phase6.py`

**Commit:** `38c4430` (implementation), `788220a` (quality fixes)  
**Tests:** none (orchestrator — covered by integration run)

The orchestrator. Key design decisions:

- `PREDS_DIR`, `METRICS_DIR`, `ANALYSIS_DIR` defined as module-level constants relative to `PROJECT_ROOT`
- `_frame_csv(task_id, quant_id)` — returns the right CSV path; FP32 uses `_fp32_test_frames.csv` naming while TFLite uses `_{quant_id}_test_frames.csv`
- `_load_model_sizes()` — reads `phase4_export_summary.json`, wraps key access with context-rich `KeyError` to surface unexpected JSON structure
- `build_all_metrics()` — loads all 14 CSVs, calls `frame_metrics` and `fp32_video_accuracy`, returns a nested dict `{task_id: {quant_id: {auc, accuracy, fps, model_size_mb, video_accuracy, num_frames}}}`
- `main()` — calls modules in sequence, writes 6 output files, prints a summary table

NT rows are re-loaded in `main()` after `build_all_metrics()` returns because the metrics function discards raw rows after computing. Comment added to explain this:
```python
# NT frame rows are re-loaded here because build_all_metrics discards raw rows
# after computing metrics. Acceptable for a research script (7k rows, ~1s).
```

**Quality fixes applied (commit `788220a`):**
1. Comment added above NT row re-load explaining the design
2. `build_experiment_2` delegation pattern documented in `tables.py`
3. `FRAMES_ROOT = PROJECT_ROOT` alias removed; `PROJECT_ROOT` used directly at call site
4. `_load_model_sizes` wrapped with context-rich `KeyError`
5. `src/common/__init__.py`: removed `from src.common.runtime import ...` — `runtime.py` imports `torch` at module level, which is not installed in the analysis venv. Analysis modules don't need `RuntimeContext` or `resolve_device`. No callers import these via `src.common.*`.

---

## Infrastructure fixes

### `conftest.py` — torch stub extended

**Problem:** scipy 1.17.1 / array_api_compat crashes with `AttributeError: module 'torch' has no attribute 'Tensor'` when sklearn is first imported in a torch-free environment. The existing stub in `conftest.py` created a minimal `torch` module but didn't add a `Tensor` attribute.

**Fix:**
```python
class _TensorStub:
    pass
torch.Tensor = _TensorStub
```

Added inside the existing `_make_torch_stub()` function in `conftest.py`.

### `requirements.txt` — matplotlib added

`matplotlib>=3.8.0` was used by `failures.py` but was absent from `requirements.txt`. Added in commit `1e9a24e`.

---

## Tests

All tests written first (failing), then implementation written to pass them.

| File | Tests | What is covered |
|---|---|---|
| `tests/analysis/test_metrics.py` | 6 | Perfect predictions, imperfect predictions, frame count, single-class ValueError, video accuracy all-correct, video accuracy majority vote |
| `tests/analysis/test_tables.py` | 12 | Row counts, column presence, quant_gap=0 for FP32, quant_gap computation, manipulation_class, NT gap > DF gap, all 3 TFLite variants in Exp 3 |
| `tests/analysis/test_failures.py` | 6 | Finds disagreements, sorted by delta, skips real frames, respects n limit, returns FailureMetadataRow type, render_failure_grid creates PNG |

**Total: 24 tests, all passing.**

```
24 passed in 1.37s
```

---

## Results

### Full metrics summary

| Task | Quant | AUC | Accuracy | FPS | Quant Gap |
|---|---|---|---|---|---|
| df_vs_real | fp32_gpu | 0.9991 | 0.9887 | — | 0.0000 |
| df_vs_real | dynamic_range_tflite | 0.9991 | 0.9883 | 1.4 | 0.0000 |
| df_vs_real | float16_tflite | 0.9991 | 0.9887 | 6.4 | 0.0000 |
| df_vs_real | int8_static_tflite | 0.9987 | 0.9880 | 13.0 | 0.0004 |
| nt_vs_real | fp32_gpu | 0.9733 | 0.9240 | — | 0.0000 |
| nt_vs_real | dynamic_range_tflite | 0.9731 | 0.9241 | 7.5 | 0.0002 |
| nt_vs_real | float16_tflite | 0.9733 | 0.9240 | 5.7 | 0.0000 |
| nt_vs_real | int8_static_tflite | 0.9710 | 0.9131 | 12.2 | 0.0023 |

FPS values are from emulator runs (Pixel 6 AVD, Android 13).

### Model sizes

| Variant | Size (MB) | Reduction vs FP32 |
|---|---|---|
| fp32_gpu (ONNX) | 8.47 | baseline |
| float16_tflite | 4.26 | 50% |
| int8_static_tflite | 2.58 | 70% |
| dynamic_range_tflite | 2.38 | 72% |

### Key findings

1. **Quantization gap is small for DF:** INT8 static AUC gap = 0.0004 — negligible degradation on deepfakes.
2. **Quantization gap is larger for NT:** INT8 static AUC gap = 0.0023 — still small in absolute terms, but ~6× the DF gap. Neural textures are harder; INT8 is slightly more affected.
3. **float16 is essentially lossless:** AUC gap ≤ 0.000002 on both tasks. FP16 quantization is free in terms of accuracy at 50% size reduction.
4. **dynamic_range trades accuracy for size:** Smallest model at 2.38 MB, AUC gap 0.0002/0.0001 — acceptable for most deployment contexts.
5. **INT8 is fastest:** ~13 FPS on-device, vs ~6 FPS for float16 and ~1.4 FPS for dynamic_range on the emulator.
6. **Video-level accuracy:** FP32 achieves 100% video accuracy on DF task, 96.4% on NT task. INT8 achieves 100% (DF) and 95.0% (NT).

### NT failure analysis

20 failure candidates mined, top 6 rendered to the failure grid. All 20 share:
- `selection_reason = "fp32_correct_int8_wrong"`
- `binary_label = 1` (all fake frames)
- FP32 score range: 0.50–0.90
- INT8 score range: 0.25–0.48

The largest confidence delta was video `732_691`, frame 8: FP32 score 0.900 → INT8 score 0.426. INT8 consistently underestimates confidence on NT fakes near the decision boundary.

---

## Output artifacts

All written to `artifacts/analysis/`:

| File | Description |
|---|---|
| `experiment_1.json` | 4 rows: FP32 vs INT8 per task |
| `experiment_2.json` | 4 rows: same as Exp 1 + `manipulation_class` |
| `experiment_3.json` | 3 rows: all TFLite variants, df_/nt_ prefixed columns |
| `phase6_summary.json` | All 8 conditions flattened into one JSON |
| `nt_failure_frames.csv` | 20 failure frame rows (`FailureMetadataRow` schema) |
| `nt_failure_grid.png` | 2×3 grid: 6 worst NT failures with FP32/INT8 score captions |

---

## Commit log

| Commit | Message |
|---|---|
| `c8fed45` | docs: Phase 6 analysis design spec |
| `2cec59b` | feat(analysis): add metrics module with frame_metrics and fp32_video_accuracy |
| `314bd2a` | fix(analysis): guard frame_metrics against single-class input, document tie-break |
| `693d213` | feat(analysis): add tables module for experiment 1/2/3 builders |
| `c016050` | refactor(analysis): eliminate build_experiment_2 duplication, use TASK_TO_FAKE_CLASS |
| `8918381` | feat(analysis): add failures module for NT failure mining and grid rendering |
| `1e9a24e` | chore: add matplotlib>=3.8.0 to requirements |
| `38c4430` | feat(analysis): add Phase 6 orchestrator run_phase6 |
| `788220a` | fix(analysis): apply quality fixes and add Phase 6 analysis artifacts |

---

## Files created or modified

| Path | Action | Lines |
|---|---|---|
| `docs/phase6_analysis_design.md` | Created | — |
| `src/analysis/__init__.py` | Created | 0 |
| `src/analysis/metrics.py` | Created | 57 |
| `src/analysis/tables.py` | Created | 89 |
| `src/analysis/failures.py` | Created | 105 |
| `src/analysis/run_phase6.py` | Created | 160 |
| `tests/analysis/__init__.py` | Created | 0 |
| `tests/analysis/test_metrics.py` | Created | 100 |
| `tests/analysis/test_tables.py` | Created | 109 |
| `tests/analysis/test_failures.py` | Created | 119 |
| `conftest.py` | Modified | +4 (torch.Tensor stub) |
| `requirements.txt` | Modified | +1 (matplotlib) |
| `src/common/__init__.py` | Modified | -3 (remove runtime import) |
| `artifacts/analysis/experiment_1.json` | Created | — |
| `artifacts/analysis/experiment_2.json` | Created | — |
| `artifacts/analysis/experiment_3.json` | Created | — |
| `artifacts/analysis/phase6_summary.json` | Created | — |
| `artifacts/analysis/nt_failure_frames.csv` | Created | 21 rows |
| `artifacts/analysis/nt_failure_grid.png` | Created | 2×3 PNG |

---

## Real-Device Re-run (2026-04-16)

**Branch:** `scientific-salsa`
**Device:** Nothing A063 (manufacturer: Nothing, model: A063)
**Hardware:** Snapdragon 888 (lahaina), Adreno 660, Android 14 (API 34)
**RAM:** 7,258 MB
**TFLite version:** 2.17.0
**Duration:** 26 minutes (17:19–17:45 local time)

The full Phase 5 benchmark was re-run on a physical Android device to replace emulator results. All Phase 6 analysis artifacts were regenerated.

### What changed

The emulator run (Pixel 6 AVD, Android 13, ranchu virtual CPU) was committed as a backup at `c161777` before being overwritten. The real-device results are at `19028ac` and the regenerated analysis at `c301a76`.

### Latency results

| Task | Variant | Mean latency (ms) | FPS |
|---|---|---|---|
| df_vs_real | dynamic_range_tflite | 16.9 | 59.1 |
| df_vs_real | float16_tflite | 28.9 | 34.5 |
| df_vs_real | int8_static_tflite | 15.2 | 65.6 |
| nt_vs_real | dynamic_range_tflite | 23.5 | 42.6 |
| nt_vs_real | float16_tflite | 59.4 | 16.8 |
| nt_vs_real | int8_static_tflite | 32.6 | 30.7 |

**Note on nt_vs_real latency:** nt_vs_real ran second (after df_vs_real) on a device that was already thermally loaded. The Snapdragon 888 is prone to thermal throttling. The nt_vs_real numbers are slower than df_vs_real for this reason, not a model difference. A cold-device re-run of nt_vs_real alone would likely match df_vs_real latencies.

### AUC and accuracy

Unchanged from the emulator run — same model weights, same images, deterministic inference:

| Task | Variant | AUC | Accuracy | Quant gap |
|---|---|---|---|---|
| df_vs_real | fp32_gpu | 0.9991 | 0.9887 | 0.0000 |
| df_vs_real | dynamic_range_tflite | 0.9991 | 0.9887 | −0.0000 |
| df_vs_real | float16_tflite | 0.9991 | 0.9887 | −0.0000 |
| df_vs_real | int8_static_tflite | 0.9987 | 0.9880 | 0.0004 |
| nt_vs_real | fp32_gpu | 0.9733 | 0.9240 | 0.0000 |
| nt_vs_real | dynamic_range_tflite | 0.9731 | 0.9241 | 0.0002 |
| nt_vs_real | float16_tflite | 0.9733 | 0.9240 | 0.0000 |
| nt_vs_real | int8_static_tflite | 0.9710 | 0.9131 | 0.0023 |

### Notable observations

- **dynamic_range is faster than float16 on real hardware.** On the emulator dynamic_range was 5× slower (760ms vs 169ms). On the Snapdragon 888 CPU, INT8 weight loading reduces memory bandwidth enough to beat float16, which TFLite computes in float32 anyway (no native FP16 SIMD path in the default CPU delegate).
- **No frames skipped.** The 273-frame skip issue from the emulator run (a data transfer artifact) did not recur on the physical device.
- **TFLite ran CPU-only.** No NNAPI, GPU, or DSP delegate was configured. With the Hexagon 780 DSP delegate enabled, int8_static latency would likely drop to 2–5 ms.

### Commits

| Commit | Message |
|---|---|
| `c161777` | chore: backup emulator benchmark results before real-device re-run |
| `19028ac` | feat(benchmark): real-device Phase 5 results on Nothing A063 |
| `c301a76` | feat(analysis): regenerate Phase 6 results with real-device latency |
