# Project Contract

This document is the source of truth for repository conventions, team ownership, file schemas, and cross-stage interfaces.

## 1. Team Ownership

### Person A

Owns:
- FF++ dataset ingestion path configuration
- Frame extraction and MTCNN face-crop pipeline
- Processed frame manifest generation
- MobileNetV2 FP32 training for `df` and `nt`
- Validation gating before export: `AUC > 0.85`
- FP32 baseline speed and accuracy measurement on GPU
- Model export path: PyTorch -> ONNX -> TensorFlow -> TFLite
- Quantization pipelines for:
  - Dynamic Range
  - Full Integer INT8 Static
  - Float16
- Calibration-set generation for Full Integer INT8
- Mobile app and on-device inference benchmarking once platform is finalized
- Experiment 1 and Experiment 3 result tables

Deliverables:
- trained checkpoints
- exported models
- benchmark outputs
- mobile prediction files that follow the schema below

### Person B

Owns:
- Verification of official FF++ split integrity
- Verification of label structure for `real`, `df`, and `nt`
- Metric code for frame-level and video-level evaluation
- Per-class AUC analysis for DF and NT
- Quantization-gap analysis
- Failure-frame mining for NT
- Experiment 2 result table
- Qualitative examples for poster/video

Deliverables:
- evaluation outputs
- analysis tables
- failure case assets
- report-ready figures and summaries

### Shared

Shared decisions must not drift across branches:
- split files
- manifest schema
- prediction schema
- metric definitions
- naming conventions for artifact files

## 2. Directory Contract

### Data directories

- `data/raw/ffpp_c23/`
  - local-only FF++ download and unpacked videos
- `data/interim/`
  - temporary extraction outputs
- `data/processed/`
  - persisted frame crops and metadata

### Artifact directories

- `artifacts/checkpoints/`
  - FP32 training checkpoints
- `artifacts/exports/`
  - `onnx/`, `saved_model/`, `tflite/`
- `artifacts/metrics/`
  - experiment result CSV and JSON files
- `artifacts/predictions/`
  - prediction exchange files between A and B
- `artifacts/failures/`
  - copied qualitative failure frames

### Execution environments

- Local laptop:
  - default environment for code edits, config updates, Git operations, small-sample sanity checks, and report work
- GCP GPU VM:
  - default environment for full preprocessing, training, validation, export, and large inference runs

Rules:
- Python pipeline code must support CPU-only execution.
- GPU use must be optional and auto-detected at runtime.
- No training or preprocessing code may hardcode `cuda` as the only valid device.

## 3. Naming Conventions

### Manipulation IDs

Use only these identifiers in code and files:
- `real`
- `df`
- `nt`

### Model IDs

Use these stable task identifiers:
- `mobilenetv2_df_vs_real`
- `mobilenetv2_nt_vs_real`

### Quantization IDs

Use only:
- `fp32_gpu`
- `dynamic_range_tflite`
- `int8_static_tflite`
- `float16_tflite`

## 4. Split Contract

The project uses the official FF++ split at the video level:
- train: `720`
- val: `140`
- test: `140`

Store split files in:
- `data/processed/splits/df_vs_real_train.csv`
- `data/processed/splits/df_vs_real_val.csv`
- `data/processed/splits/df_vs_real_test.csv`
- `data/processed/splits/nt_vs_real_train.csv`
- `data/processed/splits/nt_vs_real_val.csv`
- `data/processed/splits/nt_vs_real_test.csv`

Each split file schema:

| column | type | description |
|---|---|---|
| `video_id` | string | Stable FF++ video identifier |
| `source_video_id` | string | Original source identity/video id if available |
| `manipulation_type` | string | `real`, `df`, or `nt` |
| `binary_label` | int | `0` for real, `1` for fake |
| `split` | string | `train`, `val`, `test` |
| `video_path` | string | Local absolute or repo-relative path |

Rules:
- A video ID must appear in exactly one split for a given task.
- No frame from a video may cross splits.
- `df_vs_real_*` includes only `real` and `df`.
- `nt_vs_real_*` includes only `real` and `nt`.

## 5. Processed Frame Manifest Contract

Store frame manifests in:
- `data/processed/manifests/df_vs_real_frames.csv`
- `data/processed/manifests/nt_vs_real_frames.csv`

Schema:

| column | type | description |
|---|---|---|
| `task_id` | string | `mobilenetv2_df_vs_real` or `mobilenetv2_nt_vs_real` |
| `video_id` | string | Stable video identifier |
| `frame_idx` | int | Zero-based sampled-frame index within selected 25 |
| `frame_source_index` | int | Original frame index in source video |
| `image_path` | string | Face-crop path |
| `face_found` | int | `1` if MTCNN detected a face, else `0` |
| `bbox_x1` | float | Face box left |
| `bbox_y1` | float | Face box top |
| `bbox_x2` | float | Face box right |
| `bbox_y2` | float | Face box bottom |
| `width` | int | Output image width, expected `224` |
| `height` | int | Output image height, expected `224` |
| `manipulation_type` | string | `real`, `df`, or `nt` |
| `binary_label` | int | `0` real, `1` fake |
| `split` | string | `train`, `val`, or `test` |

Rules:
- Sample exactly `25` frames per video when possible.
- If a video yields fewer valid face crops, record the actual count and flag it in QC outputs.
- `frame_idx` is the sampled-frame order, not the original video frame number.

## 6. Calibration Set Contract

Full Integer INT8 uses a held-out calibration set of `500` validation frames.

Store:
- `data/processed/calibration/int8_calibration_frames.csv`

Schema:

| column | type | description |
|---|---|---|
| `task_id` | string | Model task ID |
| `video_id` | string | Video identifier |
| `frame_idx` | int | Sampled-frame index |
| `image_path` | string | Calibration image path |
| `manipulation_type` | string | `real`, `df`, or `nt` |
| `split` | string | Must be `val` |

Rules:
- Calibration data must come only from validation frames.
- No test frame may be used for calibration.
- Recommended target balance per task: `250` real and `250` fake, sampled across many videos.

## 7. Prediction Handoff Contract

Person A writes prediction files. Person B consumes them directly.

Store in:
- `artifacts/predictions/{task_id}/{quant_id}_{split}_frame_predictions.csv`
- `artifacts/predictions/{task_id}/{quant_id}_{split}_video_predictions.csv`

Example paths:
- `artifacts/predictions/mobilenetv2_df_vs_real/fp32_gpu_test_frame_predictions.csv`
- `artifacts/predictions/mobilenetv2_df_vs_real/int8_static_tflite_test_video_predictions.csv`

### Frame prediction schema

| column | type | description |
|---|---|---|
| `task_id` | string | Model task ID |
| `quant_id` | string | One of the quantization IDs |
| `split` | string | `train`, `val`, or `test` |
| `video_id` | string | Video identifier |
| `frame_idx` | int | Sampled-frame index |
| `image_path` | string | Evaluated image path |
| `manipulation_type` | string | `real`, `df`, or `nt` |
| `binary_label` | int | `0` real, `1` fake |
| `score_fake` | float | Probability or calibrated score for fake class |
| `pred_label` | int | Thresholded prediction at `0.5` |
| `latency_ms` | float | Per-frame inference latency |
| `device_name` | string | `nvidia_gpu` for server, phone model for device |
| `runtime` | string | `pytorch`, `tflite`, or equivalent |

### Video prediction schema

| column | type | description |
|---|---|---|
| `task_id` | string | Model task ID |
| `quant_id` | string | One of the quantization IDs |
| `split` | string | `train`, `val`, or `test` |
| `video_id` | string | Video identifier |
| `manipulation_type` | string | `real`, `df`, or `nt` |
| `binary_label` | int | `0` real, `1` fake |
| `num_frames_used` | int | Number of valid frames aggregated |
| `mean_score_fake` | float | Mean frame score for fake class |
| `majority_vote_pred` | int | Video prediction by majority vote |
| `fps` | float | Aggregate throughput measurement |
| `device_name` | string | Hardware identifier |
| `runtime` | string | Runtime backend |

Rules:
- `score_fake` must be monotonic with fake confidence.
- `pred_label` and `majority_vote_pred` use threshold `0.5`.
- Frame-level and video-level files must be generated for every reported test result.

## 8. Metrics Contract

### Required overall metrics

For every model/runtime condition:
- ROC AUC
- Binary accuracy at threshold `0.5`
- FPS
- model size in MB

### Required stratified metrics

For DF and NT:
- frame-level ROC AUC
- video-level ROC AUC
- quantization gap:
  - `AUC(fp32_gpu) - AUC(quantized_method)`

### Aggregation policy

- Frame-level metrics are computed over all evaluated frames.
- Video-level metrics are computed after majority-vote aggregation.
- The report must state clearly whether a table is frame-level or video-level.

## 9. Experiment Output Files

### Experiment 1

Store:
- `artifacts/metrics/experiment1_summary.csv`

Columns:
- `task_id`
- `quant_id`
- `level`
- `auc`
- `accuracy`
- `fps`
- `model_size_mb`
- `device_name`

Expected rows:
- `df` and `nt`
- `fp32_gpu` and `int8_static_tflite`
- both `frame` and `video` if reported

### Experiment 2

Store:
- `artifacts/metrics/experiment2_per_class_gap.csv`

Columns:
- `task_id`
- `level`
- `auc_fp32_gpu`
- `auc_int8_static_tflite`
- `quant_gap`

### Experiment 3

Store:
- `artifacts/metrics/experiment3_quantization_comparison.csv`

Columns:
- `task_id`
- `quant_id`
- `level`
- `auc`
- `accuracy`
- `fps`
- `model_size_mb`
- `device_name`

Quantization rows:
- `dynamic_range_tflite`
- `int8_static_tflite`
- `float16_tflite`

## 10. Failure Study Contract

Store NT failure examples in:
- `artifacts/failures/nt_quantization_failures/`

Metadata file:
- `artifacts/failures/nt_quantization_failures/metadata.csv`

Schema:

| column | type | description |
|---|---|---|
| `video_id` | string | Video identifier |
| `frame_idx` | int | Sampled-frame index |
| `image_path` | string | Frame path |
| `fp32_score_fake` | float | FP32 fake confidence |
| `int8_score_fake` | float | INT8 fake confidence |
| `binary_label` | int | Ground truth |
| `selection_reason` | string | Short note for poster/video use |

Selection rule:
- Choose `4` to `6` NT test frames that are correctly classified under `fp32_gpu` and misclassified under `int8_static_tflite`.

## 11. Device Logging Contract

Every on-device benchmark must record:
- phone model
- SoC / chipset
- RAM
- Android version
- TFLite runtime version if available

Store:
- `artifacts/metrics/device_specs.json`

## 12. Versioning and Reproducibility

Track at minimum:
- Python version
- PyTorch version
- CUDA version
- TensorFlow version
- ONNX version
- onnx2tf or equivalent conversion tool version
- Android Studio / Gradle / Android SDK versions
- GCP machine type, GPU type, zone, disk size, and Deep Learning VM image family

Recommended device-selection policy:
- `device=auto` by default
- use CUDA when available
- otherwise run on CPU without code changes

Store environment notes in:
- `docs/workflow.md`

## 13. Done Criteria

The implementation phase is considered structurally complete when:
- split files exist and are validated
- frame manifests exist
- FP32 checkpoints for DF and NT exist
- all three quantized exports exist
- Android app can run a single-image inference
- prediction files follow the handoff schema
- Experiments 1, 2, and 3 tables can be generated from artifact files
