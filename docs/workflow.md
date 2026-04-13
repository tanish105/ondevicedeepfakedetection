# Workflow

## Execution Model

- Local Windows laptop:
  - code editing
  - Git
  - small CPU sanity checks
  - analysis and reporting
  - optional small-sample export checks
- GCP GPU VM:
  - full frame extraction
  - MTCNN face cropping at scale
  - FP32 training
  - validation and full-set inference
  - export and quantization runs

Principles:
- all pipeline code must run on CPU if no GPU is available
- GPU acceleration is used only when detected or explicitly configured
- mobile deployment stays target-neutral until Android vs iOS is finalized

## Phase 1: Shared Setup

1. Confirm FF++ `c23` availability and local storage path.
2. Create official split files for `df_vs_real` and `nt_vs_real`.
3. Freeze manifest and prediction schemas from `docs/project_contract.md`.

## Phase 2: Data Pipeline

1. Index source videos.
2. Uniformly sample `25` frames per video.
3. Run MTCNN face detection and crop to `224x224`.
4. Write processed frame manifests.
5. Generate a `500`-frame INT8 calibration manifest from validation data.

Phase 2 execution note:
- implement and test on the laptop with a tiny subset first
- run the full dataset job on the GCP GPU VM

Laptop-safe Phase 2 work:
- install preprocessing dependencies
- run smoke tests on `2-4` videos per split
- validate manifest format and saved face crops

Recommended local smoke test command:
`python -m src.data.run_preprocessing --task-id mobilenetv2_df_vs_real --split train --device cpu --max-videos 4`

Switch to the GPU VM when:
- local smoke tests pass
- you want to process entire train/val/test splits
- MTCNN latency on CPU becomes the bottleneck
- you are ready to generate the full calibration manifest

## Phase 3: FP32 Training

1. Train `mobilenetv2_df_vs_real`.
2. Train `mobilenetv2_nt_vs_real`.
3. Validate both and gate export on `AUC > 0.85`.
4. Run inference on the full **test split** with the FP32 PyTorch model and write
   `FramePredictionRow` CSVs (`quant_id=fp32_gpu`, `runtime=pytorch_fp32`) to
   `artifacts/predictions/`.  These are the FP32-GPU baseline required by
   Experiments 1 and 2.
5. Record FP32 GPU metrics (AUC, Binary Accuracy at 0.5 threshold, model size).

Phase 3 execution note:
- training and full validation are expected to run on the GCP GPU VM
- CPU runs are only for smoke tests and correctness checks

## Phase 4: Export and Quantization

1. Export PyTorch checkpoint to ONNX.
2. Convert ONNX to TensorFlow.
3. Export TFLite variants:
   - Dynamic Range
   - Full Integer INT8 Static
   - Float16
4. Run parity checks against FP32 outputs on a small held-out sample.

## Phase 5: Mobile Inference

1. Integrate all **six** TFLite models into `android_app/` (2 tasks × 3 quant
   variants: `dynamic_range`, `float16`, `int8_static`).
2. Implement single-image inference pipeline in Kotlin:
   - Load `(1, 224, 224, 3)` float32 NHWC tensors.
   - Read `score_fake` output directly (sigmoid baked in).
3. Push the FF++ test-split face-crop images onto the device and run **full
   test-set inference** for all three quantization variants.
4. For each quant variant write a `FramePredictionRow` CSV to
   `artifacts/predictions/` (fields: `task_id`, `quant_id`, `split`,
   `video_id`, `frame_idx`, `image_path`, `manipulation_type`,
   `binary_label`, `score_fake`, `pred_label`, `latency_ms`,
   `device_name`, `runtime`).
5. Aggregate frame-level predictions to video-level with majority vote and
   write `VideoPredictionRow` CSVs to `artifacts/predictions/`.
6. Record per-variant: mean per-frame latency (ms), FPS, model size (MB),
   and device specs (device name, Android API level, chipset).

## Phase 6: Analysis

Inputs: `FramePredictionRow` and `VideoPredictionRow` CSVs from Phase 3
(FP32-GPU baseline) and Phase 5 (all three on-device quant variants).

1. Compute overall frame-level metrics per condition:
   - AUC (primary, threshold-independent).
   - Binary Accuracy at threshold 0.5.
2. Compute per-class AUC and Accuracy for DF and NT separately.
3. Compute quantization gap per method:
   `AUC(FP32-GPU) − AUC(quantized)` for Dynamic Range, Float16, INT8 Static.
4. Build Experiment 1 table: FP32-GPU vs INT8-Mobile — AUC, Accuracy, FPS,
   model size, quantization gap.
5. Build Experiment 2 table: per-class (DF vs NT) AUC and gap under
   FP32-GPU and INT8-Mobile; confirm gap is larger for NT than DF.
6. Build Experiment 3 table: all three quant methods — AUC (overall + per-class),
   Accuracy, FPS, model size, quantization gap.
7. Mine NT failure examples: identify NT test frames where FP32-GPU predicts
   correctly but INT8-Mobile misclassifies.
8. Produce qualitative failure grid: 4–6 NT failure frames shown side-by-side
   (frame image, FP32 score, INT8 score) to ground the numerical findings
   visually.
9. Compute video-level accuracy and majority-vote FPS from `VideoPredictionRow`
   CSVs.

## Immediate Next Steps

1. Configure the GCP VM environment and runtime paths.
2. Implement Phase 2 preprocessing code with CPU/GPU compatibility.
3. Test on a few videos locally.
4. Run the full preprocessing job on the VM.
