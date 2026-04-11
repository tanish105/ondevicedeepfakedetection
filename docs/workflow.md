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
4. Record FP32 GPU metrics and prediction files.

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

1. Integrate TFLite models into `android_app/`.
2. Run single-image inference.
3. Benchmark sampled FF++ test frames on device.
4. Record latency, FPS, model size, and device specs.

## Phase 6: Analysis

1. Compute overall frame-level and video-level metrics.
2. Compute per-class DF and NT AUC.
3. Compute quantization gaps.
4. Mine NT failure examples.
5. Build experiment tables.

## Immediate Next Steps

1. Configure the GCP VM environment and runtime paths.
2. Implement Phase 2 preprocessing code with CPU/GPU compatibility.
3. Test on a few videos locally.
4. Run the full preprocessing job on the VM.
