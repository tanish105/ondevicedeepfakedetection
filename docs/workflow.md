# Workflow

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

## Phase 3: FP32 Training

1. Train `mobilenetv2_df_vs_real`.
2. Train `mobilenetv2_nt_vs_real`.
3. Validate both and gate export on `AUC > 0.85`.
4. Record FP32 GPU metrics and prediction files.

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

1. Add config files for local dataset paths and experiment defaults.
2. Implement split generation and manifest-writing utilities first.
3. Only then start training and export work.
