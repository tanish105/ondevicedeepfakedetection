# On-Device Deepfake Detection

This repository is structured around the project proposal: train MobileNetV2 on FaceForensics++ (`c23`) for Deepfakes and NeuralTextures, measure FP32-GPU vs on-device quantized performance, and study whether quantization hurts subtle manipulations more.

## Scope

Included experiments:
- Experiment 1: FP32-GPU vs INT8-Mobile accuracy/efficiency comparison
- Experiment 2: Per-class quantization gap analysis for DF vs NT
- Experiment 3: Quantization method comparison across Dynamic Range, Full Integer INT8 Static, and Float16

## Repository Layout

```text
.
|-- configs/                  # Versioned experiment and path configs
|-- data/
|   |-- raw/                  # FF++ source videos, never committed
|   |-- interim/              # Temporary extraction outputs, never committed
|   |-- processed/            # Face crops, manifests, split files, never committed
|-- docs/
|   |-- project_contract.md   # Shared A/B ownership, schemas, metric contract
|   |-- workflow.md           # Execution order and milestone sequence
|-- src/
|   |-- data/                 # FF++ indexing, frame extraction, face crop, manifests
|   |-- train/                # Training loops, datasets, losses, checkpoints
|   |-- eval/                 # Metrics, ROC/AUC, per-class analysis, failure mining
|   |-- export/               # ONNX / TensorFlow / TFLite export and quantization
|   |-- mobile/               # Device-side input/output helpers and benchmarking glue
|   |-- common/               # Shared constants, schemas, utilities
|-- android_app/              # Kotlin app for single-image and batch frame inference
|-- artifacts/
|   |-- checkpoints/          # Trained FP32 weights
|   |-- exports/              # ONNX, SavedModel, TFLite outputs
|   |-- metrics/              # CSV/JSON metric outputs and summary tables
|   |-- predictions/          # FP32 and mobile prediction files
|   |-- failures/             # Qualitative failure frames for poster/video
```

## Core Decisions

- Dataset: FaceForensics++ `c23`
- Manipulation types: `df` and `nt`
- Negative class: `real`
- Split: official `720/140/140` at the video level
- Inference unit: frame-level model, video-level majority-vote aggregation
- Input preprocessing: 25 uniformly sampled frames per video, MTCNN face crop, resize to `224x224`
- Backbone: ImageNet-pretrained MobileNetV2, fine-tuned separately for DF-vs-real and NT-vs-real
- Development model: local Windows laptop for coding/debugging, GCP GPU VM for heavy preprocessing and training
- Runtime compatibility: all Python pipeline code must run on CPU by default and use GPU automatically when available
- Mobile target: on-device deployment remains mobile-platform-neutral until Android vs iOS is finalized

## First Milestones

1. Finalize dataset manifests and split files.
2. Implement frame extraction and face crop pipeline.
3. Train and validate FP32 baselines for DF and NT.
4. Export and verify quantized models.
5. Run Android benchmarking and evaluation.

See `docs/project_contract.md` for the interface contract.
See `docs/gcp_vm_setup.md` for the recommended VM profile and usage checklist.
