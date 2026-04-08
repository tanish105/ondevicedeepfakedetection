# GCP VM Setup

This project uses a GCP-first execution model for heavy jobs and a local laptop for code development and small-scale testing.

## Project

- GCP project id: `ondevicedeepfakedetection`

## Recommended VM

Primary recommendation for this project and budget:
- machine family: `N1`
- machine type: `n1-standard-4`
- GPU: `1 x NVIDIA T4`
- boot disk: `100 GB` balanced persistent disk minimum
- OS / image: Deep Learning VM image family `pytorch-2-7-cu128-ubuntu-2204-nvidia-570`
- zone: choose a T4-supported zone with available quota, preferably in `us-central1`

Why this profile:
- low enough cost to fit a `~$50` student budget if stopped when idle
- enough GPU memory for MobileNetV2 fine-tuning and preprocessing support
- prebuilt PyTorch/CUDA environment reduces setup risk

Official references:
- GPU pricing: https://cloud.google.com/compute/gpus-pricing
- GPU VM creation: https://cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose
- Deep Learning VM images: https://docs.cloud.google.com/deep-learning-vm/docs/images
- GPU machine types: https://cloud.google.com/compute/docs/gpus

## Budget Guidance

Reference pricing from Google Cloud documentation:
- `NVIDIA T4` in `us-central1`: `$0.35/hour` per GPU

Practical guidance:
- expect additional charges for vCPU, memory, and disk
- stop the VM whenever you are not actively preprocessing or training
- do not leave the GPU attached overnight unless a run is intentional

## What Runs Where

### Local laptop

Use the laptop for:
- editing code
- Git operations
- small CPU smoke tests
- inspecting manifests and metrics
- report generation
- mobile app scaffolding

### GCP GPU VM

Use the VM for:
- full frame extraction
- MTCNN face cropping on the full dataset
- FP32 training
- validation and test inference
- quantization/export experiments

## Startup Checklist

1. Confirm billing is enabled for project `ondevicedeepfakedetection`.
2. Request GPU quota if `T4` quota is zero in the target region.
3. Create a Compute Engine VM:
   - machine type `n1-standard-4`
   - `1 x nvidia-tesla-t4`
   - Deep Learning VM image family `pytorch-2-7-cu128-ubuntu-2204-nvidia-570`
   - `100 GB` or larger balanced persistent disk
4. Enable firewall access for SSH only if needed.
5. SSH into the VM.
6. Verify the environment:
   - `nvidia-smi`
   - `python -c "import torch; print(torch.cuda.is_available())"`
7. Clone the repo onto the VM.
8. Copy or download required data and large artifacts to the VM storage location.
9. Run a tiny smoke test before any full preprocessing or training run.
10. Stop the VM after each work session.

## Suggested First Commands On The VM

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
git clone https://github.com/tanish105/ondevicedeepfakedetection.git
cd ondevicedeepfakedetection
git checkout phase1
```

## Storage Guidance

- Keep code in the repo workspace.
- Keep large generated data outside Git tracking.
- Recommended VM directories:
  - code: `/home/$USER/ondevicedeepfakedetection`
  - large data/artifacts: `/mnt/disks/workspace/ondevicedeepfakedetection-data`

## CPU/GPU Compatibility Rules

- default device selection should be `auto`
- use CUDA only when available
- preprocessing, training, and evaluation scripts must not assume a GPU exists
- CPU mode is valid for debugging, not for full training throughput

## Ready-To-Use gcloud Example

Adjust the zone if quota is unavailable there.

```bash
gcloud config set project ondevicedeepfakedetection

gcloud compute instances create oddd-train-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --maintenance-policy=TERMINATE \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced
```
