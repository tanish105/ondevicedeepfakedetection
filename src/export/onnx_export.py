from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.common.constants import PROJECT_ROOT, TASK_IDS
from src.train.model import create_mobilenetv2_classifier

# ONNX opset — 17 covers all MobileNetV2 ops cleanly
ONNX_OPSET = 17
IMAGE_SIZE = 224


class _ModelWithSigmoid(nn.Module):
    """Wraps the classifier so the exported model outputs probabilities (0-1).

    This avoids applying sigmoid in the Android app — the raw output is
    score_fake directly.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def load_checkpoint(
    task_id: str,
    checkpoint_path: Path | None = None,
) -> nn.Module:
    """Load a trained FP32 checkpoint and return eval-mode model."""
    if task_id not in TASK_IDS:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {TASK_IDS}")

    if checkpoint_path is None:
        checkpoint_path = (
            PROJECT_ROOT / "artifacts" / "checkpoints" / f"{task_id}_best.pth"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run Phase 3 training first, or copy the checkpoint from the GCP VM."
        )

    model = create_mobilenetv2_classifier(num_classes=1, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_onnx(
    task_id: str,
    output_dir: Path | None = None,
    checkpoint_path: Path | None = None,
) -> Path:
    """Export a trained PyTorch checkpoint to ONNX.

    The exported model includes a sigmoid layer so outputs are probabilities
    in [0, 1] — score_fake can be read directly without post-processing.

    Input shape:  (1, 3, 224, 224)  float32  NCHW
    Output shape: (1, 1)            float32  score_fake in [0, 1]

    Returns:
        Path to the saved .onnx file.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "exports" / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / f"{task_id}.onnx"

    print(f"  Loading checkpoint for {task_id}...")
    base_model = load_checkpoint(task_id, checkpoint_path)
    model = _ModelWithSigmoid(base_model)
    model.eval()

    dummy_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print(f"  Exporting to ONNX (opset {ONNX_OPSET}): {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["score_fake"],
        # Fixed batch=1: simplifies onnx2tf conversion and matches mobile use case
        dynamic_axes=None,
    )

    # Verify the exported graph is valid
    import onnx
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX export OK - {size_mb:.1f} MB")
    return onnx_path
