from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.common.constants import PROJECT_ROOT

IMAGE_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_nchw(image_path: Path) -> np.ndarray:
    """Return (1, 3, H, W) float32 — PyTorch / ONNX input format."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis, ...]    # (1, 3, H, W)


def _load_nhwc(image_path: Path) -> np.ndarray:
    """Return (1, H, W, 3) float32 — TFLite input format."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr[np.newaxis, ...]                        # (1, H, W, 3)


# ── Image loader ──────────────────────────────────────────────────────────────

def load_parity_images(
    task_id: str,
    n_samples: int = 64,
    split: str = "test",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load up to n_samples images from the processed manifest.

    Returns:
        (nchw_list, nhwc_list) — parallel lists, one array per image.

    Raises:
        FileNotFoundError if the manifest doesn't exist (Phase 2 not run).
    """
    from src.common.csv_io import read_rows
    from src.common.schemas import FrameManifestRow
    from src.data.ffpp_preprocess import aggregate_manifest_output_path

    manifest_path = aggregate_manifest_output_path(task_id)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run the Phase 2 preprocessing pipeline first."
        )

    rows = read_rows(manifest_path, FrameManifestRow)
    rows = [r for r in rows if r.split == split and r.face_found == 1][:n_samples]

    nchw_list, nhwc_list = [], []
    missing = 0
    for row in rows:
        img_path = PROJECT_ROOT / row.image_path
        if img_path.exists():
            nchw_list.append(_load_nchw(img_path))
            nhwc_list.append(_load_nhwc(img_path))
        else:
            missing += 1

    if missing:
        print(f"  WARNING: {missing}/{len(rows)} images not found on disk (run Phase 2).")

    return nchw_list, nhwc_list


# ── Inference runners ─────────────────────────────────────────────────────────

def run_pytorch_inference(
    model: nn.Module,
    images_nchw: List[np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """Run model (must include sigmoid) on NCHW images. Returns score_fake array."""
    model.eval()
    scores = []
    with torch.no_grad():
        for img in images_nchw:
            tensor = torch.from_numpy(img).to(device)
            score = model(tensor).cpu().numpy().flatten()[0]
            scores.append(float(score))
    return np.array(scores, dtype=np.float32)


def run_onnx_inference(
    onnx_path: Path,
    images_nchw: List[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """Run ONNX model on NCHW images. Returns (score_fake array, mean_latency_ms).

    The ONNX model already includes sigmoid, so output is score_fake directly.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name

    scores, latencies = [], []
    for img in images_nchw:
        t0 = time.perf_counter()
        output = sess.run(None, {input_name: img})[0]
        latencies.append((time.perf_counter() - t0) * 1000)
        scores.append(float(output.flatten()[0]))

    return np.array(scores, dtype=np.float32), float(np.mean(latencies))


def run_tflite_inference(
    tflite_path: Path,
    images_nhwc: List[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """Run TFLite model on NHWC images. Returns (score_fake array, mean_latency_ms).

    Tries ai_edge_litert first (installed with onnx2tf), falls back to
    tensorflow.lite.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            import tflite_runtime.interpreter as _tflite
            Interpreter = _tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    scores, latencies = [], []
    for img in images_nhwc:
        interpreter.set_tensor(input_details[0]["index"], img)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)
        raw = interpreter.get_tensor(output_details[0]["index"]).flatten()[0]
        # The TFLite model includes sigmoid, so raw output is already score_fake.
        # Clamp to [0, 1] in case of tiny floating-point overshoot.
        scores.append(float(np.clip(raw, 0.0, 1.0)))

    return np.array(scores, dtype=np.float32), float(np.mean(latencies))


# ── Parity comparison ─────────────────────────────────────────────────────────

def check_parity(
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
    max_delta: float = 0.05,
    label: str = "",
) -> dict:
    """Compare candidate scores to a PyTorch FP32 reference.

    Args:
        reference_scores: score_fake values from PyTorch FP32.
        candidate_scores: score_fake values from ONNX or TFLite.
        max_delta:        maximum allowed absolute difference per sample.
        label:            human-readable name for logging.

    Returns:
        dict with keys: label, n_samples, max_delta, mean_delta, threshold, passed.
    """
    deltas = np.abs(reference_scores - candidate_scores)
    max_obs = float(np.max(deltas))
    mean_obs = float(np.mean(deltas))
    passed = max_obs <= max_delta

    status = "PASS" if passed else "FAIL"
    print(
        f"  Parity [{label}]: {status} | "
        f"max_delta={max_obs:.4f} | mean_delta={mean_obs:.4f} | "
        f"threshold={max_delta}"
    )

    return {
        "label": label,
        "n_samples": int(len(reference_scores)),
        "max_delta": round(max_obs, 6),
        "mean_delta": round(mean_obs, 6),
        "threshold": max_delta,
        "passed": passed,
    }
