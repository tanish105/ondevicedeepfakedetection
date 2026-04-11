from __future__ import annotations

from pathlib import Path
from typing import Generator, List

import numpy as np
from PIL import Image

from src.common.constants import (
    PROJECT_ROOT,
    QUANT_DYNAMIC_RANGE,
    QUANT_FLOAT16,
    QUANT_INT8_STATIC,
)
from src.common.csv_io import read_rows
from src.common.schemas import CalibrationRow

IMAGE_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Directory layout ──────────────────────────────────────────────────────────
#  artifacts/exports/
#    onnx/                      -> written by onnx_export.py
#    saved_model/{task_id}/     -> written by convert_onnx_to_saved_model()
#    tflite/                    -> written by convert_saved_model_to_tflite()

_TFLITE_SUFFIX = {
    QUANT_DYNAMIC_RANGE: "dynamic_range",
    QUANT_INT8_STATIC: "int8_static",
    QUANT_FLOAT16: "float16",
}


# ── Preprocessing helper ─────────────────────────────────────────────────────

def _preprocess_nhwc(image_path: Path) -> np.ndarray:
    """Load one image as NHWC float32, ImageNet-normalised."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0        # (H, W, 3)
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr[np.newaxis, ...]                            # (1, H, W, 3)


# ── Step 1: ONNX -> TF SavedModel ─────────────────────────────────────────────

def convert_onnx_to_saved_model(
    onnx_path: Path,
    task_id: str,
    output_dir: Path | None = None,
) -> Path:
    """Convert an ONNX model to a TF SavedModel using onnx2tf.

    onnx2tf transposes from PyTorch's NCHW to TF-native NHWC automatically.
    The SavedModel is saved at:
        {output_dir}/{task_id}/

    Returns:
        Path to the SavedModel directory.
    """
    import onnx2tf

    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "exports" / "saved_model"

    saved_model_dir = output_dir / task_id
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Converting ONNX -> TF SavedModel: {saved_model_dir}")

    # onnx2tf 1.28.x calls download_test_image_data() during its internal
    # verification pass.  That function fetches a remote .npy file which is
    # incompatible with numpy 2.x's stricter pickle handling.  We patch the
    # function to return a locally-generated dummy image instead, which is
    # sufficient for the conversion to proceed correctly.
    import numpy as _np
    # onnx2tf.onnx2tf imports download_test_image_data by name at module load,
    # so we must patch it in that module's namespace (not in common_functions).
    import onnx2tf.onnx2tf as _onnx2tf_mod

    _orig_download = _onnx2tf_mod.download_test_image_data

    def _dummy_test_image() -> _np.ndarray:
        # Provide a locally-generated dummy input so onnx2tf's verification
        # step never tries to fetch a remote .npy file.
        # Shape matches our fixed export: (1, 3, 224, 224) float32 NCHW.
        rng = _np.random.default_rng(seed=42)
        return rng.standard_normal((1, 3, 224, 224)).astype(_np.float32)

    _onnx2tf_mod.download_test_image_data = _dummy_test_image
    try:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(saved_model_dir),
            batch_size=1,
            not_use_onnxsim=True,
            verbosity="error",
        )
    finally:
        _onnx2tf_mod.download_test_image_data = _orig_download  # always restore

    print(f"  SavedModel OK: {saved_model_dir}")
    return saved_model_dir


# ── Step 2a: calibration dataset for INT8 static ────────────────────────────

def _build_calibration_generator(
    calibration_csv: Path,
    n_frames: int = 500,
) -> Generator:
    """Return a representative-dataset generator for INT8 static quantization.

    Each call to the generator yields a list with one NHWC float32 tensor
    of shape (1, 224, 224, 3), matching the SavedModel input.
    """
    rows: List[CalibrationRow] = read_rows(calibration_csv, CalibrationRow)[:n_frames]
    image_paths = [PROJECT_ROOT / row.image_path for row in rows]
    valid_paths = [p for p in image_paths if p.exists()]

    if not valid_paths:
        raise FileNotFoundError(
            f"No calibration images found under paths listed in {calibration_csv}.\n"
            "Ensure the face-crop pipeline has been run on the GCP VM."
        )

    print(f"  Calibration: {len(valid_paths)}/{len(image_paths)} images available")

    def _generator():
        for img_path in valid_paths:
            yield [_preprocess_nhwc(img_path)]

    return _generator


# ── Step 2b: SavedModel -> TFLite ─────────────────────────────────────────────

def convert_saved_model_to_tflite(
    saved_model_dir: Path,
    task_id: str,
    quant_method: str,
    output_dir: Path | None = None,
    calibration_csv: Path | None = None,
    calibration_n_frames: int = 500,
) -> Path:
    """Convert a TF SavedModel to TFLite for one quantization method.

    Supported quant_method values (from constants):
        QUANT_DYNAMIC_RANGE  — weights quantized at runtime, float32 I/O
        QUANT_INT8_STATIC    — full-integer quantization, float32 I/O kept for
                               easier Android integration
        QUANT_FLOAT16        — weights stored as float16, float32 I/O

    Returns:
        Path to the .tflite file.
    """
    import tensorflow as tf

    if quant_method not in _TFLITE_SUFFIX:
        raise ValueError(
            f"Unknown quant_method '{quant_method}'. "
            f"Valid: {list(_TFLITE_SUFFIX)}"
        )

    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "exports" / "tflite"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = _TFLITE_SUFFIX[quant_method]
    tflite_path = output_dir / f"{task_id}_{suffix}.tflite"

    print(f"  Converting SavedModel -> TFLite [{quant_method}]: {tflite_path.name}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if quant_method == QUANT_DYNAMIC_RANGE:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif quant_method == QUANT_FLOAT16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif quant_method == QUANT_INT8_STATIC:
        if calibration_csv is None:
            calibration_csv = (
                PROJECT_ROOT
                / "data"
                / "processed"
                / "calibration"
                / "int8_calibration_frames.csv"
            )
        if not calibration_csv.exists():
            raise FileNotFoundError(
                f"Calibration CSV not found: {calibration_csv}\n"
                "INT8 static quantization requires 500 val-split face crops.\n"
                "Run Phase 2 on the GCP VM, then copy the calibration manifest here."
            )

        gen = _build_calibration_generator(calibration_csv, calibration_n_frames)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = gen
        # Keep float32 I/O so the Android app doesn't need to dequantize
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)

    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    print(f"  TFLite [{suffix}] OK - {size_mb:.2f} MB")
    return tflite_path
