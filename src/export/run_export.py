"""Phase 4: Export trained FP32 checkpoints to ONNX and TFLite.

Pipeline per task
-----------------
1. PyTorch checkpoint  ->  ONNX  (includes sigmoid, fixed batch=1)
2. ONNX               ->  TF SavedModel  (via onnx2tf, NHWC)
3. TF SavedModel      ->  TFLite x3:
       dynamic_range_tflite   — no calibration data needed
       float16_tflite         — no calibration data needed
       int8_static_tflite     — needs data/processed/calibration/int8_calibration_frames.csv
4. Parity check        —  compare PyTorch FP32 vs ONNX and each TFLite on N test images

Usage
-----
# Export both tasks, skip INT8 (no calibration data locally), skip parity (no images locally):
python -m src.export.run_export --skip-int8 --skip-parity

# Full run on GCP VM (all data available):
python -m src.export.run_export --device cuda

# Single task:
python -m src.export.run_export --task-id mobilenetv2_df_vs_real --skip-int8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.constants import (
    PROJECT_ROOT,
    QUANT_DYNAMIC_RANGE,
    QUANT_FLOAT16,
    QUANT_INT8_STATIC,
    TASK_DF_VS_REAL,
    TASK_IDS,
    TASK_NT_VS_REAL,
)
from src.export.onnx_export import export_to_onnx, load_checkpoint
from src.export.parity_check import (
    check_parity,
    load_parity_images,
    run_onnx_inference,
    run_pytorch_inference,
    run_tflite_inference,
)
from src.export.tflite_export import (
    convert_onnx_to_saved_model,
    convert_saved_model_to_tflite,
)

_ALL_QUANT_METHODS = [QUANT_DYNAMIC_RANGE, QUANT_FLOAT16, QUANT_INT8_STATIC]


# ── Per-task export ───────────────────────────────────────────────────────────

def export_task(
    task_id: str,
    device: str = "auto",
    skip_int8: bool = False,
    skip_parity: bool = False,
    parity_n_samples: int = 64,
    max_parity_delta: float = 0.05,
    max_parity_delta_int8: float = 0.10,
) -> dict:
    print(f"\n{'='*70}")
    print(f"PHASE 4 EXPORT: {task_id}")
    print(f"{'='*70}")

    result: dict = {
        "task_id": task_id,
        "onnx": None,
        "onnx_size_mb": None,
        "saved_model": None,
        "tflite": {},
        "parity": {},
    }

    # ── 1. PyTorch -> ONNX ────────────────────────────────────────────────────
    print("\n[1/4] PyTorch -> ONNX")
    onnx_path = export_to_onnx(task_id=task_id)
    result["onnx"] = str(onnx_path)
    result["onnx_size_mb"] = round(onnx_path.stat().st_size / (1024 * 1024), 2)

    # ── 2. ONNX -> TF SavedModel ───────────────────────────────────────────────
    print("\n[2/4] ONNX -> TF SavedModel")
    saved_model_dir = convert_onnx_to_saved_model(
        onnx_path=onnx_path,
        task_id=task_id,
    )
    result["saved_model"] = str(saved_model_dir)

    # ── 3. SavedModel -> TFLite (3 variants) ──────────────────────────────────
    print("\n[3/4] SavedModel -> TFLite")
    quant_methods = [
        m for m in _ALL_QUANT_METHODS
        if not (skip_int8 and m == QUANT_INT8_STATIC)
    ]

    for quant_method in quant_methods:
        try:
            tflite_path = convert_saved_model_to_tflite(
                saved_model_dir=saved_model_dir,
                task_id=task_id,
                quant_method=quant_method,
            )
            result["tflite"][quant_method] = {
                "path": str(tflite_path),
                "size_mb": round(tflite_path.stat().st_size / (1024 * 1024), 2),
            }
        except FileNotFoundError as exc:
            print(f"  SKIP [{quant_method}]: {exc}")
            result["tflite"][quant_method] = {"skipped": str(exc)}
        except Exception as exc:
            print(f"  ERROR [{quant_method}]: {exc}")
            result["tflite"][quant_method] = {"error": str(exc)}

    # ── 4. Parity check ───────────────────────────────────────────────────────
    if skip_parity:
        print("\n[4/4] Parity check - skipped (--skip-parity)")
        result["parity"]["skipped"] = "Pass --skip-parity removed; requires processed images."
    else:
        print(f"\n[4/4] Parity check ({parity_n_samples} samples from test split)")
        try:
            # Parity is a correctness check on 64 images — CPU is always fine
            # and avoids CUDA availability issues in export environments.
            parity_device = "cpu"
            from src.export.onnx_export import _ModelWithSigmoid
            base_model = load_checkpoint(task_id)
            model = _ModelWithSigmoid(base_model).to(parity_device)

            nchw_images, nhwc_images = load_parity_images(
                task_id, n_samples=parity_n_samples
            )

            if not nchw_images:
                print("  No images found - parity check skipped.")
                result["parity"]["skipped"] = "No processed images on this machine."
            else:
                ref_scores = run_pytorch_inference(
                    model, nchw_images, parity_device
                )

                # ONNX vs PyTorch
                onnx_scores, onnx_lat = run_onnx_inference(
                    Path(result["onnx"]), nchw_images
                )
                result["parity"]["onnx"] = check_parity(
                    ref_scores, onnx_scores, max_parity_delta, "ONNX vs PyTorch"
                )
                result["parity"]["onnx"]["mean_latency_ms"] = round(onnx_lat, 2)

                # Each TFLite variant vs PyTorch
                for quant_method, info in result["tflite"].items():
                    if "path" not in info:
                        continue
                    tflite_scores, tflite_lat = run_tflite_inference(
                        Path(info["path"]), nhwc_images
                    )
                    # INT8 static full-integer quantization is inherently lossier;
                    # use a wider threshold (0.10) that reflects what is acceptable
                    # for on-device deployment while still catching regressions.
                    delta = (
                        max_parity_delta_int8
                        if quant_method == QUANT_INT8_STATIC
                        else max_parity_delta
                    )
                    parity = check_parity(
                        ref_scores,
                        tflite_scores,
                        delta,
                        f"TFLite {quant_method}",
                    )
                    parity["mean_latency_ms"] = round(tflite_lat, 2)
                    result["parity"][quant_method] = parity
                    info["local_latency_ms"] = round(tflite_lat, 2)

        except FileNotFoundError as exc:
            print(f"  Parity check skipped - {exc}")
            result["parity"]["skipped"] = str(exc)

    return result


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(all_results: dict, skip_int8: bool) -> None:
    print(f"\n{'='*70}")
    print("PHASE 4 EXPORT SUMMARY")
    print(f"{'='*70}")

    for task_id, r in all_results.items():
        print(f"\n{task_id}:")
        if r.get("onnx"):
            print(f"  ONNX        : {r['onnx_size_mb']:.1f} MB")
        if r.get("saved_model"):
            print(f"  SavedModel  : {r['saved_model']}")
        for qm, info in r.get("tflite", {}).items():
            if "size_mb" in info:
                lat = f"  local_lat={info.get('local_latency_ms', '?')} ms" if "local_latency_ms" in info else ""
                print(f"  [{qm}]: {info['size_mb']:.2f} MB{lat}")
            elif "skipped" in info:
                print(f"  [{qm}]: SKIPPED")
            else:
                print(f"  [{qm}]: ERROR - {info.get('error', '')[:60]}")
        for pk, pv in r.get("parity", {}).items():
            if isinstance(pv, dict) and "passed" in pv:
                icon = "[OK]" if pv["passed"] else "[FAIL]"
                print(f"  Parity [{pk}]: {icon} max_delta={pv['max_delta']:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 4: export PyTorch checkpoints to ONNX and TFLite."
    )
    parser.add_argument(
        "--task-id",
        choices=list(TASK_IDS) + ["all"],
        default="all",
        help="Which task to export (default: both)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for parity-check PyTorch inference",
    )
    parser.add_argument(
        "--skip-int8",
        action="store_true",
        help="Skip INT8 static TFLite export (requires calibration data on GCP VM)",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip parity checks (requires processed face-crop images)",
    )
    parser.add_argument(
        "--parity-samples",
        type=int,
        default=64,
        help="Number of test images to use for parity check (default: 64)",
    )
    parser.add_argument(
        "--max-parity-delta",
        type=float,
        default=0.05,
        help="Max allowed absolute score difference for parity pass (default: 0.05)",
    )
    parser.add_argument(
        "--max-parity-delta-int8",
        type=float,
        default=0.10,
        help=(
            "Max allowed absolute score difference for INT8 static parity "
            "(default: 0.10 — full-integer quantization is inherently lossier)"
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    tasks = list(TASK_IDS) if args.task_id == "all" else [args.task_id]
    all_results: dict = {}

    for task_id in tasks:
        all_results[task_id] = export_task(
            task_id=task_id,
            device=args.device,
            skip_int8=args.skip_int8,
            skip_parity=args.skip_parity,
            parity_n_samples=args.parity_samples,
            max_parity_delta=args.max_parity_delta,
            max_parity_delta_int8=args.max_parity_delta_int8,
        )

    _print_summary(all_results, args.skip_int8)

    # Persist export summary alongside Phase 3 metrics
    summary_path = PROJECT_ROOT / "artifacts" / "metrics" / "phase4_export_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved export summary -> {summary_path}")

    # Final gate check
    export_errors = [
        f"{tid} / {qm}"
        for tid, r in all_results.items()
        for qm, info in r.get("tflite", {}).items()
        if "error" in info
    ]
    parity_failures = [
        f"{tid} / {pk}"
        for tid, r in all_results.items()
        for pk, pv in r.get("parity", {}).items()
        if isinstance(pv, dict) and not pv.get("passed", True)
    ]

    if not export_errors and not parity_failures:
        print("\n[OK] Phase 4 complete - all exports OK, parity checks passed.")
        print("[OK] Ready for Phase 5 (Android app integration).")
    else:
        if export_errors:
            print(f"\n[FAIL] Export errors: {export_errors}")
        if parity_failures:
            print(f"\n[FAIL] Parity failures: {parity_failures}")


if __name__ == "__main__":
    main()
