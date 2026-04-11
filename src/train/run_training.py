from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common.constants import TASK_DF_VS_REAL, TASK_NT_VS_REAL
from src.common.runtime import resolve_device
from src.train.dataset import create_dataloaders
from src.train.model import create_mobilenetv2_classifier, freeze_backbone
from src.train.trainer import ModelTrainer, TrainingConfig
from src.train.validation import validate_model, save_metrics, save_frame_predictions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FP32 MobileNetV2 baselines for DF-vs-real and NT-vs-real."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use: auto (GPU if available), cpu, or cuda",
    )
    parser.add_argument(
        "--task-id",
        choices=[TASK_DF_VS_REAL, TASK_NT_VS_REAL],
        default=None,
        help="Specific task to train. If None, train both.",
    )
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone, train only classifier")
    return parser


def train_task(
    task_id: str,
    device: str = "auto",
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    freeze_backbone_flag: bool = False,
) -> dict:
    """Train a single task. Enforces Phase 3 gate: validation AUC > 0.85."""
    print(f"\n{'='*70}")
    print(f"Training {task_id}")
    print(f"{'='*70}")

    # Device
    runtime = resolve_device(device)
    print(f"Device: {runtime.resolved_device} (requested: {runtime.requested_device})")

    # Config
    config = TrainingConfig(
        task_id=task_id,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Dataloaders
    print(f"\nLoading data for {task_id}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        task_id=task_id,
        batch_size=batch_size,
        num_workers=0,
        augment=True,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Model
    print(f"\nCreating MobileNetV2 model...")
    model = create_mobilenetv2_classifier(num_classes=1, pretrained=True)
    
    if freeze_backbone_flag:
        print("Freezing backbone, training only classifier...")
        freeze_backbone(model, freeze=True)
    else:
        print("Training full model (backbone + classifier)...")
        freeze_backbone(model, freeze=False)

    # Trainer
    trainer = ModelTrainer(model, config, runtime)

    # Train
    print(f"\nStarting training...")
    train_result = trainer.train(train_loader, val_loader)
    
    # Check Phase 3 gate: validation AUC > 0.85
    best_val_auc = train_result["best_auc"]
    print(f"\n{'='*70}")
    print(f"PHASE 3 VALIDATION GATE")
    print(f"{'='*70}")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    
    if best_val_auc < 0.85:
        print(f"⚠️  GATE FAILURE: Validation AUC ({best_val_auc:.4f}) < 0.85")
        print(f"Model does not meet Phase 3 gate requirement.")
        print(f"Cannot proceed to export phase.")
        return {
            "task_id": task_id,
            "training": train_result,
            "testing": None,
            "validation_gate_passed": False,
            "gate_auc": best_val_auc,
            "checkpoint_path": None,
        }
    else:
        print(f"✓ GATE PASSED: Validation AUC ({best_val_auc:.4f}) >= 0.85")
        print(f"Model approved for export phase.")

    # Test/Validate on full test set
    print(f"\n{'='*70}")
    print(f"FINAL TEST EVALUATION")
    print(f"{'='*70}")
    metrics_test, test_preds, test_labels = validate_model(
        model,
        test_loader,
        device=runtime.resolved_device,
        task_id=task_id,
        split="test",
        quant_id="fp32_gpu" if runtime.use_cuda else "fp32_cpu",
    )
    print(f"\nTest Metrics:")
    print(f"  Test Accuracy: {metrics_test['accuracy']:.4f}")
    print(f"  Test AUC: {metrics_test['auc']:.4f}")
    print(f"  Test ROC-AUC: {metrics_test['roc_auc']:.4f}")

    # Save test metrics
    test_metrics_path = Path("artifacts/metrics") / f"{task_id}_test_metrics.json"
    save_metrics(metrics_test, test_metrics_path)
    print(f"Saved test metrics to {test_metrics_path}")

    # Save validation metrics from training
    val_metrics_path = Path("artifacts/metrics") / f"{task_id}_validation_metrics.json"
    val_metrics = {
        "task_id": task_id,
        "split": "val",
        "best_auc": best_val_auc,
        "note": "Best AUC achieved during training on validation set",
    }
    save_metrics(val_metrics, val_metrics_path)
    print(f"Saved validation metrics to {val_metrics_path}")

    # Save per-frame predictions
    test_pred_path = Path("artifacts/predictions") / f"{task_id}_fp32_test_frames.csv"
    save_frame_predictions(
        task_id=task_id,
        split="test",
        predictions=test_preds,
        labels=test_labels,
        quant_id="fp32_gpu" if runtime.use_cuda else "fp32_cpu",
        output_path=test_pred_path,
    )
    print(f"Saved test frame predictions to {test_pred_path}")

    # Get checkpoint path
    checkpoint_path = Path("artifacts/checkpoints") / f"{task_id}_best.pth"

    # Aggregate result
    result = {
        "task_id": task_id,
        "training": train_result,
        "testing": metrics_test,
        "validation_gate_passed": True,
        "gate_auc": best_val_auc,
        "checkpoint_path": str(checkpoint_path),
    }

    return result


def main() -> None:
    args = build_arg_parser().parse_args()

    tasks = [args.task_id] if args.task_id else [TASK_DF_VS_REAL, TASK_NT_VS_REAL]
    results = {}

    for task_id in tasks:
        result = train_task(
            task_id=task_id,
            device=args.device,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            freeze_backbone_flag=args.freeze_backbone,
        )
        results[task_id] = result

    # Summary
    print(f"\n{'='*70}")
    print("PHASE 3 TRAINING SUMMARY")
    print(f"{'='*70}")

    for task_id, result in results.items():
        print(f"\n{task_id}:")
        print(f"  Validation AUC: {result['gate_auc']:.4f}")
        print(f"  Gate Passed: {'✓ YES' if result['validation_gate_passed'] else '✗ NO'}")
        if result['testing']:
            print(f"  Test AUC: {result['testing']['auc']:.4f}")
            print(f"  Test Accuracy: {result['testing']['accuracy']:.4f}")
        print(f"  Checkpoint: {result['checkpoint_path']}")

    # Save summary
    summary_path = Path("artifacts/metrics/phase3_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")

    # Check if all tasks passed gate
    all_passed = all(r.get("validation_gate_passed", False) for r in results.values())
    if all_passed:
        print(f"\n✓ All tasks passed Phase 3 validation gate (AUC > 0.85)")
        print(f"✓ Ready to proceed to Phase 4 (Export & Quantization)")
    else:
        print(f"\n✗ Some tasks failed Phase 3 validation gate")
        print(f"✗ Cannot proceed to Phase 4 until validation improves")


if __name__ == "__main__":
    main()
