from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm

from src.common.csv_io import read_rows, write_rows
from src.common.schemas import FramePredictionRow, FrameManifestRow


def validate_model(
    model: nn.Module,
    test_loader,
    device: str,
    task_id: str,
    split: str = "test",
    quant_id: str = "fp32_gpu",
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Validate model and compute metrics.

    Returns:
        (metrics_dict, predictions, labels)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Validating on {split}", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    accuracy = np.mean((all_preds > 0.5) == all_labels)
    auc_score = roc_auc_score(all_labels, all_preds)

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "task_id": task_id,
        "split": split,
        "quant_id": quant_id,
        "accuracy": float(accuracy),
        "auc": float(auc_score),
        "roc_auc": float(roc_auc),
        "num_samples": len(all_labels),
        "num_real": int(np.sum(all_labels == 0)),
        "num_fake": int(np.sum(all_labels == 1)),
    }

    return metrics, all_preds, all_labels


def save_metrics(
    metrics: dict,
    output_path: Path,
) -> None:
    """Save metrics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(metrics, f, indent=2)


def save_frame_predictions(
    task_id: str,
    split: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    quant_id: str,
    output_path: Path,
) -> None:
    """Save per-frame predictions to CSV."""
    # Load manifest to get metadata
    from src.data.ffpp_preprocess import aggregate_manifest_output_path

    manifest_path = aggregate_manifest_output_path(task_id)
    all_rows = read_rows(manifest_path, FrameManifestRow)
    split_rows = [row for row in all_rows if row.split == split and row.face_found == 1]

    if len(predictions) != len(split_rows):
        raise ValueError(
            f"Mismatch: got {len(predictions)} predictions but {len(split_rows)} manifest rows"
        )

    # Build prediction rows
    prediction_rows = []
    for i, (pred_score, label, manifest_row) in enumerate(zip(predictions, labels, split_rows)):
        pred_label = 1 if pred_score > 0.5 else 0
        latency_ms = 0.0  # Placeholder, can be filled during actual inference

        row = FramePredictionRow(
            task_id=task_id,
            quant_id=quant_id,
            split=split,
            video_id=manifest_row.video_id,
            frame_idx=manifest_row.frame_idx,
            image_path=manifest_row.image_path,
            manipulation_type=manifest_row.manipulation_type,
            binary_label=int(label),
            score_fake=float(pred_score),
            pred_label=pred_label,
            latency_ms=latency_ms,
            device_name="unknown",
            runtime="pytorch_fp32",
        )
        prediction_rows.append(row)

    # Write to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(output_path, prediction_rows, FramePredictionRow)

