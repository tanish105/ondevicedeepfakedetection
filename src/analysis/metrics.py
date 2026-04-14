"""Frame-level and video-level metric computations for Phase 6."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

from sklearn.metrics import roc_auc_score

from src.common.schemas import FramePredictionRow


def frame_metrics(rows: List[FramePredictionRow]) -> dict:
    """Compute AUC, binary accuracy (threshold=0.5), and frame count.

    Returns:
        {"auc": float, "accuracy": float, "num_frames": int}
    """
    labels = [r.binary_label for r in rows]
    scores = [r.score_fake for r in rows]
    preds = [r.pred_label for r in rows]

    if len(set(labels)) < 2:
        raise ValueError(
            f"frame_metrics requires both classes present in labels; "
            f"got only {set(labels)} across {len(rows)} rows"
        )

    auc = float(roc_auc_score(labels, scores))
    correct = sum(p == l for p, l in zip(preds, labels))
    accuracy = correct / len(rows)

    return {"auc": auc, "accuracy": accuracy, "num_frames": len(rows)}


def fp32_video_accuracy(frame_rows: List[FramePredictionRow]) -> Tuple[float, int]:
    """Compute video-level accuracy via majority vote from frame predictions.

    Used for the FP32 baseline which has no pre-built video CSVs.
    Tie-breaking: equal counts of 0 and 1 resolve to 0 (predicted real).

    Returns:
        (accuracy, num_videos)
    """
    votes: dict[str, list[int]] = defaultdict(list)
    labels: dict[str, int] = {}
    for row in frame_rows:
        votes[row.video_id].append(row.pred_label)
        labels[row.video_id] = row.binary_label

    correct = 0
    for video_id, preds in votes.items():
        majority = 1 if preds.count(1) > preds.count(0) else 0
        if majority == labels[video_id]:
            correct += 1

    num_videos = len(votes)
    return (correct / num_videos if num_videos > 0 else 0.0, num_videos)
