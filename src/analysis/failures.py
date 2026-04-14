"""Mine NT failure frames and render the qualitative failure grid."""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.common.schemas import FailureMetadataRow, FramePredictionRow


def mine_nt_failures(
    fp32_rows: List[FramePredictionRow],
    int8_rows: List[FramePredictionRow],
    n: int = 20,
) -> List[FailureMetadataRow]:
    """Find NT frames where FP32-GPU is correct but INT8-Mobile is wrong.

    Criteria:
    - binary_label == 1 (fake frame)
    - FP32 pred_label == 1 (correct)
    - INT8 pred_label == 0 (wrong)

    Sorted descending by (fp32_score_fake - int8_score_fake). Returns top n.
    """
    # Index int8 rows by (video_id, frame_idx)
    int8_index: dict[tuple[str, int], FramePredictionRow] = {
        (r.video_id, r.frame_idx): r for r in int8_rows
    }

    candidates: list[tuple[float, FailureMetadataRow]] = []
    for fp32_row in fp32_rows:
        key = (fp32_row.video_id, fp32_row.frame_idx)
        int8_row = int8_index.get(key)
        if int8_row is None:
            continue
        # Must be a fake frame where fp32 is right but int8 is wrong
        if fp32_row.binary_label != 1:
            continue
        if fp32_row.pred_label != 1:
            continue
        if int8_row.pred_label != 0:
            continue

        delta = fp32_row.score_fake - int8_row.score_fake
        failure = FailureMetadataRow(
            video_id=fp32_row.video_id,
            frame_idx=fp32_row.frame_idx,
            image_path=fp32_row.image_path,
            fp32_score_fake=fp32_row.score_fake,
            int8_score_fake=int8_row.score_fake,
            binary_label=fp32_row.binary_label,
            selection_reason="fp32_correct_int8_wrong",
        )
        candidates.append((delta, failure))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in candidates[:n]]


def render_failure_grid(
    failures: List[FailureMetadataRow],
    frames_root: Path,
    output_path: Path,
    n: int = 6,
) -> None:
    """Render a 2×3 grid PNG of the top n NT failure frames.

    Each cell shows the frame image with a caption:
    "FP32: {fp32_score:.3f} | INT8: {int8_score:.3f}"

    Args:
        failures: Ordered list of FailureMetadataRow (best first).
        frames_root: Root path; image_path values are relative to this.
        output_path: Where to save the PNG.
        n: Number of frames to show (default 6 for a 2×3 grid).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    selected = failures[:n]
    cols = 3
    rows = (len(selected) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes_flat:
        ax.axis("off")

    for i, failure in enumerate(selected):
        img_path = frames_root / failure.image_path
        img = Image.open(img_path).convert("RGB")
        axes_flat[i].imshow(img)
        axes_flat[i].set_title(
            f"FP32: {failure.fp32_score_fake:.3f} | INT8: {failure.int8_score_fake:.3f}",
            fontsize=8,
        )
        axes_flat[i].axis("off")

    fig.suptitle("NT Failure Frames: FP32-correct, INT8-wrong", fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
