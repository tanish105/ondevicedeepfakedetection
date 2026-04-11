from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import csv

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.common.constants import (
    DEFAULT_FRAME_SAMPLE_COUNT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MANIFESTS_BY_SPLIT_DIR,
    DEFAULT_MANIFESTS_DIR,
    PROJECT_ROOT,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TASK_DF_VS_REAL,
    TASK_NT_VS_REAL,
)
from src.common.csv_io import ensure_parent_dir, read_rows
from src.common.runtime import RuntimeContext, resolve_device
from src.common.schemas import CalibrationRow, FrameManifestRow, SplitRow

TASK_TO_SPLIT_PREFIX = {
    TASK_DF_VS_REAL: "df_vs_real",
    TASK_NT_VS_REAL: "nt_vs_real",
}


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for Phase 2 preprocessing. "
            "Install it with `python -m pip install opencv-python`."
        ) from exc
    return cv2


def _require_mtcnn():
    try:
        from facenet_pytorch import MTCNN  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "facenet-pytorch is required for MTCNN face cropping. "
            "Install it with `python -m pip install facenet-pytorch`."
        ) from exc
    return MTCNN


@dataclass(frozen=True)
class PreprocessConfig:
    task_id: str
    split: str
    frame_sample_count: int = DEFAULT_FRAME_SAMPLE_COUNT
    image_size: int = DEFAULT_IMAGE_SIZE
    device: str = "auto"
    mtcnn_margin: int = 0
    overwrite_images: bool = False
    max_videos: int | None = None
    fail_on_missing_face: bool = False


@dataclass(frozen=True)
class PreprocessSummary:
    task_id: str
    split: str
    videos_seen: int
    rows_written: int
    faces_found: int
    faces_missing: int
    manifest_path: str


def task_split_csv_path(task_id: str, split: str) -> Path:
    prefix = TASK_TO_SPLIT_PREFIX[task_id]
    return PROJECT_ROOT / "data" / "processed" / "splits" / f"{prefix}_{split}.csv"


def split_manifest_output_path(task_id: str, split: str) -> Path:
    prefix = TASK_TO_SPLIT_PREFIX[task_id]
    return DEFAULT_MANIFESTS_BY_SPLIT_DIR / f"{prefix}_{split}_frames.csv"


def aggregate_manifest_output_path(task_id: str) -> Path:
    prefix = TASK_TO_SPLIT_PREFIX[task_id]
    return DEFAULT_MANIFESTS_DIR / f"{prefix}_frames.csv"


def load_split_rows(task_id: str, split: str) -> List[SplitRow]:
    return read_rows(task_split_csv_path(task_id, split), SplitRow)


def sample_frame_indices(total_frames: int, sample_count: int) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= sample_count:
        return list(range(total_frames))
    indices = np.linspace(0, total_frames - 1, sample_count)
    return sorted({int(round(index)) for index in indices})


class MTCNNFaceCropper:
    def __init__(self, image_size: int, runtime: RuntimeContext, margin: int = 0) -> None:
        MTCNN = _require_mtcnn()
        device = runtime.resolved_device if runtime.use_cuda else "cpu"
        self._detector = MTCNN(
            image_size=image_size,
            margin=margin,
            select_largest=True,
            post_process=False,
            keep_all=False,
            device=device,
        )
        self._image_size = image_size

    def crop(self, frame_rgb: np.ndarray) -> tuple[Image.Image | None, tuple[float, float, float, float] | None]:
        image = Image.fromarray(frame_rgb)
        boxes, _ = self._detector.detect(image)
        if boxes is None or len(boxes) == 0:
            return None, None

        box = boxes[0]
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        crop = image.crop((max(0, x1), max(0, y1), max(0, x2), max(0, y2)))
        crop = crop.resize((self._image_size, self._image_size))
        return crop, (x1, y1, x2, y2)


def _video_path_from_split_row(row: SplitRow) -> Path:
    return PROJECT_ROOT / "data" / "raw" / "ffpp_c23" / row.video_path


def _output_image_path(task_id: str, split: str, video_id: str, frame_idx: int) -> Path:
    prefix = TASK_TO_SPLIT_PREFIX[task_id]
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "frames"
        / prefix
        / split
        / video_id
        / f"{frame_idx:04d}.png"
    )


def _write_manifest_rows(rows: Sequence[FrameManifestRow], output_path: Path) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FrameManifestRow.field_names())
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def rebuild_aggregate_manifest(task_id: str) -> Path:
    output_path = aggregate_manifest_output_path(task_id)
    ensure_parent_dir(output_path)
    split_paths = [
        split_manifest_output_path(task_id, split_name)
        for split_name in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)
        if split_manifest_output_path(task_id, split_name).exists()
    ]

    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=FrameManifestRow.field_names())
        writer.writeheader()
        for path in split_paths:
            with path.open("r", newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                for row in reader:
                    writer.writerow(row)
    return output_path


def run_preprocessing(config: PreprocessConfig) -> PreprocessSummary:
    cv2 = _require_cv2()
    runtime = resolve_device(config.device)
    cropper = MTCNNFaceCropper(
        image_size=config.image_size,
        runtime=runtime,
        margin=config.mtcnn_margin,
    )

    split_rows = load_split_rows(config.task_id, config.split)
    if config.max_videos is not None:
        split_rows = split_rows[: config.max_videos]

    manifest_rows: List[FrameManifestRow] = []
    faces_found = 0
    faces_missing = 0

    for split_row in tqdm(split_rows, desc=f"{config.task_id}/{config.split} videos", unit="video"):
        video_path = _video_path_from_split_row(split_row)
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video file: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_indices = sample_frame_indices(total_frames, config.frame_sample_count)

        for frame_idx, source_index in enumerate(
            tqdm(source_indices, desc=f"{split_row.video_id} frames", leave=False, unit="frame")
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                faces_missing += 1
                manifest_rows.append(
                    FrameManifestRow(
                        task_id=config.task_id,
                        video_id=split_row.video_id,
                        frame_idx=frame_idx,
                        frame_source_index=source_index,
                        image_path="",
                        face_found=0,
                        bbox_x1=None,
                        bbox_y1=None,
                        bbox_x2=None,
                        bbox_y2=None,
                        width=config.image_size,
                        height=config.image_size,
                        manipulation_type=split_row.manipulation_type,
                        binary_label=split_row.binary_label,
                        split=split_row.split,
                    )
                )
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            crop, bbox = cropper.crop(frame_rgb)
            if crop is None or bbox is None:
                faces_missing += 1
                if config.fail_on_missing_face:
                    capture.release()
                    raise RuntimeError(
                        f"No face detected for {split_row.video_id} frame {source_index}"
                    )
                manifest_rows.append(
                    FrameManifestRow(
                        task_id=config.task_id,
                        video_id=split_row.video_id,
                        frame_idx=frame_idx,
                        frame_source_index=source_index,
                        image_path="",
                        face_found=0,
                        bbox_x1=None,
                        bbox_y1=None,
                        bbox_x2=None,
                        bbox_y2=None,
                        width=config.image_size,
                        height=config.image_size,
                        manipulation_type=split_row.manipulation_type,
                        binary_label=split_row.binary_label,
                        split=split_row.split,
                    )
                )
                continue

            output_path = _output_image_path(config.task_id, config.split, split_row.video_id, frame_idx)
            ensure_parent_dir(output_path)
            if config.overwrite_images or not output_path.exists():
                crop.save(output_path)

            x1, y1, x2, y2 = bbox
            faces_found += 1
            manifest_rows.append(
                FrameManifestRow(
                    task_id=config.task_id,
                    video_id=split_row.video_id,
                    frame_idx=frame_idx,
                    frame_source_index=source_index,
                    image_path=output_path.relative_to(PROJECT_ROOT).as_posix(),
                    face_found=1,
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    width=config.image_size,
                    height=config.image_size,
                    manipulation_type=split_row.manipulation_type,
                    binary_label=split_row.binary_label,
                    split=split_row.split,
                )
            )
        capture.release()

    manifest_path = split_manifest_output_path(config.task_id, config.split)
    _write_manifest_rows(manifest_rows, manifest_path)
    rebuild_aggregate_manifest(config.task_id)

    return PreprocessSummary(
        task_id=config.task_id,
        split=config.split,
        videos_seen=len(split_rows),
        rows_written=len(manifest_rows),
        faces_found=faces_found,
        faces_missing=faces_missing,
        manifest_path=manifest_path.as_posix(),
    )


def build_int8_calibration_rows(
    manifest_rows: Sequence[FrameManifestRow],
    task_id: str,
    total_frames: int = 500,
    real_target: int = 250,
    fake_target: int = 250,
) -> List[CalibrationRow]:
    val_rows = [
        row
        for row in manifest_rows
        if row.task_id == task_id and row.split == SPLIT_VAL and row.face_found == 1
    ]

    real_rows = [row for row in val_rows if row.binary_label == 0]
    fake_rows = [row for row in val_rows if row.binary_label == 1]

    selected = real_rows[:real_target] + fake_rows[:fake_target]
    if len(selected) < total_frames:
        raise ValueError(
            f"Not enough validation face crops for calibration: requested {total_frames}, got {len(selected)}"
        )

    return [
        CalibrationRow(
            task_id=row.task_id,
            video_id=row.video_id,
            frame_idx=row.frame_idx,
            image_path=row.image_path,
            manipulation_type=row.manipulation_type,
            split=row.split,
        )
        for row in selected[:total_frames]
    ]


def build_combined_int8_calibration_rows(
    manifest_rows: Sequence[FrameManifestRow],
    task_ids: Sequence[str] = (TASK_DF_VS_REAL, TASK_NT_VS_REAL),
    total_frames_per_task: int = 500,
    real_target_per_task: int = 250,
    fake_target_per_task: int = 250,
) -> List[CalibrationRow]:
    combined_rows: List[CalibrationRow] = []
    for task_id in task_ids:
        combined_rows.extend(
            build_int8_calibration_rows(
                manifest_rows=manifest_rows,
                task_id=task_id,
                total_frames=total_frames_per_task,
                real_target=real_target_per_task,
                fake_target=fake_target_per_task,
            )
        )
    return combined_rows
