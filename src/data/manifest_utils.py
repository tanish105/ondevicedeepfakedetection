from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from src.common.constants import DEFAULT_CALIBRATION_DIR, DEFAULT_MANIFESTS_DIR
from src.common.csv_io import write_rows
from src.common.schemas import CalibrationRow, FrameManifestRow


def write_frame_manifest(rows: Iterable[FrameManifestRow], output_path: Path) -> None:
    write_rows(output_path, rows, FrameManifestRow)


def write_task_frame_manifest(task_prefix: str, rows: Iterable[FrameManifestRow]) -> Path:
    output_path = DEFAULT_MANIFESTS_DIR / f"{task_prefix}_frames.csv"
    write_rows(output_path, rows, FrameManifestRow)
    return output_path


def write_calibration_manifest(rows: Iterable[CalibrationRow], output_path: Path) -> None:
    write_rows(output_path, rows, CalibrationRow)


def write_default_calibration_manifest(rows: Iterable[CalibrationRow]) -> Path:
    output_path = DEFAULT_CALIBRATION_DIR / "int8_calibration_frames.csv"
    write_rows(output_path, rows, CalibrationRow)
    return output_path


def filter_valid_face_rows(rows: Iterable[FrameManifestRow]) -> List[FrameManifestRow]:
    return [row for row in rows if row.face_found == 1]
