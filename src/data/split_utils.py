from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.common.constants import (
    DEFAULT_SPLITS_DIR,
    MANIPULATION_REAL,
    SPLIT_NAMES,
    TASK_DF_VS_REAL,
    TASK_NT_VS_REAL,
)
from src.common.csv_io import read_rows, write_rows
from src.common.schemas import SplitRow, VideoIndexRow

TASK_OUTPUT_PREFIX = {
    TASK_DF_VS_REAL: "df_vs_real",
    TASK_NT_VS_REAL: "nt_vs_real",
}

TASK_TO_FAKE_CLASS = {
    TASK_DF_VS_REAL: "df",
    TASK_NT_VS_REAL: "nt",
}


def read_id_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_video_index(path: Path) -> List[VideoIndexRow]:
    return read_rows(path, VideoIndexRow)


def _validate_split_assignments(assignments: Dict[str, Sequence[str]]) -> None:
    missing = [split_name for split_name in SPLIT_NAMES if split_name not in assignments]
    if missing:
        raise ValueError(f"Missing split assignments for: {missing}")

    seen: Dict[str, str] = {}
    for split_name, video_ids in assignments.items():
        for video_id in video_ids:
            if video_id in seen:
                raise ValueError(
                    f"Video '{video_id}' appears in both '{seen[video_id]}' and '{split_name}'"
                )
            seen[video_id] = split_name


def _index_by_id(rows: Iterable[VideoIndexRow]) -> Dict[str, VideoIndexRow]:
    index: Dict[str, VideoIndexRow] = {}
    for row in rows:
        if row.video_id in index:
            raise ValueError(f"Duplicate video_id in video index: {row.video_id}")
        index[row.video_id] = row
    return index


def build_task_splits(
    video_index_rows: Sequence[VideoIndexRow],
    split_assignments: Dict[str, Sequence[str]],
    task_id: str,
) -> Dict[str, List[SplitRow]]:
    _validate_split_assignments(split_assignments)
    fake_class = TASK_TO_FAKE_CLASS[task_id]
    allowed_classes = {MANIPULATION_REAL, fake_class}
    task_rows = [row for row in video_index_rows if row.manipulation_type in allowed_classes]
    rows_by_id = _index_by_id(task_rows)

    output: Dict[str, List[SplitRow]] = {}
    for split_name, video_ids in split_assignments.items():
        split_rows: List[SplitRow] = []
        for video_id in video_ids:
            if video_id not in rows_by_id:
                raise ValueError(f"Video '{video_id}' not found in video index")
            source = rows_by_id[video_id]
            if source.manipulation_type not in allowed_classes:
                continue
            split_rows.append(
                SplitRow(
                    video_id=source.video_id,
                    source_video_id=source.source_video_id,
                    manipulation_type=source.manipulation_type,
                    binary_label=0 if source.manipulation_type == MANIPULATION_REAL else 1,
                    split=split_name,
                    video_path=source.video_path,
                )
            )
        output[split_name] = split_rows

    _validate_class_balance(output, task_id)
    return output


def _validate_class_balance(task_splits: Dict[str, Sequence[SplitRow]], task_id: str) -> None:
    fake_class = TASK_TO_FAKE_CLASS[task_id]
    for split_name, rows in task_splits.items():
        counts = Counter(row.manipulation_type for row in rows)
        if counts[MANIPULATION_REAL] == 0 or counts[fake_class] == 0:
            raise ValueError(
                f"Split '{split_name}' for task '{task_id}' is missing one class: {counts}"
            )


def write_task_splits(task_splits: Dict[str, Sequence[SplitRow]], task_id: str, output_dir: Path) -> None:
    prefix = TASK_OUTPUT_PREFIX[task_id]
    for split_name, rows in task_splits.items():
        output_path = output_dir / f"{prefix}_{split_name}.csv"
        write_rows(output_path, rows, SplitRow)


def generate_and_write_task_splits(
    *,
    video_index_csv: Path,
    train_ids_path: Path,
    val_ids_path: Path,
    test_ids_path: Path,
    output_dir: Path = DEFAULT_SPLITS_DIR,
) -> Dict[str, Dict[str, List[SplitRow]]]:
    video_index_rows = load_video_index(video_index_csv)
    split_assignments = {
        "train": read_id_file(train_ids_path),
        "val": read_id_file(val_ids_path),
        "test": read_id_file(test_ids_path),
    }
    all_outputs: Dict[str, Dict[str, List[SplitRow]]] = {}
    for task_id in (TASK_DF_VS_REAL, TASK_NT_VS_REAL):
        task_splits = build_task_splits(video_index_rows, split_assignments, task_id)
        write_task_splits(task_splits, task_id, output_dir)
        all_outputs[task_id] = task_splits
    return all_outputs
