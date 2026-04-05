from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.common.schemas import VideoIndexRow

CSV_TO_MANIPULATION = {
    "original.csv": "real",
    "Deepfakes.csv": "df",
    "NeuralTextures.csv": "nt",
}


def _rows_from_csv(csv_path: Path) -> List[VideoIndexRow]:
    csv_name = csv_path.name
    if csv_name not in CSV_TO_MANIPULATION:
        raise ValueError(f"Unsupported FF++ csv file: {csv_name}")

    manipulation_type = CSV_TO_MANIPULATION[csv_name]
    df = pd.read_csv(csv_path)
    rows: List[VideoIndexRow] = []

    for file_path in df["File Path"]:
        normalized = Path(str(file_path).replace("\\", "/"))
        video_id = normalized.stem

        if manipulation_type == "real":
            source_video_id = video_id
        else:
            parts = video_id.split("_")
            if len(parts) != 2:
                raise ValueError(f"Unexpected manipulated FF++ video id: {video_id}")
            source_video_id = parts[1]

        rows.append(
            VideoIndexRow(
                video_id=video_id,
                source_video_id=source_video_id,
                manipulation_type=manipulation_type,
                video_path=normalized.as_posix(),
            )
        )

    return rows


def build_ffpp_video_index(ffpp_csv_dir: Path) -> List[VideoIndexRow]:
    rows: List[VideoIndexRow] = []
    for csv_name in ("original.csv", "Deepfakes.csv", "NeuralTextures.csv"):
        rows.extend(_rows_from_csv(ffpp_csv_dir / csv_name))
    return rows
