from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.request import urlopen

from src.common.constants import PROJECT_ROOT
from src.common.csv_io import write_rows
from src.common.schemas import VideoIndexRow
from src.data.ffpp_index import build_ffpp_video_index
from src.data.split_utils import generate_and_write_task_splits

OFFICIAL_SPLIT_BASE_URL = (
    "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits"
)
SPLIT_NAMES = ("train", "val", "test")


def _load_official_pairs(split_name: str) -> List[Tuple[str, str]]:
    with urlopen(f"{OFFICIAL_SPLIT_BASE_URL}/{split_name}.json") as response:
        data = json.load(response)
    return [(pair[0], pair[1]) for pair in data]


def _materialize_split_ids(pairs: Sequence[Tuple[str, str]]) -> List[str]:
    ids: List[str] = []
    for left_id, right_id in pairs:
        ids.extend([left_id, right_id, f"{left_id}_{right_id}", f"{right_id}_{left_id}"])
    return ids


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_id_list(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def prepare_official_ffpp_splits(
    *,
    ffpp_csv_dir: Path,
    processed_dir: Path,
) -> Dict[str, Dict[str, int]]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    video_index_path = processed_dir / "video_index.csv"
    splits_dir = processed_dir / "splits"
    official_dir = splits_dir / "official"

    video_index_rows = build_ffpp_video_index(ffpp_csv_dir)
    write_rows(video_index_path, video_index_rows, VideoIndexRow)

    split_id_files: Dict[str, Path] = {}
    pair_counts: Dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        pairs = _load_official_pairs(split_name)
        pair_counts[split_name] = len(pairs)
        ids = _materialize_split_ids(pairs)
        _write_json(official_dir / f"{split_name}.json", pairs)
        _write_id_list(official_dir / f"{split_name}_ids.txt", ids)
        split_id_files[split_name] = official_dir / f"{split_name}_ids.txt"

    outputs = generate_and_write_task_splits(
        video_index_csv=video_index_path,
        train_ids_path=split_id_files["train"],
        val_ids_path=split_id_files["val"],
        test_ids_path=split_id_files["test"],
        output_dir=splits_dir,
    )

    summary: Dict[str, Dict[str, int]] = {
        "official_pair_counts": pair_counts,
        "video_index_counts": {
            "rows": len(video_index_rows),
        },
    }
    for task_id, split_rows in outputs.items():
        summary[task_id] = {split_name: len(rows) for split_name, rows in split_rows.items()}
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build FF++ video_index.csv and official df-vs-real / nt-vs-real split CSVs."
    )
    parser.add_argument(
        "--ffpp-csv-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "ffpp_c23" / "csv",
        help="Directory containing FF++ csv files such as original.csv and Deepfakes.csv",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Processed-data root where video_index.csv and split files will be written",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = prepare_official_ffpp_splits(
        ffpp_csv_dir=args.ffpp_csv_dir,
        processed_dir=args.processed_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
