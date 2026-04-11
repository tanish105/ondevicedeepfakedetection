from __future__ import annotations

import argparse
from pathlib import Path

from src.common.constants import DEFAULT_SPLITS_DIR, TASK_DF_VS_REAL, TASK_NT_VS_REAL
from src.data.split_utils import generate_and_write_task_splits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate df-vs-real and nt-vs-real split CSVs from a video index and "
            "official FF++ train/val/test ID lists."
        )
    )
    parser.add_argument(
        "--video-index-csv",
        type=Path,
        required=True,
        help="CSV with columns: video_id, source_video_id, manipulation_type, video_path",
    )
    parser.add_argument("--train-ids", type=Path, required=True, help="Text file with one video_id per line")
    parser.add_argument("--val-ids", type=Path, required=True, help="Text file with one video_id per line")
    parser.add_argument("--test-ids", type=Path, required=True, help="Text file with one video_id per line")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help=f"Output directory for split CSVs. Defaults to {DEFAULT_SPLITS_DIR.as_posix()}",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    outputs = generate_and_write_task_splits(
        video_index_csv=args.video_index_csv,
        train_ids_path=args.train_ids,
        val_ids_path=args.val_ids,
        test_ids_path=args.test_ids,
        output_dir=args.output_dir,
    )

    df_counts = {split: len(rows) for split, rows in outputs[TASK_DF_VS_REAL].items()}
    nt_counts = {split: len(rows) for split, rows in outputs[TASK_NT_VS_REAL].items()}

    print("Generated split CSVs:")
    print(f"  {TASK_DF_VS_REAL}: {df_counts}")
    print(f"  {TASK_NT_VS_REAL}: {nt_counts}")


if __name__ == "__main__":
    main()
