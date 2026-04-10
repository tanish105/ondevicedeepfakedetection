from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import List

from src.common.constants import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, TASK_DF_VS_REAL, TASK_NT_VS_REAL
from src.common.csv_io import read_rows
from src.common.schemas import FrameManifestRow
from src.data.ffpp_preprocess import (
    PreprocessConfig,
    aggregate_manifest_output_path,
    build_combined_int8_calibration_rows,
    rebuild_aggregate_manifest,
    run_preprocessing,
)
from src.data.manifest_utils import write_default_calibration_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full Phase 2 preprocessing for DF-vs-real and NT-vs-real across "
            "train/val/test, then build aggregate manifests and the INT8 calibration manifest."
        )
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--frame-sample-count", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--fail-on-missing-face", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--calibration-total-frames-per-task", type=int, default=500)
    parser.add_argument("--calibration-real-frames-per-task", type=int, default=250)
    parser.add_argument("--calibration-fake-frames-per-task", type=int, default=250)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summaries: List[dict] = []

    for task_id in (TASK_DF_VS_REAL, TASK_NT_VS_REAL):
        for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
            summary = run_preprocessing(
                PreprocessConfig(
                    task_id=task_id,
                    split=split,
                    frame_sample_count=args.frame_sample_count,
                    image_size=args.image_size,
                    device=args.device,
                    overwrite_images=args.overwrite_images,
                    max_videos=args.max_videos,
                    fail_on_missing_face=args.fail_on_missing_face,
                )
            )
            summaries.append(asdict(summary))
        rebuild_aggregate_manifest(task_id)

    all_manifest_rows: List[FrameManifestRow] = []
    aggregate_manifests = {}
    for task_id in (TASK_DF_VS_REAL, TASK_NT_VS_REAL):
        aggregate_path = aggregate_manifest_output_path(task_id)
        aggregate_manifests[task_id] = aggregate_path.as_posix()
        all_manifest_rows.extend(read_rows(aggregate_path, FrameManifestRow))

    calibration_rows = build_combined_int8_calibration_rows(
        all_manifest_rows,
        total_frames_per_task=args.calibration_total_frames_per_task,
        real_target_per_task=args.calibration_real_frames_per_task,
        fake_target_per_task=args.calibration_fake_frames_per_task,
    )
    calibration_path = write_default_calibration_manifest(calibration_rows)

    print(
        json.dumps(
            {
                "summaries": summaries,
                "aggregate_manifests": aggregate_manifests,
                "calibration_manifest_path": calibration_path.as_posix(),
                "calibration_rows": len(calibration_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
