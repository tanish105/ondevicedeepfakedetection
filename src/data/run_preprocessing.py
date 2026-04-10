from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.csv_io import read_rows
from src.common.schemas import FrameManifestRow
from src.data.ffpp_preprocess import (
    PreprocessConfig,
    build_int8_calibration_rows,
    rebuild_aggregate_manifest,
    run_preprocessing,
    split_manifest_output_path,
)
from src.data.manifest_utils import write_default_calibration_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FF++ frame sampling and MTCNN face cropping for one task split."
    )
    parser.add_argument("--task-id", required=True, choices=["mobilenetv2_df_vs_real", "mobilenetv2_nt_vs_real"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--frame-sample-count", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--fail-on-missing-face", action="store_true")
    parser.add_argument("--rebuild-calibration-manifest", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_preprocessing(
        PreprocessConfig(
            task_id=args.task_id,
            split=args.split,
            frame_sample_count=args.frame_sample_count,
            image_size=args.image_size,
            device=args.device,
            overwrite_images=args.overwrite_images,
            max_videos=args.max_videos,
            fail_on_missing_face=args.fail_on_missing_face,
        )
    )

    aggregate_manifest_path = rebuild_aggregate_manifest(args.task_id)
    output = {
        "summary": summary.__dict__,
        "aggregate_manifest_path": aggregate_manifest_path.as_posix(),
    }

    if args.rebuild_calibration_manifest:
        manifest_rows = read_rows(aggregate_manifest_path, FrameManifestRow)
        calibration_rows = build_int8_calibration_rows(manifest_rows, args.task_id)
        calibration_path = write_default_calibration_manifest(calibration_rows)
        output["calibration_manifest_path"] = calibration_path.as_posix()
        output["calibration_rows"] = len(calibration_rows)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
