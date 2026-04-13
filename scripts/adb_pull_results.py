"""Phase 5 helper: pull benchmark result CSVs from the Android device.

Pulls everything under /sdcard/deepfake_benchmark/results/ into
artifacts/predictions/, then validates row counts.

Usage
-----
  python scripts/adb_pull_results.py

Run this after the benchmark app has finished (the app logs 'All benchmark
runs complete' on screen).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ── Locate ADB (same logic as adb_push_frames.py) ───────────────────────────
_ADB_CANDIDATES = [
    "adb",
    r"C:\Program Files\ASUS\GlideX\adb.exe",
    r"C:\Users\%USERNAME%\AppData\Local\Android\Sdk\platform-tools\adb.exe",
    r"C:\Android\platform-tools\adb.exe",
]


def _find_adb() -> str:
    import os
    import shutil
    for candidate in _ADB_CANDIDATES:
        expanded = os.path.expandvars(candidate)
        if shutil.which(expanded) or Path(expanded).exists():
            return expanded
    print("ERROR: adb not found.")
    sys.exit(1)


def adb(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [ADB] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


# ── Main ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADB = _find_adb()

DEVICE_RESULTS = "/sdcard/deepfake_benchmark/results"
LOCAL_DEST     = PROJECT_ROOT / "artifacts" / "predictions"


def check_device() -> None:
    result = adb("devices")
    lines  = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    devices = [l for l in lines[1:] if "\tdevice" in l]
    if not devices:
        print("No device found.")
        sys.exit(1)
    print(f"Device: {devices[0].split()[0]}")


def pull_results() -> None:
    LOCAL_DEST.mkdir(parents=True, exist_ok=True)
    print(f"\nPulling {DEVICE_RESULTS} -> {LOCAL_DEST}")
    result = adb("pull", DEVICE_RESULTS, str(LOCAL_DEST), check=False)
    if result.returncode != 0:
        print(f"ERROR pulling results:\n{result.stderr}")
        sys.exit(1)
    print("Pull complete.")


def validate() -> None:
    print("\nValidating pulled CSVs:")
    from src.common.constants import TASK_IDS, QUANT_DYNAMIC_RANGE, QUANT_FLOAT16, QUANT_INT8_STATIC
    quant_ids = [QUANT_DYNAMIC_RANGE, QUANT_FLOAT16, QUANT_INT8_STATIC]

    all_ok = True
    for task_id in TASK_IDS:
        for quant_id in quant_ids:
            # frame CSV
            frame_csv = LOCAL_DEST / f"results/{task_id}_{quant_id}_test_frames.csv"
            if not frame_csv.exists():
                print(f"  MISSING: {frame_csv.name}")
                all_ok = False
                continue
            n = sum(1 for _ in frame_csv.open()) - 1  # subtract header
            print(f"  {frame_csv.name}: {n} rows")
            # video CSV
            video_csv = LOCAL_DEST / f"results/{task_id}_{quant_id}_test_videos.csv"
            if video_csv.exists():
                nv = sum(1 for _ in video_csv.open()) - 1
                print(f"  {video_csv.name}: {nv} rows")
            else:
                print(f"  MISSING: {video_csv.name}")
                all_ok = False

    if all_ok:
        print("\n[OK] All 12 result CSVs present. Ready for Phase 6 analysis.")
    else:
        print("\n[WARN] Some CSVs are missing. Check the benchmark app log.")


def main() -> None:
    print(f"ADB: {ADB}")
    check_device()
    pull_results()
    validate()


if __name__ == "__main__":
    main()
