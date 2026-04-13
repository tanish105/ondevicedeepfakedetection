"""Phase 5 helper: push test-frame images and manifest CSVs to the Android device.

What gets pushed
----------------
  artifacts/predictions/*_fp32_test_frames.csv
      -> /sdcard/deepfake_benchmark/manifests/

  data/processed/frames/df_vs_real/test/
      -> /sdcard/deepfake_benchmark/frames/df_vs_real/test/

  data/processed/frames/nt_vs_real/test/
      -> /sdcard/deepfake_benchmark/frames/nt_vs_real/test/

Usage
-----
  python scripts/adb_push_frames.py

The script auto-discovers adb in PATH and a few common Windows locations.
Ensure exactly one device is connected (adb devices).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ── Locate ADB ────────────────────────────────────────────────────────────────
_ADB_CANDIDATES = [
    "adb",  # in PATH
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
    print("ERROR: adb not found. Install Android SDK Platform Tools and add to PATH.")
    print("  Download: https://developer.android.com/tools/releases/platform-tools")
    sys.exit(1)


def adb(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [ADB] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def adb_shell(*args: str) -> subprocess.CompletedProcess:
    return adb("shell", *args)


# ── Main ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADB = _find_adb()

DEVICE_BASE    = "/sdcard/deepfake_benchmark"
MANIFESTS_DIR  = f"{DEVICE_BASE}/manifests"
FRAMES_BASE    = f"{DEVICE_BASE}/frames"


def check_device() -> None:
    result = adb("devices")
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    devices = [l for l in lines[1:] if "\tdevice" in l]
    if not devices:
        print("No device found. Check USB connection and authorize USB debugging on the phone.")
        sys.exit(1)
    print(f"Device connected: {devices[0].split()[0]}")


def push_manifests() -> None:
    print("\n[1/3] Pushing manifest CSVs...")
    adb_shell("mkdir", "-p", MANIFESTS_DIR)

    for csv in (PROJECT_ROOT / "artifacts" / "predictions").glob("*_fp32_test_frames.csv"):
        print(f"  {csv.name}")
        adb("push", str(csv), f"{MANIFESTS_DIR}/{csv.name}")

    print("  Manifests pushed.")


def push_frames(task_base: str) -> None:
    """Push test frames for one task_base (e.g. 'df_vs_real')."""
    src = PROJECT_ROOT / "data" / "processed" / "frames" / task_base / "test"
    dst = f"{FRAMES_BASE}/{task_base}/test"

    if not src.exists():
        print(f"  WARNING: {src} does not exist, skipping.")
        return

    n = sum(1 for _ in src.rglob("*.png"))
    print(f"  Pushing {n} PNGs: {src} -> {dst}")

    # adb push with the directory pushes the entire tree
    adb_shell("mkdir", "-p", dst)
    adb("push", str(src) + "/.", dst)
    print(f"  Done ({task_base}).")


def main() -> None:
    print(f"ADB: {ADB}")
    check_device()
    push_manifests()

    print("\n[2/3] Pushing df_vs_real test frames...")
    push_frames("df_vs_real")

    print("\n[3/3] Pushing nt_vs_real test frames...")
    push_frames("nt_vs_real")

    print("\nAll data pushed. Launch the app on the device and tap 'Start Benchmark'.")


if __name__ == "__main__":
    main()
