"""Push extracted phone_data/ folder onto the connected Android device.

Usage (collaborator's machine)
------------------------------
  1. Extract phone_data.zip anywhere, e.g. C:\\Downloads\\phone_data\\
  2. Connect Android phone via USB (USB debugging on, authorize the prompt)
  3. python scripts/adb_push_device.py C:\\Downloads\\phone_data

The script creates the required directory tree on the device:
  /sdcard/deepfake_benchmark/manifests/   <- manifest CSVs
  /sdcard/deepfake_benchmark/frames/      <- test-frame PNGs
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ── ADB discovery ─────────────────────────────────────────────────────────────
_ADB_CANDIDATES = [
    "adb",
    r"C:\Program Files\ASUS\GlideX\adb.exe",
    r"C:\Users\%USERNAME%\AppData\Local\Android\Sdk\platform-tools\adb.exe",
    r"C:\Android\platform-tools\adb.exe",
    "/usr/bin/adb",
    "/usr/local/bin/adb",
]


def _find_adb() -> str:
    import os, shutil
    for c in _ADB_CANDIDATES:
        exp = os.path.expandvars(c)
        if shutil.which(exp) or Path(exp).exists():
            return exp
    print("ERROR: adb not found.")
    print("Install Android SDK Platform Tools: https://developer.android.com/tools/releases/platform-tools")
    sys.exit(1)


def adb(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([ADB] + list(args), capture_output=True, text=True, check=True)


def check_device() -> None:
    lines = [l.strip() for l in adb("devices").stdout.splitlines()]
    devices = [l for l in lines[1:] if "\tdevice" in l]
    if not devices:
        print("No device found. Check USB + USB debugging authorization.")
        sys.exit(1)
    print(f"Device: {devices[0].split()[0]}")


def push_dir(src: Path, device_dst: str) -> None:
    """Push an entire local directory tree to the device."""
    adb("shell", "mkdir", "-p", device_dst)
    result = subprocess.run(
        [ADB, "push", str(src) + "/.", device_dst],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
    else:
        # adb push prints one summary line, e.g. "14000 files pushed."
        summary = [l for l in result.stdout.splitlines() if l.strip()]
        if summary:
            print(f"  {summary[-1]}")


# ── Main ──────────────────────────────────────────────────────────────────────

ADB = _find_adb()
DEVICE_BASE = "/sdcard/deepfake_benchmark"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python adb_push_device.py <path/to/extracted/phone_data>")
        sys.exit(1)

    data_root = Path(sys.argv[1]).resolve()
    if not data_root.exists():
        print(f"ERROR: {data_root} does not exist.")
        sys.exit(1)

    print(f"ADB : {ADB}")
    check_device()

    # Push manifests
    manifests_src = data_root / "manifests"
    if manifests_src.exists():
        print("\nPushing manifests...")
        push_dir(manifests_src, f"{DEVICE_BASE}/manifests")
    else:
        print(f"WARNING: {manifests_src} not found, skipping manifests.")

    # Push frames (df_vs_real and nt_vs_real)
    frames_src = data_root / "frames"
    if frames_src.exists():
        for task_dir in sorted(frames_src.iterdir()):
            if task_dir.is_dir():
                n = sum(1 for _ in task_dir.rglob("*.png"))
                print(f"\nPushing {task_dir.name} ({n} frames)...")
                push_dir(task_dir, f"{DEVICE_BASE}/frames/{task_dir.name}")
    else:
        print(f"WARNING: {frames_src} not found, skipping frames.")

    print("\nDone. Launch the Deepfake Benchmark app and tap 'Start Benchmark'.")
    print(f"Pull results when done:  adb pull {DEVICE_BASE}/results/ .")


if __name__ == "__main__":
    main()
