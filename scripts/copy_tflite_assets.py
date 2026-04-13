"""Copy the 6 exported TFLite models into the Android app's assets folder.

Run this once before opening the project in Android Studio (or after any
re-export in Phase 4).

Usage
-----
  python scripts/copy_tflite_assets.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "artifacts" / "exports" / "tflite"
DST_DIR = PROJECT_ROOT / "android_app" / "app" / "src" / "main" / "assets"

EXPECTED = [
    "mobilenetv2_df_vs_real_dynamic_range.tflite",
    "mobilenetv2_df_vs_real_float16.tflite",
    "mobilenetv2_df_vs_real_int8_static.tflite",
    "mobilenetv2_nt_vs_real_dynamic_range.tflite",
    "mobilenetv2_nt_vs_real_float16.tflite",
    "mobilenetv2_nt_vs_real_int8_static.tflite",
]


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    missing = []

    for name in EXPECTED:
        src = SRC_DIR / name
        if not src.exists():
            missing.append(str(src))
            continue
        dst = DST_DIR / name
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"  Copied {name} ({size_mb:.2f} MB)")

    if missing:
        print("\nERROR: The following TFLite files were not found:")
        for m in missing:
            print(f"  {m}")
        print("Run Phase 4 export first:  python -m src.export.run_export")
        sys.exit(1)

    print(f"\n[OK] All 6 TFLite models copied to {DST_DIR}")
    print("Now open android_app/ in Android Studio and build.")


if __name__ == "__main__":
    main()
