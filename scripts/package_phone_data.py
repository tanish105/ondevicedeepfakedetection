"""Package the test frames and manifest CSVs into a zip for sharing.

The zip can be sent to a collaborator who then extracts it and runs
adb_push_device.py (or manually pushes via adb) to put the files on
the phone.

Output: artifacts/phone_data.zip  (~250-350 MB depending on compression)

Usage
-----
  python scripts/package_phone_data.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_GLOB = "artifacts/predictions/*_fp32_test_frames.csv"
FRAMES_DIRS = [
    "data/processed/frames/df_vs_real/test",
    "data/processed/frames/nt_vs_real/test",
]
OUT_ZIP = PROJECT_ROOT / "artifacts" / "phone_data.zip"


def main() -> None:
    manifests = list(PROJECT_ROOT.glob(MANIFEST_GLOB))
    if not manifests:
        print("ERROR: No manifest CSVs found. Run Phase 3 first.")
        return

    print(f"Writing {OUT_ZIP} ...")
    total = 0

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Manifest CSVs
        for csv in manifests:
            arcname = f"manifests/{csv.name}"
            zf.write(csv, arcname)
            print(f"  + {arcname}")
            total += 1

        # Test frames
        for frames_rel in FRAMES_DIRS:
            frames_dir = PROJECT_ROOT / frames_rel
            if not frames_dir.exists():
                print(f"  WARNING: {frames_dir} not found, skipping.")
                continue
            task_base = frames_dir.parent.parent.name   # e.g. "df_vs_real"
            for png in sorted(frames_dir.rglob("*.png")):
                rel = png.relative_to(frames_dir.parent.parent)
                arcname = f"frames/{task_base}/{rel}"
                zf.write(png, arcname)
                total += 1
                if total % 1000 == 0:
                    print(f"  ... {total} files added")

    size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n[OK] {total} files -> {OUT_ZIP} ({size_mb:.0f} MB)")
    print("\nShare this zip with your collaborator.")
    print("They extract it and run:  python scripts/adb_push_device.py <path/to/extracted>")


if __name__ == "__main__":
    main()
