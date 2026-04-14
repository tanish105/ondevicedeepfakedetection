# Phase 5 Handoff: On-Device Benchmark Results

> **For incoming agents:** Read this document before starting Phase 6 (analysis)
> or before re-running the benchmark on a physical device.
> All code is on branch `Sudhanva-M/deployment`. Latest commit: `3b4cd41`.

---

## What This Project Is

On-device deepfake detection research using MobileNetV2. The pipeline:

1. FF++ (FaceForensics++) dataset, compression level `c23`
2. Two binary classification tasks:
   - `mobilenetv2_df_vs_real` — DeepFakes vs real
   - `mobilenetv2_nt_vs_real` — NeuralTextures vs real
3. Models trained in PyTorch (FP32), exported through ONNX → TFLite in three
   quantization variants each: `dynamic_range`, `float16`, `int8_static`
4. All six TFLite models benchmarked on Android for latency + prediction accuracy

Phase 6 (not yet started) compares quantized accuracy against the FP32 baseline.

---

## Repository Layout (relevant paths)

```
android_app/                     Kotlin benchmark app (Gradle 8.4, TFLite 2.17.0)
  app/src/main/assets/           6 × .tflite model files (bundled into APK)
  app/src/main/kotlin/…/         BenchmarkRunner.kt, DeviceInfo.kt, MainActivity.kt
  gradlew                        CLI build wrapper (no Android Studio needed)

scripts/
  adb_push_frames.py             Push test frames + manifests to device
  adb_pull_results.py            Pull result CSVs from device (optional validator)

src/
  common/constants.py            TASK_IDS, QUANT_* constants, shared schemas
  mobile/collect_phase5_results.py  Post-process pulled CSVs → artifacts/

artifacts/
  predictions/                   All FramePredictionRow + VideoPredictionRow CSVs
  metrics/
    device_specs.json            Hardware info from the benchmark run
    phase5_summary.json          Per-variant mean latency and FPS
```

---

## What Was Done in Phase 5

### Infrastructure built from scratch

| Component | Detail |
|---|---|
| `android_app/gradlew` | Gradle 8.4 wrapper — full CLI build, no Android Studio |
| `DeviceInfo.kt` | Writes `device_specs.json` at end of each run |
| `BenchmarkRunner.kt` | Updated to call DeviceInfo and report progress every 500 frames |
| `collect_phase5_results.py` | Moves CSVs out of `results/` subdir, copies device specs, computes JSON summary |
| `conftest.py` (root) | Torch stub for pytest on machines without PyTorch |

### Build issues resolved

| Problem | Fix |
|---|---|
| FULLY_CONNECTED op version 12 not supported | Upgraded TFLite runtime from 2.14.0 → 2.17.0 |
| TFLite 2.17.0 duplicate class conflict (`litert-api` vs `tensorflow-lite-api`) | Added `exclude(group="org.tensorflow", module="tensorflow-lite-api")` in `build.gradle.kts` |
| `android:icon="@mipmap/ic_launcher"` build fail | Removed from AndroidManifest (mipmap resources not committed) |
| Missing `gradle.properties` | Added `android.useAndroidX=true`, `enableJetifier=true` |

### Emulator-specific workarounds

- `python3` not `python` — system default was Python 2.7
- UTF-8 encoding declarations in ADB scripts (box-drawing chars in comments)
- `data/processed/frames` symlink → actual frame location at `data/phone_data/frames/frames/`
- Storage permissions: `pm grant ... READ/WRITE_EXTERNAL_STORAGE` + `appops set ... MANAGE_EXTERNAL_STORAGE allow` — must be run **after** install and **before** launch
- Emulator boot: wait for `init.svc.bootanim` to stop + `service check package` to return found before installing APK

---

## Phase 5 Results

### Device

| Field | Value |
|---|---|
| Device | Google sdk_gphone64_arm64 (Android emulator) |
| Android version | 16 (API 36) |
| Hardware | ranchu (virtual CPU, no NNAPI/GPU) |
| RAM | 1974 MB |
| TFLite version | 2.17.0 |

### Latency summary (`artifacts/metrics/phase5_summary.json`)

| Task | Quantization | Mean latency (ms) | FPS | Frames |
|---|---|---|---|---|
| df_vs_real | dynamic_range | 838.7 | 1.2 | 6727 |
| df_vs_real | float16 | 168.9 | 5.9 | 7000 |
| df_vs_real | int8_static | 86.5 | 11.6 | 7000 |
| nt_vs_real | dynamic_range | 683.6 | 1.5 | 7000 |
| nt_vs_real | float16 | 190.1 | 5.3 | 7000 |
| nt_vs_real | int8_static | 88.7 | 11.3 | 7000 |

**Note on df_vs_real/dynamic_range (6727 frames, not 7000):** 273 image files
were absent from the device during that model's run (data transfer artifact from
the first run attempt). All other 5 models processed the full 7000 frames. For
Phase 6 accuracy comparisons on this variant, either exclude those 273 samples
from all conditions or note the sample size discrepancy.

### Prediction CSVs (`artifacts/predictions/`)

12 CSVs total — one frame-level and one video-level per model variant:

```
mobilenetv2_{task}_{quant_id}_test_frames.csv   # FramePredictionRow schema
mobilenetv2_{task}_{quant_id}_test_videos.csv   # VideoPredictionRow schema
```

Frame CSV schema (confirmed):
```
task_id, quant_id, split, video_id, frame_idx, image_path,
manipulation_type, binary_label, score_fake, pred_label,
latency_ms, device_name, runtime
```

`runtime` value is `tflite_android` for all Phase 5 CSVs.
FP32 baseline CSVs (`*_fp32_test_frames.csv`) also present — `runtime=pytorch_fp32`.

---

## What Phase 6 Needs To Do

From `docs/workflow.md`:

1. Compute frame-level AUC and Binary Accuracy (threshold 0.5) per condition
2. Compute per-class AUC for DF and NT separately
3. Compute quantization gap: `AUC(FP32-GPU) − AUC(quantized)` for each variant
4. **Experiment 1 table:** FP32-GPU vs INT8-Mobile — AUC, Accuracy, FPS, model size, gap
5. **Experiment 2 table:** per-class DF vs NT AUC under FP32-GPU and INT8-Mobile; confirm gap is larger for NT
6. **Experiment 3 table:** all three quant methods — AUC (overall + per-class), Accuracy, FPS, model size, gap
7. Mine NT failure frames: find frames where FP32-GPU correct but INT8-Mobile wrong
8. Qualitative failure grid: 4–6 NT failure frames (image + FP32 score + INT8 score)
9. Video-level accuracy and majority-vote FPS from VideoPredictionRow CSVs

All inputs are ready in `artifacts/predictions/`. No additional data collection needed.

---

## Re-Running on a Physical Android Device

If you want real-device numbers (expected ~10–15 min total vs ~5 hours on emulator):

### Expected speeds on a mid-range physical device

| Quantization | Emulator (this run) | Estimated real device |
|---|---|---|
| dynamic_range | ~760 ms/frame | ~20–40 ms/frame |
| float16 | ~180 ms/frame | ~15–25 ms/frame |
| int8_static | ~88 ms/frame | ~5–15 ms/frame |

Accuracy results would be **identical** — same TFLite files, same weights.
The prediction `score_fake` values will be the same; only `latency_ms` and
`device_name` fields change.

### Steps

```bash
# 1. Build APK (already works, no changes needed)
cd android_app
./gradlew assembleDebug

# 2. Connect physical device with USB debugging enabled
adb devices   # should show device, not emulator

# 3. Install
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 4. Grant storage permissions
adb shell pm grant com.deepfakedetect.benchmark android.permission.READ_EXTERNAL_STORAGE
adb shell pm grant com.deepfakedetect.benchmark android.permission.WRITE_EXTERNAL_STORAGE
adb shell appops set com.deepfakedetect.benchmark MANAGE_EXTERNAL_STORAGE allow

# 5. Push frames and manifests
python3 scripts/adb_push_frames.py

# 6. Launch app and tap START BENCHMARK
adb shell am start -n com.deepfakedetect.benchmark/.MainActivity
# (tap the button — or wire up an automated tap if needed)

# 7. Wait for completion (watch logcat)
adb logcat | grep -E "All benchmark runs complete|FATAL"

# 8. Pull results
adb pull /sdcard/deepfake_benchmark/results/ artifacts/predictions/

# 9. Post-process
python3 -m src.mobile.collect_phase5_results
```

### Key gotchas to avoid

- **Storage permissions must be granted after install, before launch** — appops alone is insufficient; `pm grant` the runtime permissions too
- **If results directory exists from a previous run owned by a different app UID**, the new install will get EACCES. Fix: `adb shell rm -rf /sdcard/deepfake_benchmark/results` before launching, or uninstall/reinstall the app
- **`python` = Python 2 on some machines** — always use `python3`
- **`adb_pull_results.py` validate() will fail** if run from a directory not on PYTHONPATH — the pull itself succeeds, validation is optional
- Physical device will not have the 273-frame skip issue — that was emulator-specific

### To update `device_specs.json` after a physical run

`DeviceInfo.kt` auto-populates device name, manufacturer, model, hardware (SoC),
Android version, API level, RAM, and TFLite version. No code changes needed.

---

## Key Technical Notes for Future Reference

**Why dynamic_range is so much slower than the other two:**
Dynamic range quantization compresses weights to INT8 for storage but
*dequantizes them back to float32 before every matmul*. You get file size savings
but no inference speed benefit. float16 and int8_static run natively in their
respective numeric formats.

**Why the emulator numbers can't be used for the paper's latency claims:**
The emulator runs all TFLite on a software CPU (hardware = `ranchu`) with no
NNAPI, GPU, or DSP delegate. Real device int8_static would be ~5–15 ms, not
88 ms. The paper should either re-run on a physical device for latency numbers,
or clearly note these are software-emulated figures.

**The `score_fake` field:**
The sigmoid is baked into the TFLite models during export. The raw model output
IS the probability — no post-processing needed. Values close to 1.0 = predicted
fake, close to 0.0 = predicted real.

**TFLite version constraint:**
The exported models use FULLY_CONNECTED op version 12, which requires TFLite
runtime ≥ 2.17.0. Do not downgrade the dependency in `libs.versions.toml`.
