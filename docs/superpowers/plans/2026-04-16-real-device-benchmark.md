# Real-Device Benchmark Re-run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the Phase 5 on-device benchmark on a real Nothing A063 (Snapdragon 888, API 34) device, then regenerate all Phase 6 analysis artifacts with the real-device numbers.

**Architecture:** Seven sequential tasks — backup existing emulator results via git, build and install the APK, set up the device (permissions + storage cleanup), push test frames and manifests, run the benchmark (user taps button, agent monitors logcat), pull and post-process results, then re-run Phase 6 analysis.

**Tech Stack:** Android ADB, Gradle 8.4, TFLite 2.17.0, Python 3, scikit-learn, matplotlib

---

## Project root

All commands run from:
```
/Users/sudhanva/.superset/worktrees/ondevicedeepfakedetection/scientific-salsa
```

App package name: `com.deepfakedetect.benchmark`

---

### Task 1: Backup emulator results via git commit

**Files:**
- No code changes. Only git operations.

- [ ] **Step 1: Stage all emulator artifacts**

```bash
cd /Users/sudhanva/.superset/worktrees/ondevicedeepfakedetection/scientific-salsa
git add artifacts/predictions/ artifacts/analysis/ artifacts/metrics/phase5_summary.json artifacts/metrics/device_specs.json
```

- [ ] **Step 2: Commit as emulator backup**

```bash
git commit -m "$(cat <<'EOF'
chore: backup emulator benchmark results before real-device re-run

Preserves all 14 prediction CSVs, Phase 6 analysis artifacts, and
phase5_summary.json from the emulator run (Pixel 6 AVD, Android 13,
ranchu virtual CPU). Real-device results will overwrite these files.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify commit**

```bash
git log --oneline -3
```

Expected: top commit has message starting with `chore: backup emulator benchmark results`.

---

### Task 2: Build and install the APK

**Files:**
- No code changes. Build + install only.

- [ ] **Step 1: Build the debug APK**

```bash
cd /Users/sudhanva/.superset/worktrees/ondevicedeepfakedetection/scientific-salsa/android_app
./gradlew assembleDebug
```

Expected: `BUILD SUCCESSFUL` at the end. APK written to:
`app/build/outputs/apk/debug/app-debug.apk`

- [ ] **Step 2: Verify device is connected**

```bash
adb devices
```

Expected: one line like `P122B8002280    device` (not `unauthorized`).

- [ ] **Step 3: Install APK on device**

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Expected: `Performing Streamed Install` then `Success`.

---

### Task 3: Set up device storage and permissions

**Files:**
- No code changes. ADB shell commands only.

- [ ] **Step 1: Grant runtime storage permissions**

```bash
adb shell pm grant com.deepfakedetect.benchmark android.permission.READ_EXTERNAL_STORAGE
adb shell pm grant com.deepfakedetect.benchmark android.permission.WRITE_EXTERNAL_STORAGE
```

Expected: No output (silent success).

- [ ] **Step 2: Grant MANAGE_EXTERNAL_STORAGE (required for Android 11+)**

```bash
adb shell appops set com.deepfakedetect.benchmark MANAGE_EXTERNAL_STORAGE allow
```

Expected: No output.

- [ ] **Step 3: Clean any stale benchmark results directory**

This prevents EACCES errors if an old run (from a different APK install UID) left files behind.

```bash
adb shell rm -rf /sdcard/deepfake_benchmark/results
```

Expected: No output (directory may not exist — that is fine).

- [ ] **Step 4: Verify permissions are set**

```bash
adb shell dumpsys package com.deepfakedetect.benchmark | grep -E "MANAGE_EXTERNAL|READ_EXTERNAL|WRITE_EXTERNAL"
```

Expected: lines showing the three permissions are granted.

---

### Task 4: Push test frames and manifests to device

**Files:**
- No code changes. Runs `scripts/adb_push_frames.py`.

- [ ] **Step 1: Run push script**

```bash
cd /Users/sudhanva/.superset/worktrees/ondevicedeepfakedetection/scientific-salsa
python3 scripts/adb_push_frames.py
```

Expected output (in order):
```
ADB: adb
Device connected: P122B8002280

[1/3] Pushing manifest CSVs...
  mobilenetv2_df_vs_real_fp32_test_frames.csv
  mobilenetv2_nt_vs_real_fp32_test_frames.csv
  Manifests pushed.

[2/3] Pushing df_vs_real test frames...
  Pushing 7000 PNGs: ...data/processed/frames/df_vs_real/test -> ...
  Done (df_vs_real).

[3/3] Pushing nt_vs_real test frames...
  Pushing 7000 PNGs: ...data/processed/frames/nt_vs_real/test -> ...
  Done (nt_vs_real).

All data pushed. Launch the app on the device and tap 'Start Benchmark'.
```

This will take several minutes (14,000 PNGs over USB). Do not interrupt.

- [ ] **Step 2: Spot-check that frames landed on device**

```bash
adb shell ls /sdcard/deepfake_benchmark/frames/df_vs_real/test/ | wc -l
adb shell ls /sdcard/deepfake_benchmark/manifests/
```

Expected: `280` (subdirectories, one per video) for the first command. Two CSV files for the second.

---

### Task 5: Run benchmark and wait for completion

**Files:**
- No code changes. App interaction + logcat monitoring.

- [ ] **Step 1: Launch the benchmark app**

```bash
adb shell am start -n com.deepfakedetect.benchmark/.MainActivity
```

Expected: `Starting: Intent { cmp=com.deepfakedetect.benchmark/.MainActivity }`

- [ ] **Step 2: Notify the user to tap START BENCHMARK — this is a required human action**

Print this message for the user, then pause:
```
=====================================================================
ACTION REQUIRED: The benchmark app is now open on the device.
Please TAP the "Start Benchmark" button on the phone screen.
The benchmark will run ~16-20 min (6 models × 7000 frames).
=====================================================================
```

The subagent cannot tap the physical screen. Do not proceed to Step 3 until the user confirms they have tapped the button.

- [ ] **Step 3: Monitor logcat for progress and completion (blocking, ~20 min)**

Run logcat with a long timeout, capturing to a temp file so progress is visible:

```bash
adb logcat -T 1 2>&1 | tee /tmp/benchmark_logcat.txt | grep --line-buffered -E "progress|complete|Complete|ERROR|FATAL|BenchmarkRunner|benchmark" &
LOGCAT_PID=$!
echo "Monitoring logcat (PID $LOGCAT_PID). Waiting for 'All benchmark runs complete'..."
# Poll the log file every 60 seconds
for i in $(seq 1 30); do
    sleep 60
    echo "--- Elapsed: ${i} min ---"
    grep -E "complete|Complete|ERROR|FATAL" /tmp/benchmark_logcat.txt | tail -5
    if grep -q "All benchmark runs complete" /tmp/benchmark_logcat.txt; then
        echo "BENCHMARK COMPLETE."
        kill $LOGCAT_PID 2>/dev/null
        break
    fi
done
```

If after 30 minutes `All benchmark runs complete` has not appeared, check for errors:
```bash
grep -E "ERROR|FATAL|Exception" /tmp/benchmark_logcat.txt | tail -20
```
Stop and report the error before proceeding to Task 6.

If you see `FATAL` or unrecoverable `ERROR`, note the message and stop — do not proceed to Task 6 until resolved.

- [ ] **Step 4: Confirm results directory exists on device**

```bash
adb shell ls /sdcard/deepfake_benchmark/results/
```

Expected: 12 CSV files + 1 `device_specs.json`:
```
mobilenetv2_df_vs_real_dynamic_range_tflite_test_frames.csv
mobilenetv2_df_vs_real_dynamic_range_tflite_test_videos.csv
mobilenetv2_df_vs_real_float16_tflite_test_frames.csv
... (12 CSVs total)
device_specs.json
```

---

### Task 6: Pull results and post-process

**Files:**
- Overwrites: `artifacts/predictions/mobilenetv2_*_tflite_test_frames.csv` (6 files)
- Overwrites: `artifacts/predictions/mobilenetv2_*_tflite_test_videos.csv` (6 files)
- Overwrites: `artifacts/metrics/device_specs.json`
- Overwrites: `artifacts/metrics/phase5_summary.json`

- [ ] **Step 1: Pull raw results from device**

```bash
cd /Users/sudhanva/.superset/worktrees/ondevicedeepfakedetection/scientific-salsa
python3 scripts/adb_pull_results.py
```

Expected: CSVs land in `artifacts/predictions/results/`.

If `adb_pull_results.py` fails on the validate step, that is acceptable — the pull itself is what matters. Verify with:
```bash
ls artifacts/predictions/results/*.csv | wc -l
```
Expected: `12`

- [ ] **Step 2: Post-process results**

```bash
python3 -m src.mobile.collect_phase5_results
```

Expected output:
```
[1/3] Moving result CSVs to artifacts/predictions/ ...
  Done.
[2/3] Copying device_specs.json to artifacts/metrics/ ...
  Done.
[3/3] Computing per-variant latency summary ...
  Written: artifacts/metrics/phase5_summary.json

Phase 5 post-processing complete.
```

- [ ] **Step 3: Spot-check the new latency numbers**

```bash
python3 -c "import json; d=json.load(open('artifacts/metrics/phase5_summary.json')); [print(t,q,v['mean_latency_ms'],'ms') for t,td in d.items() for q,v in td.items()]"
```

Expected: int8_static latency should be ~5–15 ms (vs 87–89 ms on emulator), confirming hardware acceleration is active.

- [ ] **Step 4: Commit raw real-device results**

```bash
cd /Users/sudhanva/.superset/worktrees/ondevicedeepfakedetection/scientific-salsa
git add artifacts/predictions/ artifacts/metrics/
git commit -m "$(cat <<'EOF'
feat(benchmark): real-device Phase 5 results on Nothing A063

Replaces emulator results with benchmark run on Nothing A063
(Snapdragon 888, API 34). All 6 TFLite variants × 7000 frames each.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Re-run Phase 6 analysis

**Files:**
- Overwrites: `artifacts/analysis/experiment_1.json`
- Overwrites: `artifacts/analysis/experiment_2.json`
- Overwrites: `artifacts/analysis/experiment_3.json`
- Overwrites: `artifacts/analysis/phase6_summary.json`
- Overwrites: `artifacts/analysis/nt_failure_frames.csv`
- Overwrites: `artifacts/analysis/nt_failure_grid.png`

- [ ] **Step 1: Run Phase 6 orchestrator**

```bash
cd /Users/sudhanva/.superset/worktrees/ondevicedeepfakedetection/scientific-salsa
python3 -m src.analysis.run_phase6
```

Expected: Prints a summary table to stdout like:

```
task_id                     quant_id              auc      accuracy   fps   quant_gap
mobilenetv2_df_vs_real      fp32_gpu             0.9991   0.9887     -      0.0000
mobilenetv2_df_vs_real      int8_static_tflite   0.9987   0.9880     XX.X   0.0004
...
```

`fps` values should now reflect real-device speeds (~70–200 fps for int8_static on Snapdragon 888).

AUC and accuracy values should be **identical** to the emulator run — same model weights, same images, only latency changes.

- [ ] **Step 2: Verify output files exist**

```bash
ls -lh artifacts/analysis/
```

Expected: 6 files — `experiment_1.json`, `experiment_2.json`, `experiment_3.json`, `phase6_summary.json`, `nt_failure_frames.csv`, `nt_failure_grid.png`.

- [ ] **Step 3: Run the analysis test suite to confirm nothing broke**

```bash
python3 -m pytest tests/analysis/ -v
```

Expected: `24 passed` (or more if tests were added).

- [ ] **Step 4: Commit final analysis artifacts**

```bash
git add artifacts/analysis/ docs/phase6_implementation_log.md
git commit -m "$(cat <<'EOF'
feat(analysis): regenerate Phase 6 results with real-device latency

All experiment tables, failure grid, and summary JSON updated with
Nothing A063 (Snapdragon 888) benchmark results. AUC/accuracy unchanged;
latency and FPS reflect hardware-accelerated inference.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
