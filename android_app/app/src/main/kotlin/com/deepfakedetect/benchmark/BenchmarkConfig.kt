package com.deepfakedetect.benchmark

object BenchmarkConfig {

    // ── On-device paths ───────────────────────────────────────────────────────
    // All data lives under /sdcard/deepfake_benchmark/
    //   manifests/   — FP32 test-split CSVs pushed from artifacts/predictions/
    //   frames/      — face-crop PNGs pushed from data/processed/frames/
    //   results/     — FramePredictionRow / VideoPredictionRow CSVs written here

    const val DEVICE_BASE      = "/sdcard/deepfake_benchmark"
    const val FRAMES_DIR       = "$DEVICE_BASE/frames"
    const val MANIFESTS_DIR    = "$DEVICE_BASE/manifests"
    const val RESULTS_DIR      = "$DEVICE_BASE/results"

    // ── Task / quant enumerations ─────────────────────────────────────────────
    val TASK_IDS = listOf(
        "mobilenetv2_df_vs_real",
        "mobilenetv2_nt_vs_real",
    )

    // quant_id values must match Python constants.py QUANTIZATION_IDS
    val QUANT_IDS = listOf(
        "dynamic_range_tflite",
        "float16_tflite",
        "int8_static_tflite",
    )

    // ── TFLite asset naming ───────────────────────────────────────────────────
    // Asset file: {taskId}_{suffix}.tflite
    // Suffix is derived from the quant_id by stripping the "_tflite" tail.
    fun modelAssetName(taskId: String, quantId: String): String {
        val suffix = quantId.removeSuffix("_tflite")  // e.g. "dynamic_range"
        return "${taskId}_${suffix}.tflite"
    }

    // ── Image preprocessing ───────────────────────────────────────────────────
    const val IMAGE_SIZE = 224
    val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val IMAGENET_STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

    // ── Inference runtime label written into result CSVs ──────────────────────
    const val RUNTIME = "tflite_android"

    // ── CSV column headers (must match Python FramePredictionRow schema) ──────
    val FRAME_CSV_HEADER = listOf(
        "task_id", "quant_id", "split", "video_id", "frame_idx",
        "image_path", "manipulation_type", "binary_label",
        "score_fake", "pred_label", "latency_ms", "device_name", "runtime",
    )

    val VIDEO_CSV_HEADER = listOf(
        "task_id", "quant_id", "split", "video_id",
        "manipulation_type", "binary_label",
        "num_frames_used", "mean_score_fake", "majority_vote_pred",
        "fps", "device_name", "runtime",
    )
}
