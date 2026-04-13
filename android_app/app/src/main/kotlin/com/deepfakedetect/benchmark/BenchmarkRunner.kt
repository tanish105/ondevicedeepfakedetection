package com.deepfakedetect.benchmark

import android.content.Context
import android.os.Build
import java.io.File

/**
 * Orchestrates the full Phase 5 benchmarking pipeline.
 *
 * For each (task_id, quant_id) pair it:
 *   1. Reads the FP32 manifest CSV from MANIFESTS_DIR.
 *   2. Loads the corresponding TFLite model from assets.
 *   3. Runs inference on every test frame, measuring wall-clock latency.
 *   4. Writes a FramePredictionRow CSV to RESULTS_DIR.
 *   5. Aggregates to video-level (majority vote) and writes a VideoPredictionRow CSV.
 *
 * Progress messages are delivered via [onProgress] (called on a background thread).
 */
class BenchmarkRunner(
    private val context: Context,
    private val onProgress: (String) -> Unit,
) {

    private val deviceName: String =
        "${Build.MANUFACTURER} ${Build.MODEL}".trim()

    fun runAll() {
        val resultsDir = File(BenchmarkConfig.RESULTS_DIR)
        resultsDir.mkdirs()

        for (taskId in BenchmarkConfig.TASK_IDS) {
            val manifestPath =
                "${BenchmarkConfig.MANIFESTS_DIR}/${taskId}_fp32_test_frames.csv"

            onProgress("[$taskId] Reading manifest...")
            val rows: List<ManifestRow> = try {
                readManifestCsv(manifestPath)
            } catch (e: Exception) {
                onProgress("[$taskId] ERROR reading manifest: ${e.message}")
                continue
            }
            onProgress("[$taskId] ${rows.size} frames loaded from manifest.")

            for (quantId in BenchmarkConfig.QUANT_IDS) {
                runSingleVariant(taskId, quantId, rows, resultsDir)
            }
        }

        // Write device specs alongside result CSVs
        try {
            DeviceInfo.write(context, resultsDir)
            onProgress("\nDevice specs written: ${BenchmarkConfig.RESULTS_DIR}/device_specs.json")
        } catch (e: Exception) {
            onProgress("WARNING: Could not write device_specs.json: ${e.message}")
        }

        onProgress("\nAll benchmark runs complete.")
        onProgress("Results in: ${BenchmarkConfig.RESULTS_DIR}")
    }

    private fun runSingleVariant(
        taskId: String,
        quantId: String,
        manifestRows: List<ManifestRow>,
        resultsDir: File,
    ) {
        val assetName = BenchmarkConfig.modelAssetName(taskId, quantId)
        onProgress("\n[$taskId / $quantId] Loading model: $assetName")

        val engine = try {
            TFLiteEngine(context, assetName)
        } catch (e: Exception) {
            onProgress("[$taskId / $quantId] ERROR loading model: ${e.message}")
            return
        }

        val frameResults = mutableListOf<FrameResultRow>()
        var skipped = 0
        val total = manifestRows.size

        engine.use {
            for ((index, row) in manifestRows.withIndex()) {
                // image_path in manifest: "data/processed/frames/df_vs_real/test/953/0000.png"
                // strip "data/processed/frames/" prefix → "df_vs_real/test/953/0000.png"
                val relPath = row.imagePath.removePrefix("data/processed/frames/")
                val fullPath = "${BenchmarkConfig.FRAMES_DIR}/$relPath"

                val inputFloats = ImagePreprocessor.preprocess(fullPath)
                if (inputFloats == null) {
                    skipped++
                    continue
                }

                // Measure wall-clock inference latency
                val t0 = System.nanoTime()
                val scoreFake = engine.infer(inputFloats)
                val latencyMs = (System.nanoTime() - t0) / 1_000_000f

                frameResults += FrameResultRow(
                    taskId           = row.taskId,
                    quantId          = quantId,
                    split            = row.split,
                    videoId          = row.videoId,
                    frameIdx         = row.frameIdx,
                    imagePath        = row.imagePath,
                    manipulationType = row.manipulationType,
                    binaryLabel      = row.binaryLabel,
                    scoreFake        = scoreFake,
                    predLabel        = if (scoreFake >= 0.5f) 1 else 0,
                    latencyMs        = latencyMs,
                    deviceName       = deviceName,
                    runtime          = BenchmarkConfig.RUNTIME,
                )

                if ((index + 1) % 500 == 0 || index + 1 == total) {
                    val mean = frameResults.takeLast(500).map { it.latencyMs }.average()
                    onProgress(
                        "  ${index + 1}/$total frames | mean_lat=${
                            "%.1f".format(mean)
                        } ms | skipped=$skipped"
                    )
                }
            }
        }

        if (skipped > 0) {
            onProgress("[$taskId / $quantId] WARNING: $skipped frames skipped (image not found).")
        }

        // Write frame-level CSV
        val frameCsvName = "${taskId}_${quantId}_test_frames.csv"
        val frameCsvPath = "${resultsDir.path}/$frameCsvName"
        writeFrameCsv(frameResults, frameCsvPath)
        onProgress("[$taskId / $quantId] Frame CSV -> $frameCsvPath (${frameResults.size} rows)")

        // Video-level aggregation
        val videoResults = aggregateToVideo(frameResults)
        val videoCsvName = "${taskId}_${quantId}_test_videos.csv"
        val videoCsvPath = "${resultsDir.path}/$videoCsvName"
        writeVideoCsv(videoResults, videoCsvPath)

        val meanFps = videoResults.map { it.fps }.average()
        onProgress(
            "[$taskId / $quantId] Video CSV -> $videoCsvPath " +
                "(${videoResults.size} videos | mean_fps=${"%.1f".format(meanFps)})"
        )
    }
}
