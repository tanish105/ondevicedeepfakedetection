package com.deepfakedetect.benchmark

import java.io.File
import java.io.PrintWriter

// ── Data classes (mirror Python FramePredictionRow / VideoPredictionRow) ──────

data class ManifestRow(
    val taskId: String,
    val videoId: String,
    val frameIdx: Int,
    val imagePath: String,      // relative: "data/processed/frames/df_vs_real/test/…"
    val manipulationType: String,
    val binaryLabel: Int,
    val split: String,
)

data class FrameResultRow(
    val taskId: String,
    val quantId: String,
    val split: String,
    val videoId: String,
    val frameIdx: Int,
    val imagePath: String,
    val manipulationType: String,
    val binaryLabel: Int,
    val scoreFake: Float,
    val predLabel: Int,
    val latencyMs: Float,
    val deviceName: String,
    val runtime: String,
)

data class VideoResultRow(
    val taskId: String,
    val quantId: String,
    val split: String,
    val videoId: String,
    val manipulationType: String,
    val binaryLabel: Int,
    val numFramesUsed: Int,
    val meanScoreFake: Float,
    val majorityVotePred: Int,
    val fps: Float,
    val deviceName: String,
    val runtime: String,
)

// ── CSV reader ────────────────────────────────────────────────────────────────

/**
 * Read the FP32 manifest CSV written by Phase 3 and return only the columns
 * needed for benchmarking (skip score_fake / pred_label / latency_ms).
 */
fun readManifestCsv(path: String): List<ManifestRow> {
    val file = File(path)
    if (!file.exists()) error("Manifest not found: $path")

    val lines = file.readLines()
    val header = lines.first().split(",")
    val idx = header::indexOf

    return lines.drop(1).mapNotNull { line ->
        if (line.isBlank()) return@mapNotNull null
        val cols = line.split(",")
        ManifestRow(
            taskId           = cols[idx("task_id")],
            videoId          = cols[idx("video_id")],
            frameIdx         = cols[idx("frame_idx")].toInt(),
            imagePath        = cols[idx("image_path")],
            manipulationType = cols[idx("manipulation_type")],
            binaryLabel      = cols[idx("binary_label")].toInt(),
            split            = cols[idx("split")],
        )
    }
}

// ── CSV writers ───────────────────────────────────────────────────────────────

fun writeFrameCsv(rows: List<FrameResultRow>, outputPath: String) {
    File(outputPath).parentFile?.mkdirs()
    PrintWriter(outputPath).use { pw ->
        pw.println(BenchmarkConfig.FRAME_CSV_HEADER.joinToString(","))
        for (r in rows) {
            pw.println(
                listOf(
                    r.taskId, r.quantId, r.split, r.videoId, r.frameIdx,
                    r.imagePath, r.manipulationType, r.binaryLabel,
                    r.scoreFake, r.predLabel, r.latencyMs,
                    r.deviceName, r.runtime,
                ).joinToString(",")
            )
        }
    }
}

fun writeVideoCsv(rows: List<VideoResultRow>, outputPath: String) {
    File(outputPath).parentFile?.mkdirs()
    PrintWriter(outputPath).use { pw ->
        pw.println(BenchmarkConfig.VIDEO_CSV_HEADER.joinToString(","))
        for (r in rows) {
            pw.println(
                listOf(
                    r.taskId, r.quantId, r.split, r.videoId,
                    r.manipulationType, r.binaryLabel,
                    r.numFramesUsed, r.meanScoreFake, r.majorityVotePred,
                    r.fps, r.deviceName, r.runtime,
                ).joinToString(",")
            )
        }
    }
}

// ── Video-level aggregation (majority vote) ───────────────────────────────────

/**
 * Aggregate frame-level results to video-level.
 * majority_vote_pred = 1 if mean_score_fake >= 0.5, else 0.
 * fps = 1000 / mean_latency_ms_per_frame.
 */
fun aggregateToVideo(frames: List<FrameResultRow>): List<VideoResultRow> {
    return frames
        .groupBy { it.videoId }
        .map { (videoId, group) ->
            val first = group.first()
            val meanScore = group.map { it.scoreFake }.average().toFloat()
            val meanLatencyMs = group.map { it.latencyMs }.average()
            val fps = if (meanLatencyMs > 0) (1000.0 / meanLatencyMs).toFloat() else 0f
            VideoResultRow(
                taskId           = first.taskId,
                quantId          = first.quantId,
                split            = first.split,
                videoId          = videoId,
                manipulationType = first.manipulationType,
                binaryLabel      = first.binaryLabel,
                numFramesUsed    = group.size,
                meanScoreFake    = meanScore,
                majorityVotePred = if (meanScore >= 0.5f) 1 else 0,
                fps              = fps,
                deviceName       = first.deviceName,
                runtime          = first.runtime,
            )
        }
}
