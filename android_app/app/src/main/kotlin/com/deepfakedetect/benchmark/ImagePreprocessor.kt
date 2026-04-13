package com.deepfakedetect.benchmark

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File

/**
 * Loads a PNG image from disk and converts it to a float32 NHWC tensor
 * matching the TFLite model's expected input: (1, 224, 224, 3).
 *
 * Preprocessing mirrors Python tflite_export._preprocess_nhwc():
 *   1. Decode PNG as RGB.
 *   2. Bilinear resize to IMAGE_SIZE x IMAGE_SIZE.
 *   3. Normalize each channel: (pixel/255 - mean) / std.
 *   4. Layout: NHWC — the output ByteBuffer / FloatArray is row-major
 *      [H][W][C] with a leading batch dimension of 1.
 */
object ImagePreprocessor {

    private val size = BenchmarkConfig.IMAGE_SIZE
    private val mean = BenchmarkConfig.IMAGENET_MEAN
    private val std  = BenchmarkConfig.IMAGENET_STD

    /**
     * Returns a FloatArray of length 1 * size * size * 3 (NHWC, batch=1).
     * Returns null if the file cannot be decoded.
     */
    fun preprocess(imagePath: String): FloatArray? {
        val file = File(imagePath)
        if (!file.exists()) return null

        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val raw = BitmapFactory.decodeFile(imagePath, options) ?: return null

        // Bilinear resize to 224×224
        val resized = Bitmap.createScaledBitmap(raw, size, size, true)
        if (resized !== raw) raw.recycle()

        // Flatten to NHWC float32
        val pixels = IntArray(size * size)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)
        resized.recycle()

        val floats = FloatArray(size * size * 3)
        var idx = 0
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8)  and 0xFF) / 255f
            val b = (pixel          and 0xFF) / 255f
            floats[idx++] = (r - mean[0]) / std[0]
            floats[idx++] = (g - mean[1]) / std[1]
            floats[idx++] = (b - mean[2]) / std[2]
        }
        return floats
    }
}
