package com.deepfakedetect.benchmark

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Wraps a single TFLite model loaded from assets.
 *
 * Input:  (1, 224, 224, 3) float32 NHWC — pass a FloatArray of length
 *         IMAGE_SIZE * IMAGE_SIZE * 3, batch dimension is handled internally.
 * Output: (1, 1) float32 — score_fake in [0, 1] (sigmoid baked into export).
 *
 * Call [close] when done to release native resources.
 */
class TFLiteEngine(context: Context, assetName: String) : AutoCloseable {

    private val interpreter: Interpreter
    private val size = BenchmarkConfig.IMAGE_SIZE

    init {
        val model = loadModelFile(context, assetName)
        val options = Interpreter.Options().apply {
            numThreads = 4
        }
        interpreter = Interpreter(model, options)
    }

    /**
     * Run inference on one preprocessed image.
     *
     * @param inputFloats FloatArray of length size*size*3 (NHWC, no batch dim).
     * @return score_fake clamped to [0, 1].
     */
    fun infer(inputFloats: FloatArray): Float {
        // Input buffer: 1 * H * W * 3 * 4 bytes
        val inputBuffer = ByteBuffer
            .allocateDirect(inputFloats.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        inputBuffer.asFloatBuffer().put(inputFloats)
        inputBuffer.rewind()

        // Output buffer: 1 * 1 * 4 bytes
        val outputBuffer = ByteBuffer
            .allocateDirect(Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val raw = outputBuffer.float
        return raw.coerceIn(0f, 1f)
    }

    override fun close() {
        interpreter.close()
    }

    private fun loadModelFile(context: Context, assetName: String): MappedByteBuffer {
        val afd = context.assets.openFd(assetName)
        val inputStream = FileInputStream(afd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
    }
}
