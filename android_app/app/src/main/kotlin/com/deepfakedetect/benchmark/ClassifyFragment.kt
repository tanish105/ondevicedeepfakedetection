package com.deepfakedetect.benchmark

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ClassifyFragment : Fragment() {

    // ── Views ──────────────────────────────────────────────────────────────────
    private lateinit var ivPreview: ImageView
    private lateinit var btnPick: Button
    private lateinit var btnRun: Button
    private lateinit var tvResultEmpty: TextView
    private lateinit var layoutLoading: LinearLayout
    private lateinit var layoutResult: LinearLayout
    private lateinit var tvVerdictIcon: TextView
    private lateinit var tvVerdict: TextView
    private lateinit var tvScore: TextView
    private lateinit var pbConfidence: ProgressBar
    private lateinit var tvLatency: TextView

    // ── State ──────────────────────────────────────────────────────────────────
    private var selectedBitmap: Bitmap? = null
    private var engine: TFLiteEngine? = null

    // ── Image picker ───────────────────────────────────────────────────────────
    private val pickImage = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) loadImage(uri)
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View = inflater.inflate(R.layout.fragment_classify, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        ivPreview     = view.findViewById(R.id.ivPreview)
        btnPick       = view.findViewById(R.id.btnPick)
        btnRun        = view.findViewById(R.id.btnRun)
        tvResultEmpty = view.findViewById(R.id.tvResultEmpty)
        layoutLoading = view.findViewById(R.id.layoutLoading)
        layoutResult  = view.findViewById(R.id.layoutResult)
        tvVerdictIcon = view.findViewById(R.id.tvVerdictIcon)
        tvVerdict     = view.findViewById(R.id.tvVerdict)
        tvScore       = view.findViewById(R.id.tvScore)
        pbConfidence  = view.findViewById(R.id.pbConfidence)
        tvLatency     = view.findViewById(R.id.tvLatency)

        engine = try {
            TFLiteEngine(requireContext(), MODEL_ASSET)
        } catch (e: Exception) {
            showErrorState("Model could not be loaded: ${e.message}")
            null
        }

        btnPick.setOnClickListener {
            pickImage.launch(arrayOf("image/*"))
        }

        btnRun.setOnClickListener { runDetection() }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        selectedBitmap?.recycle()
        selectedBitmap = null
        engine?.close()
        engine = null
    }

    // ── Image loading ──────────────────────────────────────────────────────────

    private fun loadImage(uri: Uri) {
        val contentResolver = requireContext().contentResolver
        viewLifecycleOwner.lifecycleScope.launch {
            val bitmap: Bitmap? = withContext(Dispatchers.IO) {
                try {
                    contentResolver.openInputStream(uri)?.use { stream ->
                        BitmapFactory.decodeStream(stream)
                    }
                } catch (e: Exception) {
                    null
                }
            }
            if (bitmap != null) {
                selectedBitmap?.recycle()
                selectedBitmap = bitmap
                ivPreview.setImageBitmap(bitmap)
                // Reset result panel so previous result doesn't persist
                tvResultEmpty.text = "Select an image and tap Run Detection"
                tvResultEmpty.visibility = View.VISIBLE
                layoutLoading.visibility = View.GONE
                layoutResult.visibility  = View.GONE
                if (engine != null) btnRun.isEnabled = true
            } else {
                Toast.makeText(requireContext(), "Could not load image", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ── Inference ──────────────────────────────────────────────────────────────

    private fun runDetection() {
        val bitmap = selectedBitmap ?: return
        val eng = engine ?: return

        btnRun.isEnabled = false
        btnPick.isEnabled = false
        showLoadingState()

        viewLifecycleOwner.lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                try {
                    val floats = ImagePreprocessor.preprocess(bitmap)
                    val t0 = System.nanoTime()
                    val score = eng.infer(floats)
                    val latencyMs = (System.nanoTime() - t0) / 1_000_000f
                    Result.success(score to latencyMs)
                } catch (e: Exception) {
                    Result.failure(e)
                }
            }
            if (result.isSuccess) {
                val (score, latency) = result.getOrThrow()
                showResult(scoreFake = score, latencyMs = latency)
            } else {
                showErrorState("Detection failed: ${result.exceptionOrNull()?.message}")
            }
            btnRun.isEnabled = true
            btnPick.isEnabled = true
        }
    }

    // ── UI state helpers ───────────────────────────────────────────────────────

    private fun showLoadingState() {
        tvResultEmpty.visibility = View.GONE
        layoutLoading.visibility = View.VISIBLE
        layoutResult.visibility  = View.GONE
    }

    private fun showResult(scoreFake: Float, latencyMs: Float) {
        val isDeepfake = scoreFake >= 0.5f

        tvVerdictIcon.text = if (isDeepfake) "⚠️" else "✅"
        tvVerdict.text     = if (isDeepfake) "DEEPFAKE" else "REAL"
        tvVerdict.setTextColor(
            requireContext().getColor(
                if (isDeepfake) android.R.color.holo_red_light
                else            android.R.color.holo_green_light
            )
        )
        tvScore.text        = "score_fake: ${"%.2f".format(scoreFake)}"
        pbConfidence.progress = (scoreFake * 100).toInt()
        tvLatency.text      = "inference only: ${"%.1f".format(latencyMs)} ms"

        tvResultEmpty.visibility = View.GONE
        layoutLoading.visibility = View.GONE
        layoutResult.visibility  = View.VISIBLE
    }

    private fun showErrorState(message: String) {
        tvResultEmpty.text = message
        tvResultEmpty.visibility = View.VISIBLE
        layoutLoading.visibility = View.GONE
        layoutResult.visibility  = View.GONE
        btnRun.isEnabled  = false
        btnPick.isEnabled = false
    }

    // ── Constants ──────────────────────────────────────────────────────────────

    companion object {
        private const val MODEL_ASSET = "mobilenetv2_df_vs_real_float16.tflite"
    }
}
