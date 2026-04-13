package com.deepfakedetect.benchmark

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var btnStart: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var tvStatus: TextView
    private lateinit var tvLog: TextView
    private lateinit var scrollView: ScrollView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnStart    = findViewById(R.id.btnStart)
        progressBar = findViewById(R.id.progressBar)
        tvStatus    = findViewById(R.id.tvStatus)
        tvLog       = findViewById(R.id.tvLog)
        scrollView  = findViewById(R.id.scrollView)

        requestStoragePermissionIfNeeded()

        btnStart.setOnClickListener { startBenchmark() }
    }

    private fun startBenchmark() {
        btnStart.isEnabled = false
        progressBar.visibility = View.VISIBLE
        tvStatus.text = "Running benchmark..."
        tvLog.text = ""

        val runner = BenchmarkRunner(this) { message ->
            // Called from background thread — post to main thread for UI update
            runOnUiThread { appendLog(message) }
        }

        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                try {
                    runner.runAll()
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        appendLog("\nFATAL ERROR: ${e.message}")
                    }
                }
            }
            // Back on main thread
            progressBar.visibility = View.GONE
            tvStatus.text = "Done — pull results with: adb pull /sdcard/deepfake_benchmark/results/"
            btnStart.isEnabled = true
        }
    }

    private fun appendLog(message: String) {
        tvLog.append(message + "\n")
        scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
    }

    // ── Storage permission (Android 11+) ──────────────────────────────────────
    private fun requestStoragePermissionIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                    data = Uri.parse("package:$packageName")
                }
                startActivity(intent)
            }
        }
    }
}
