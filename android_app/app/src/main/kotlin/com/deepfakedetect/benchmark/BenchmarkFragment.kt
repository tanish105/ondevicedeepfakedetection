package com.deepfakedetect.benchmark

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class BenchmarkFragment : Fragment() {

    private lateinit var btnStart: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var tvStatus: TextView
    private lateinit var tvLog: TextView
    private lateinit var scrollView: ScrollView

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View = inflater.inflate(R.layout.fragment_benchmark, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        btnStart    = view.findViewById(R.id.btnStart)
        progressBar = view.findViewById(R.id.progressBar)
        tvStatus    = view.findViewById(R.id.tvStatus)
        tvLog       = view.findViewById(R.id.tvLog)
        scrollView  = view.findViewById(R.id.scrollView)

        btnStart.setOnClickListener { startBenchmark() }
    }

    private fun startBenchmark() {
        btnStart.isEnabled = false
        progressBar.visibility = View.VISIBLE
        tvStatus.text = "Running benchmark..."
        tvLog.text = ""

        val runner = BenchmarkRunner(requireContext()) { message ->
            if (isAdded) requireActivity().runOnUiThread { appendLog(message) }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                try {
                    runner.runAll()
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        appendLog("\nFATAL ERROR: ${e.message}")
                    }
                }
            }
            progressBar.visibility = View.GONE
            tvStatus.text = "Done — pull results with: adb pull /sdcard/deepfake_benchmark/results/"
            btnStart.isEnabled = true
        }
    }

    private fun appendLog(message: String) {
        tvLog.append(message + "\n")
        scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
    }
}
