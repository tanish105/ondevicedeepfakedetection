package com.deepfakedetect.benchmark

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import org.json.JSONObject
import java.io.File

/**
 * Collects hardware and OS info and writes device_specs.json to RESULTS_DIR.
 * Required by project_contract.md §11.
 */
object DeviceInfo {

    // TFLite version comes from libs.versions.toml — hardcoded here to avoid
    // a runtime dependency on internal TFLite APIs.
    private const val TFLITE_VERSION = "2.17.0"

    fun write(context: Context, resultsDir: File) {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val totalRamMb = memInfo.totalMem / (1024L * 1024L)

        val manufacturer = Build.MANUFACTURER.trim()
        val model = Build.MODEL.trim()
        val deviceName = if (model.startsWith(manufacturer, ignoreCase = true)) model
                         else "$manufacturer $model"

        val json = JSONObject().apply {
            put("device_name",        deviceName)
            put("manufacturer",       Build.MANUFACTURER)
            put("model",              Build.MODEL)
            put("hardware",           Build.HARDWARE)          // SoC / chipset
            put("android_version",    Build.VERSION.RELEASE)
            put("android_api_level",  Build.VERSION.SDK_INT)
            put("total_ram_mb",       totalRamMb)
            put("tflite_version",     TFLITE_VERSION)
        }

        resultsDir.mkdirs()
        File(resultsDir, "device_specs.json").writeText(json.toString(2))
    }
}
