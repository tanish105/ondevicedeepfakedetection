plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.deepfakedetect.benchmark"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.deepfakedetect.benchmark"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    // TFLite models must not be compressed inside the APK — they are
    // memory-mapped at runtime and compression breaks that.
    aaptOptions {
        noCompress("tflite")
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.lifecycle.runtime.ktx)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.tensorflow.lite)
    implementation(libs.tensorflow.lite.support) {
        // tensorflow-lite-support:0.4.4 pulls in tensorflow-lite-api:2.13.0, which
        // conflicts with litert-api:1.0.1 bundled in tensorflow-lite:2.17.0.
        // Exclude the old api artifact so only litert-api is on the classpath.
        exclude(group = "org.tensorflow", module = "tensorflow-lite-api")
    }
}
