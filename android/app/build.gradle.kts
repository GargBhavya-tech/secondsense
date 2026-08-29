plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.secondsense.app"
    compileSdk = 34
    // Pin to an installed build-tools so the build doesn't try to auto-download 34.0.0
    // (this machine has 35/36/37). Bump if your SDK differs.
    buildToolsVersion = "35.0.0"

    defaultConfig {
        applicationId = "ai.secondsense.app"
        minSdk = 30          // Android 11; iQOO 15 ships far above this
        targetSdk = 34
        versionCode = 1
        versionName = "0.2-tflite-testpath"

        // ABIs. arm64-v8a is the iQOO's real-device ABI; x86_64 is added so the tflite
        // TEST PATH runs on a standard x86_64 emulator too (LiteRT/TFLite AARs ship native
        // .so per ABI). Drop x86_64 again for the final real-device/QNN build if you want a
        // smaller APK — QNN context binaries are arm64-only.
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }

    // .tflite/.bin must NOT be compressed in the APK, so the engines can open/memory-map the
    // asset fd directly (TfliteInferenceEngine.loadModel uses FileChannel.map; QnnInferenceEngine
    // .readAsset uses openFd — both throw "probably compressed" FileNotFoundException otherwise,
    // as .bin genuinely did on first real-device test before this was added).
    androidResources {
        noCompress += "tflite"
        noCompress += "bin"
    }

    // ---------------------------------------------------------------------
    // NATIVE (QNN) LAYER — off by default. The tflite test path is pure JVM (LiteRT/TFLite
    // Java API), no NDK/CMake needed for it. The QNN JNI bridge (src/main/cpp/) exists on
    // disk but is only compiled when you pass -PenableQnnNative=true AND have set
    // qnn.sdk.root in gradle.properties (or the QNN_SDK_ROOT env var) to the unpacked
    // Qualcomm QNN SDK. Without both, `gradlew assembleDebug` skips this block entirely and
    // the app builds exactly as it did before this bridge was added.
    val enableQnnNative = project.findProperty("enableQnnNative")?.toString().toBoolean()
    if (enableQnnNative) {
        val qnnSdkRoot = (project.findProperty("qnn.sdk.root") as String?)
            ?: System.getenv("QNN_SDK_ROOT")
            ?: throw GradleException(
                "enableQnnNative=true but no QNN SDK path found. Set qnn.sdk.root in " +
                    "gradle.properties or the QNN_SDK_ROOT environment variable to the " +
                    "unpacked Qualcomm QNN SDK directory."
            )
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
        defaultConfig {
            externalNativeBuild {
                cmake {
                    arguments += "-DQNN_SDK_ROOT=$qnnSdkRoot"
                }
            }
        }
        // The QNN backend .so files (libQnnHtp.so, libQnnSystem.so, etc.) themselves are NOT
        // compiled — they're Qualcomm's prebuilt binaries. They need to be copied into
        // src/main/jniLibs/arm64-v8a/ from $qnnSdkRoot/lib/aarch64-android/ (a one-time manual
        // step, or a Gradle Copy task if this gets automated later) so they're packaged into
        // the APK and resolvable by NativeQnnBackend's dlopen() path.
        sourceSets.getByName("main").jniLibs.srcDir("src/main/jniLibs")

        // REAL BUG FOUND on first device test: modern AGP defaults to NOT extracting native
        // libraries to disk (they're mmap'd straight out of the uncompressed APK instead, an
        // install-size optimization). System.loadLibrary() handles that transparently via the
        // classloader's native search path, but NativeQnnBackend.dlopen()'s ABSOLUTE PATH to
        // libQnnHtp.so does not — dlopen() needs a real file on disk. Confirmed on-device:
        // "dlopen(.../lib/arm64/libQnnHtp.so) failed: ... not found" until this was added.
        packaging {
            jniLibs {
                useLegacyPackaging = true
            }
        }
    }

    // ---------------------------------------------------------------------
    // OPTIONAL OFFLINE ASR — sherpa-onnx keyword spotting (Phase 4 voice, non-QNN path).
    // OFF by default so `gradlew assembleDebug` is green with no extra files. Enable with
    // -PenableSherpa=true AFTER adding, per android/app/src/sherpa/README.md:
    //   - the sherpa-onnx Kotlin wrapper sources under src/sherpa/kotlin/com/k2fsa/sherpa/onnx/
    //   - libsherpa-onnx-jni.so (+ deps) per ABI under src/main/jniLibs/<abi>/
    //   - the KWS model files under src/main/assets/kws/
    // Without the flag, src/sherpa/kotlin/ (incl. SherpaKwsRecognizer.kt) is NOT compiled and
    // VoiceRecognizers falls back to the QNN Whisper stub.
    val enableSherpa = project.findProperty("enableSherpa")?.toString().toBoolean()
    if (enableSherpa) {
        sourceSets.getByName("main").java.srcDir("src/sherpa/kotlin")
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // CameraX — the live frame stream (build-map #6, feeds #12/#13)
    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // Coroutines — off-main-thread inference loop
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // ---- TFLite TEST PATH ----------------------------------------------------
    // Classic Interpreter API (org.tensorflow.lite.Interpreter) — stable, universally
    // documented, and what TfliteInferenceEngine is written against. Let Gradle bump to
    // the latest if Android Studio flags a newer one.
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    // Optional GPU delegate (Adreno). Enable useGpu=true in the engine AND uncomment the
    // addDelegate line there after adding this:
    // implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    //
    // NOTE: the current *rebranded* successor is LiteRT — `com.google.ai.edge.litert:litert`
    // (its headline API is the newer CompiledModel API). The classic Interpreter path above
    // consumes the identical .tflite and is the lower-risk choice for a test harness; move
    // to LiteRT's CompiledModel later if you want its automatic accelerator selection.

    // ---- #30 laptop dashboard + QR ---------------------------------------------
    // Both fully offline/local — no network dependency beyond the phone's own Wi-Fi/hotspot
    // interface, which is required for the airplane-mode demo (#31).
    // NanoHTTPD: tiny embedded HTTP server (~7 classes) serving the dashboard page + live
    // telemetry JSON to any laptop browser on the same local network.
    implementation("org.nanohttpd:nanohttpd:2.3.1")
    // ZXing core only (no android-embedded scanning lib needed) — encodes the dashboard URL
    // as a QR bitmap so a laptop can join by camera-scan instead of typing an IP.
    implementation("com.google.zxing:core:3.5.3")

    // ---- ML Kit — bundled, fully OFFLINE (models ship in the APK, no Play Services fetch) --
    // Text recognition -> read signs / room numbers / bus numbers (Latin script).
    implementation("com.google.mlkit:text-recognition:16.0.1")
    // Face detection -> "person facing you" via head Euler-Y angle. Contour/classification off.
    implementation("com.google.mlkit:face-detection:16.1.7")

    // Unit tests for the pure-logic layer (targeting, channel math)
    testImplementation("junit:junit:4.13.2")
}
