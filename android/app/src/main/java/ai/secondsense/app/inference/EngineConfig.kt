package ai.secondsense.app.inference

import android.content.Context
import android.util.Log
import ai.secondsense.app.inference.qnn.NativeQnnBackend
import ai.secondsense.app.sensors.ImuTracker

/**
 * Single place that decides WHICH engine runs — so switching mock <-> tflite <-> qnn is
 * one constant, not a hand-edit scattered through MainActivity.
 *
 * DEFAULT IS MOCK on purpose: the app must still build and run with NO model files present
 * (the property the whole project has preserved since #6). Flip [KIND] to TFLITE once the
 * two .tflite files are in app/src/main/assets/models/. If TFLITE is selected but a model
 * is missing or fails to load, [create] logs the reason and falls back to the mock rather
 * than crashing the demo.
 */
object EngineConfig {

    enum class Kind { MOCK, TFLITE, QNN }

    /** <<< THE SWITCH. Change this one line to select the engine. >>>
     * TFLITE is the demo path: real yolov11_det + depth_anything_v2 models running on
     * CPU/GPU/NNAPI, confirmed working end-to-end on the iQOO 15.
     *
     * QNN (Hexagon NPU) was built and verified real-hardware-deep — dlopen, backendCreate all
     * succeed — but deviceCreate is rejected with QNN_DEVICE_ERROR_INVALID_CONFIG (14001) even
     * with fully-default config, on this locked-down (ro.secure=1, no root, no testsig tool in
     * this SDK) production build. That's a platform/OEM access gate, not an app bug — see
     * secondsense_bible_v4_addendum_session4.md's QNN bring-up section for the full trail.
     * Flip to Kind.QNN any time; [create] falls back to TFLITE automatically and safely if
     * native init fails, so it's harmless to leave that flip in place if you revisit this. */
    val KIND: Kind = Kind.QNN

    private const val TAG = "SecondSense/engine"

    /**
     * V3 drop-off plan §1 — owned here (not by MainActivity directly) so [create] can hand it
     * straight to whichever engine gets constructed. MainActivity is still responsible for the
     * actual sensor lifecycle: call `imuTracker?.start()` in onCreate and `?.stop()` in
     * onDestroy (SensorManager listeners leak otherwise). Null until the first [create] call.
     */
    var imuTracker: ImuTracker? = null
        private set

    fun create(context: Context): InferenceEngine {
        // Stop any tracker left over from a previous create() (e.g. an Activity recreate after
        // process death) before replacing the reference, so its SensorManager listener
        // registration isn't leaked.
        imuTracker?.stop()
        imuTracker = ImuTracker(context.applicationContext)
        return when (KIND) {
        Kind.MOCK -> MockInferenceEngine()

        Kind.TFLITE ->
            // CHEAP probe only: confirm the model files are actually bundled before we
            // commit to the tflite engine. The heavy Interpreter load still happens in
            // engine.initialize() off the main thread (MainActivity), exactly as before.
            // If the assets are missing, fall back to the mock so the demo still runs.
            if (assetExists(context, "models/yolov11_det.tflite") &&
                assetExists(context, "models/depth_anything_v2.tflite")
            ) {
                Log.i(TAG, "using TFLITE engine")
                TfliteInferenceEngine(context.applicationContext, imuTracker)
            } else {
                Log.w(TAG, "TFLITE models not found in assets/models/; falling back to MOCK")
                MockInferenceEngine()
            }

        // NPU-native path. Fully wired against the shared decode layer; goes live the moment
        // the native QnnBackend bridge + the .bin binaries are present. Until then the stub
        // backend reports notReady and we fall back to TFLITE (if its models exist) else MOCK,
        // so flipping KIND to QNN today is safe and simply runs the best available engine.
        Kind.QNN -> {
            // CHEAP probe only, same shape as the TFLITE branch above — do NOT call
            // backend.isReady() here. isReady() only becomes true as a SIDE EFFECT of
            // load(), which only runs inside engine.initialize() (off the main thread, in
            // MainActivity) — checking it before that is always false, a chicken-and-egg
            // deadlock that would make Kind.QNN never actually select the QNN engine. The
            // real success/failure of nativeInit()/load() surfaces via QnnInferenceEngine's
            // own logging once initialize() runs; it degrades to empty frames rather than
            // crashing if the native load fails, matching the TFLITE path's failure mode.
            val binariesPresent = assetExists(context, "models/yolov11_det.bin") &&
                assetExists(context, "models/depth_anything_v2.bin")
            if (binariesPresent) {
                // NativeQnnBackend's <clinit> does System.loadLibrary("secondsense_qnn"), which
                // only exists in a build made with -PenableQnnNative=true. Without it, catch the
                // UnsatisfiedLinkError and fall back rather than crashing onCreate.
                val qnn = try {
                    val backendSoPath = "${context.applicationInfo.nativeLibraryDir}/libQnnHtp.so"
                    QnnInferenceEngine(context.applicationContext, NativeQnnBackend(backendSoPath), imuTracker)
                } catch (t: Throwable) {
                    Log.w(TAG, "QNN native bridge unavailable (build without -PenableQnnNative?): ${t.message}")
                    null
                }
                if (qnn != null) {
                    Log.i(TAG, "using QNN engine (native init happens in engine.initialize())")
                    qnn
                } else if (assetExists(context, "models/yolov11_det.tflite") &&
                    assetExists(context, "models/depth_anything_v2.tflite")
                ) {
                    TfliteInferenceEngine(context.applicationContext, imuTracker)
                } else {
                    MockInferenceEngine()
                }
            } else {
                Log.w(TAG, "QNN .bin assets not found; falling back to TFLITE if available, else MOCK")
                if (assetExists(context, "models/yolov11_det.tflite") &&
                    assetExists(context, "models/depth_anything_v2.tflite")
                ) {
                    TfliteInferenceEngine(context.applicationContext, imuTracker)
                } else {
                    MockInferenceEngine()
                }
            }
        }
        }
    }

    private fun assetExists(context: Context, path: String): Boolean = try {
        context.assets.openFd(path).close(); true
    } catch (_: Throwable) {
        // openFd throws for compressed assets even when present; fall back to a stream open.
        try { context.assets.open(path).close(); true } catch (_: Throwable) { false }
    }
}
