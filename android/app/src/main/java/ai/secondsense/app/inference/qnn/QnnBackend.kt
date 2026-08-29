package ai.secondsense.app.inference.qnn

import ai.secondsense.app.inference.decode.RawTensor
import java.nio.ByteBuffer

/**
 * The seam between the QNN runtime (Qualcomm AI Engine Direct — the Hexagon NPU / HTP) and
 * SecondSense's runtime-agnostic decode layer.
 *
 * WHY A SEAM: the compiled models on the phone are `qnn_context_binary` (.bin) files that run
 * through a native C++ runtime regardless of app framework. A real implementation of this
 * interface is a thin JNI bridge (src/main/cpp + the externalNativeBuild block in
 * build.gradle, currently stubbed off) that loads a context binary and executes it on the
 * HTP, returning flat float tensors. Everything DOWNSTREAM of this — [ai.secondsense.app
 * .inference.decode] — is shared with the TFLite path unchanged, which is the whole point:
 * when the iQOO 15 + QNN SDK are in hand, only this bridge is filled in; the decode + fusion
 * + sonification stack does not change.
 *
 * Until that native bridge exists, [StubQnnBackend] reports notReady, so [ai.secondsense.app
 * .inference.EngineConfig] falls back to the TFLite engine (or the mock) and the app never
 * hard-crashes on a missing runtime.
 */
interface QnnBackend {
    /** Human-readable name for the debug HUD. */
    val name: String

    /** True once the native runtime is loaded and every model asked for has loaded. */
    fun isReady(): Boolean

    /**
     * Load a context binary for logical [model] name (e.g. "yolo", "depth") from raw bytes.
     * @return true on success.
     */
    fun load(model: String, bytes: ByteBuffer): Boolean

    /**
     * Execute [model] with one float input buffer (NHWC/NCHW per the binary) and return its
     * outputs as flat [RawTensor]s — the exact shape the shared decode layer consumes.
     */
    fun run(model: String, input: ByteBuffer): List<RawTensor>

    /** Release native handles. */
    fun close()
}

/**
 * Placeholder backend used until the native QNN JNI bridge is added (needs the phone + the
 * Qualcomm QNN SDK to build). It is never "ready", so EngineConfig routes around it. Kept as
 * a real type — not a null — so the QNN engine and its wiring compile and are exercised today.
 */
class StubQnnBackend : QnnBackend {
    override val name: String = "qnn:stub(no-native-bridge-yet)"
    override fun isReady(): Boolean = false
    override fun load(model: String, bytes: ByteBuffer): Boolean = false
    override fun run(model: String, input: ByteBuffer): List<RawTensor> =
        throw UnsupportedOperationException(
            "QNN native bridge not present yet — add src/main/cpp + the QNN SDK, then " +
                "implement QnnBackend.run(). Until then EngineConfig uses TFLite/MOCK."
        )
    override fun close() {}
}
