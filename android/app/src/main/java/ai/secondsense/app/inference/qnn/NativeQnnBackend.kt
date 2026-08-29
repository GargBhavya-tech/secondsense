package ai.secondsense.app.inference.qnn

import ai.secondsense.app.inference.decode.RawTensor
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Real [QnnBackend] over the JNI bridge in `src/main/cpp/qnn_backend_jni.cpp`.
 *
 * WIRED IN via [EngineConfig] (Kind.QNN). Builds clean against the real QAIRT 2.49 headers,
 * including full graphExecute tensor binding — see qnn_backend_jni.cpp's header comment for
 * exact status. Not yet confirmed working end-to-end on-device; if nativeInit()/load() fails
 * for any reason, [EngineConfig] falls back to TFLite automatically.
 *
 * @param backendSoPath absolute path to the backend .so ON THE DEVICE after it's unpacked from
 *   jniLibs (e.g. via `applicationInfo.nativeLibraryDir + "/libQnnHtp.so"`) — Android extracts
 *   every jniLibs .so for the device's ABI into that directory at install time, so a plain
 *   filename resolves via the linker's search path but an absolute path is what dlopen wants
 *   for a specific backend when more than one backend .so is bundled (HTP + CPU fallback).
 */
class NativeQnnBackend(private val backendSoPath: String) : QnnBackend {

    override val name: String = "qnn:native(htp)"

    private var initialized = false

    override fun isReady(): Boolean = initialized

    override fun load(model: String, bytes: ByteBuffer): Boolean {
        if (!initialized) {
            initialized = nativeInit(backendSoPath)
            if (!initialized) {
                Log.e(TAG, "nativeInit failed for $backendSoPath — backend .so missing, wrong " +
                    "ABI, or QNN SDK/device mismatch. Falls back to TFLite via EngineConfig.")
                return false
            }
        }
        // JNI's GetDirectBufferAddress requires a DIRECT buffer — a plain ByteBuffer.wrap()
        // (heap-backed) returns null on the native side and load silently fails there, so
        // enforce it here where the error is legible.
        val direct = ensureDirect(bytes)
        return nativeLoadModel(model, direct)
    }

    override fun run(model: String, input: ByteBuffer): List<RawTensor> {
        val direct = ensureDirect(input)
        val raw = nativeRun(model, direct)
        // Native returns a flat Object[] alternating IntArray (shape) / FloatArray (data) per
        // output tensor — see qnn_backend_jni.cpp's nativeRun doc comment. Empty until
        // nativeRun's TODO (graphExecute tensor binding) is finished against the real SDK.
        val tensors = ArrayList<RawTensor>(raw.size / 2)
        var i = 0
        while (i + 1 < raw.size) {
            val shape = raw[i] as IntArray
            val data = raw[i + 1] as FloatArray
            tensors += RawTensor(data, shape)
            i += 2
        }
        return tensors
    }

    override fun close() {
        if (initialized) nativeClose()
        initialized = false
    }

    private fun ensureDirect(buf: ByteBuffer): ByteBuffer {
        if (buf.isDirect) return buf
        val d = ByteBuffer.allocateDirect(buf.remaining()).order(ByteOrder.nativeOrder())
        val dup = buf.duplicate()
        d.put(dup)
        d.flip()
        return d
    }

    private external fun nativeInit(backendSoPath: String): Boolean
    private external fun nativeLoadModel(model: String, bytes: ByteBuffer): Boolean
    private external fun nativeRun(model: String, input: ByteBuffer): Array<Any>
    private external fun nativeClose()

    private companion object {
        const val TAG = "SecondSense/qnn_native"
        init {
            // Matches add_library(secondsense_qnn ...) in CMakeLists.txt. Throws
            // UnsatisfiedLinkError if the .so wasn't built (native build disabled by default —
            // see build.gradle.kts's enableQnnNative gate) or isn't bundled for this ABI.
            System.loadLibrary("secondsense_qnn")
        }
    }
}
