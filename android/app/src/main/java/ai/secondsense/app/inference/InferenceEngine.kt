package ai.secondsense.app.inference

import android.graphics.Bitmap

/**
 * The one seam that keeps SecondSense laptop-buildable before the phone/NPU exist.
 *
 * WHY THIS EXISTS
 * ---------------
 * Ticket #6 has to be built and tested with NO converted model binaries in hand
 * (those come from convert.py on Qualcomm's cloud, tickets #8–#11). So the camera
 * loop, the output channels, and the UI all talk to THIS interface, and we ship a
 * [MockInferenceEngine] now. When the first QNN context binary lands, a
 * `QnnInferenceEngine : InferenceEngine` drops in behind the same interface and
 * NOTHING upstream changes.
 *
 * INVARIANT (Bible, build-map #6): the laptop is never a compute dependency. Both
 * implementations run entirely on-device. The mock computes locally too — it just
 * fabricates detections instead of reading the NPU.
 */
interface InferenceEngine {

    /** Human-readable name for the debug HUD (e.g. "mock", "qnn:yolov11+depth"). */
    val name: String

    /**
     * True once [initialize] has actually finished loading models (default true — the mock
     * has nothing to load). Real engines (TFLite/QNN) flip this only after the NNAPI/NPU
     * graph compile completes, which can take 15-30s on first run. Exists so the HUD can
     * show "warming up" instead of a frame counter that looks broken because [infer] is
     * silently returning empty results while the camera keeps streaming frames.
     */
    val isReady: Boolean get() = true

    /** One-time setup (load binaries, allocate tensors). Safe to call off the main thread. */
    fun initialize()

    /**
     * Run one frame through the pipeline.
     *
     * @param frame       the current camera frame as an ARGB bitmap.
     * @param centerCrop  if true, only the center ~30% region is considered
     *                    (flow mode, ticket #14). The mock honors this so the
     *                    center-crop behavior is demoable before YOLO exists.
     * @return the detections + timing for this frame.
     */
    fun infer(frame: Bitmap, centerCrop: Boolean): FrameResult

    /** Release native resources. No-op for the mock. */
    fun close()
}
