package ai.secondsense.app.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import ai.secondsense.app.inference.FrameResult
import ai.secondsense.app.inference.InferenceEngine

/**
 * Bridges CameraX's live frame stream (build-map #6) into the [InferenceEngine].
 *
 * Runs on CameraX's analysis executor (a background thread), so inference never
 * touches the main thread. Uses STRATEGY_KEEP_ONLY_LATEST so we always process the
 * freshest frame and drop backlog — correct for a real-time aid where a stale cue
 * is worse than a skipped one.
 *
 * The result is handed back via [onResult] for the output channels + debug HUD.
 */
class FrameAnalyzer(
    private val engine: InferenceEngine,
    /** Optional secondary consumer of the same upright frame (ML Kit OCR / face). Must be
     *  cheap + non-blocking — it runs inline on the analysis thread before the frame is freed.
     *  Kept BEFORE onResult so a trailing-lambda call `FrameAnalyzer(engine) { r -> ... }`
     *  still binds to onResult. */
    private val frameSink: ((Bitmap) -> Unit)? = null,
    private val onResult: (FrameResult) -> Unit,
) : ImageAnalysis.Analyzer {

    /** Flow mode center-crop toggle (#14). Default on: flow mode is the walking default. */
    @Volatile var centerCrop: Boolean = true

    override fun analyze(image: ImageProxy) {
        try {
            val bitmap = image.toBitmapUpright()
            if (bitmap != null) {
                val result = engine.infer(bitmap, centerCrop)
                onResult(result)
                runCatching { frameSink?.invoke(bitmap) }
            }
        } finally {
            // MUST close or the pipeline stalls — CameraX reuses the buffer.
            image.close()
        }
    }
}

/**
 * CameraX 1.3+ provides ImageProxy.toBitmap() for RGBA_8888 output, but it hands the frame
 * back in SENSOR orientation. On a portrait-held phone the sensor is landscape, so the raw
 * bitmap is rotated 90°. If we fed that straight to the models, every bounding box — and
 * therefore every pan/azimuth cue (#18) and drop-off band (#17) — would be measured along
 * the wrong axis. `Preprocess` explicitly assumes an UPRIGHT bitmap, so we rotate here by
 * `imageInfo.rotationDegrees` (the degrees needed to make the frame upright for the current
 * target rotation) before the engine ever sees it. This is the prerequisite the TFLite
 * integration guide flagged.
 */
fun ImageProxy.toBitmapUpright(): Bitmap? = try {
    val raw = this.toBitmap()
    val deg = this.imageInfo.rotationDegrees
    if (deg == 0) {
        raw
    } else {
        val m = Matrix().apply { postRotate(deg.toFloat()) }
        val rotated = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
        if (rotated !== raw) raw.recycle()   // free the pre-rotation frame now, not at GC
        rotated
    }
} catch (t: Throwable) {
    null
}
