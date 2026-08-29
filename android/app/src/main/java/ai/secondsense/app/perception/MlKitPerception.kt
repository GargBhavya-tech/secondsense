package ai.secondsense.app.perception

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Two extra sensing modalities on the SAME camera frame, both fully OFFLINE (bundled ML Kit):
 *
 *  - **Sign / text reading (OCR)** — indoor story: "read the sign", room numbers, bus numbers,
 *    EXIT signs. Two bundled recognizers run on ALTERNATING passes: Latin script and
 *    Devanagari (Hindi — शौचालय / निकास / प्रवेश …). Only center-of-frame text, debounced so it
 *    doesn't machine-gun the same sign. The callback reports which script matched so the
 *    caller can pick a TTS locale / translate.
 *  - **"Person facing you"** — a social-navigation cue no competitor markets: a detected face
 *    whose head is turned toward the camera (small Euler-Y) and is close enough to matter.
 *
 * Throttled hard (every Nth frame, skip while a prior pass is in flight) — ML Kit is async and
 * must never back up the real-time inference loop. Callbacks fire on the main thread.
 */
class MlKitPerception(
    private val onSign: (text: String, isDevanagari: Boolean) -> Unit,
    private val onFacingPerson: () -> Unit,
    private val processEvery: Int = 6,
    private val textCooldownMs: Long = 4_000L,
    private val faceCooldownMs: Long = 5_000L,
    private val facingMaxDeg: Float = 18f,
    private val minFaceFrac: Float = 0.02f,
) {
    private val textClientLatin: TextRecognizer =
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private val textClientDevanagari: TextRecognizer =
        TextRecognition.getClient(DevanagariTextRecognizerOptions.Builder().build())
    private val faceClient = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .build()
    )

    private var frameCounter = 0L
    private val textBusy = AtomicBoolean(false)
    private val faceBusy = AtomicBoolean(false)
    private var useDevanagariNext = false
    private var lastText: String = ""
    private var lastTextAtMs = 0L
    private var lastFaceAtMs = 0L

    /** Feed the current upright camera frame. Cheap + non-blocking; may no-op (throttled). */
    fun offer(frame: Bitmap) {
        if (frameCounter++ % processEvery != 0L) return
        if (frame.isRecycled) return
        val cfg = frame.config ?: Bitmap.Config.ARGB_8888
        val w = frame.width
        val h = frame.height

        if (textBusy.compareAndSet(false, true)) {
            val copy = runCatching { frame.copy(cfg, false) }.getOrNull()
            if (copy == null) {
                textBusy.set(false)
            } else {
                val deva = useDevanagariNext
                useDevanagariNext = !useDevanagariNext
                runText(copy, w, h, deva)
            }
        }
        if (faceBusy.compareAndSet(false, true)) {
            val copy = runCatching { frame.copy(cfg, false) }.getOrNull()
            if (copy == null) faceBusy.set(false) else runFace(copy, w, h)
        }
    }

    private fun runText(bmp: Bitmap, w: Int, h: Int, devanagari: Boolean) {
        val client = if (devanagari) textClientDevanagari else textClientLatin
        client.process(InputImage.fromBitmap(bmp, 0))
            .addOnSuccessListener { result ->
                val center = StringBuilder()
                for (block in result.textBlocks) {
                    val r = block.boundingBox ?: continue
                    val cx = r.centerX().toFloat() / w
                    val cy = r.centerY().toFloat() / h
                    if (cx in 0.22f..0.78f && cy in 0.12f..0.88f) {
                        if (center.isNotEmpty()) center.append(" - ")
                        center.append(block.text.replace('\n', ' ').trim())
                    }
                }
                val text = center.toString().trim()
                val now = System.currentTimeMillis()
                val hasContent = text.length >= 2 &&
                    text.any { it.isLetterOrDigit() || Character.UnicodeBlock.of(it) == DEVA_BLOCK }
                if (hasContent && text != lastText && now - lastTextAtMs > textCooldownMs) {
                    lastText = text
                    lastTextAtMs = now
                    onSign(text, devanagari)
                }
            }
            .addOnFailureListener { Log.w(TAG, "text(${if (devanagari) "deva" else "latin"}): ${it.message}") }
            .addOnCompleteListener { bmp.recycle(); textBusy.set(false) }
    }

    /**
     * One-shot, on-demand OCR for a spoken "read this" (Phase 3 voice intent). Runs BOTH
     * script recognizers on [frame] once, ignores the [offer] throttle / cooldown / dedupe
     * entirely, and reports the longer of the two center-of-frame results (or null / blank when
     * neither finds text). [onResult] fires on the ML Kit callback thread. [frame] is recycled
     * here — pass a copy you no longer need.
     */
    fun readNow(frame: Bitmap, onResult: (text: String?, isDevanagari: Boolean) -> Unit) {
        if (frame.isRecycled) { onResult(null, false); return }
        val w = frame.width
        val h = frame.height
        val results = arrayOfNulls<String>(2)
        val done = java.util.concurrent.atomic.AtomicInteger(0)

        fun finish() {
            if (done.incrementAndGet() < 2) return
            runCatching { frame.recycle() }
            val latin = results[0].orEmpty()
            val deva = results[1].orEmpty()
            when {
                deva.length > latin.length && deva.isNotBlank() -> onResult(deva, true)
                latin.isNotBlank() -> onResult(latin, false)
                else -> onResult(null, false)
            }
        }

        fun scan(client: TextRecognizer, slot: Int) {
            client.process(InputImage.fromBitmap(frame, 0))
                .addOnSuccessListener { r ->
                    val sb = StringBuilder()
                    for (block in r.textBlocks) {
                        val bb = block.boundingBox ?: continue
                        val cx = bb.centerX().toFloat() / w
                        val cy = bb.centerY().toFloat() / h
                        if (cx in 0.15f..0.85f && cy in 0.08f..0.92f) {
                            if (sb.isNotEmpty()) sb.append(" - ")
                            sb.append(block.text.replace('\n', ' ').trim())
                        }
                    }
                    results[slot] = sb.toString().trim()
                }
                .addOnFailureListener { Log.w(TAG, "readNow slot$slot: ${it.message}") }
                .addOnCompleteListener { finish() }
        }
        scan(textClientLatin, 0)
        scan(textClientDevanagari, 1)
    }

    private fun runFace(bmp: Bitmap, w: Int, h: Int) {
        faceClient.process(InputImage.fromBitmap(bmp, 0))
            .addOnSuccessListener { faces ->
                val frameArea = (w * h).toFloat()
                val facing = faces.any { f ->
                    val b = f.boundingBox
                    val frac = (b.width().toFloat() * b.height().toFloat()) / frameArea
                    kotlin.math.abs(f.headEulerAngleY) <= facingMaxDeg && frac >= minFaceFrac
                }
                val now = System.currentTimeMillis()
                if (facing && now - lastFaceAtMs > faceCooldownMs) {
                    lastFaceAtMs = now
                    onFacingPerson()
                }
            }
            .addOnFailureListener { Log.w(TAG, "face: ${it.message}") }
            .addOnCompleteListener { bmp.recycle(); faceBusy.set(false) }
    }

    fun close() {
        runCatching { textClientLatin.close() }
        runCatching { textClientDevanagari.close() }
        runCatching { faceClient.close() }
    }

    private companion object {
        const val TAG = "SecondSense/mlkit"
        val DEVA_BLOCK: Character.UnicodeBlock = Character.UnicodeBlock.DEVANAGARI
    }
}
