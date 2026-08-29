package ai.secondsense.app.voice

import android.graphics.Bitmap
import ai.secondsense.app.inference.BBox
import ai.secondsense.app.inference.qnn.QnnBackend

/**
 * Phase 4 — Voice goal-seeking (the co-headline, Bible §7; "they describe, we vector").
 *
 * This file holds the two model seams and a tiny noun extractor. Both models are the ones
 * that have NO usable TFLite export (Whisper-Tiny's tflite is unsupported; OWL-ViT's convert
 * failed), so per the plan the voice path targets QNN specifically — these recognizers wrap a
 * [QnnBackend] and are wired end-to-end now, going live when the native bridge + binaries land.
 * Until then they report notReady and callers degrade honestly (a spoken toast, no fake cue).
 */

/** #26 — turn captured mic audio into a transcript (or a spotted keyword). */
interface SpeechRecognizer {
    /** One-time model load. Safe to call off the main thread; default no-op (QNN stub). */
    fun initialize() {}
    fun isReady(): Boolean
    /** @param pcm mono 16-bit PCM. @return transcript / spotted word, or null if not ready / no speech. */
    fun transcribe(pcm: ShortArray, sampleRate: Int): String?
    /** Release native/model resources. Default no-op. */
    fun release() {}
}

/** #27 — open-vocabulary grounding: locate an arbitrary named thing in a frame. */
interface OpenVocabGrounder {
    fun isReady(): Boolean
    /** @return normalized box of [query] in [frame], or null if not found / not ready. */
    fun ground(frame: Bitmap, query: String): BBox?
}

/**
 * #26 helper — pull the target noun out of a spoken command ("find the door" -> "door").
 * Deliberately dumb: strip filler words, take the last remaining token. Good enough for the
 * short imperative commands the scan/seek mode expects, and trivially testable off-device.
 */
object TargetNoun {
    private val fillers = setOf(
        "find", "the", "a", "an", "me", "to", "go", "get", "where", "is", "are",
        "my", "please", "take", "look", "for", "locate", "show", "nearest", "closest",
    )

    fun extract(text: String?): String? {
        if (text.isNullOrBlank()) return null
        val words = text.lowercase().trim()
            .split(Regex("[^a-z]+"))
            .filter { it.isNotBlank() && it !in fillers }
        return words.lastOrNull()
    }
}

/**
 * Whisper-Tiny speech recognition on the Hexagon NPU (encoder.bin + decoder.bin).
 * The heavy lifting — log-mel features, encoder run, greedy decoder loop — lives in the
 * native bridge; this class orchestrates and stays framework-thin. Stub until the bridge lands.
 */
class WhisperQnnRecognizer(private val backend: QnnBackend) : SpeechRecognizer {
    override fun isReady(): Boolean = backend.isReady()

    override fun transcribe(pcm: ShortArray, sampleRate: Int): String? {
        if (!backend.isReady()) return null
        // TODO(phone): pcm -> 80-bin log-mel -> backend.run("whisper_encoder", mel) ->
        //              greedy decode via backend.run("whisper_decoder", ...) -> detokenized text.
        return null
    }
}

/**
 * OWL-ViT open-vocabulary grounding on the NPU (text tower + image tower). Returns the best
 * box for the spoken query. This is the moat — it works for anything you can say, not a fixed
 * class list. Stub until the native bridge + binaries land.
 */
class OwlVitQnnGrounder(private val backend: QnnBackend) : OpenVocabGrounder {
    override fun isReady(): Boolean = backend.isReady()

    override fun ground(frame: Bitmap, query: String): BBox? {
        if (!backend.isReady()) return null
        // TODO(phone): query -> text embedding via backend.run("owlvit_text", ...);
        //              frame -> image embeddings via backend.run("owlvit_image", ...);
        //              score boxes against the text embedding, return the argmax box.
        return null
    }
}
