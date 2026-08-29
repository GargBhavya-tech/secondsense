package ai.secondsense.app.perception

import android.util.Log
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import java.util.concurrent.atomic.AtomicBoolean

/**
 * On-device Hindi <-> English translation for OCR'd sign text, so a sign can be spoken in the
 * listener's preferred language regardless of the script it's printed in.
 *
 * OFFLINE after a ONE-TIME model fetch: [prewarm] downloads both direction models over Wi-Fi
 * (~30 MB each) into ML Kit's private store; from then on translation is fully local. Until
 * that finishes (or if it never does — no Wi-Fi at the venue), [localize] transparently falls
 * back to speaking the sign in its native script. Nothing here ever blocks the camera loop:
 * every call is async and the [done] callback fires on the main thread.
 */
class OcrTranslator {

    private val hiToEn: Translator = Translation.getClient(
        TranslatorOptions.Builder()
            .setSourceLanguage(TranslateLanguage.HINDI)
            .setTargetLanguage(TranslateLanguage.ENGLISH)
            .build()
    )
    private val enToHi: Translator = Translation.getClient(
        TranslatorOptions.Builder()
            .setSourceLanguage(TranslateLanguage.ENGLISH)
            .setTargetLanguage(TranslateLanguage.HINDI)
            .build()
    )

    private val hiToEnReady = AtomicBoolean(false)
    private val enToHiReady = AtomicBoolean(false)

    /** Kick off the one-time model downloads (Wi-Fi only). Safe to call more than once. */
    fun prewarm() {
        val conds = DownloadConditions.Builder().requireWifi().build()
        hiToEn.downloadModelIfNeeded(conds)
            .addOnSuccessListener { hiToEnReady.set(true); Log.i(TAG, "hi->en model ready") }
            .addOnFailureListener { Log.w(TAG, "hi->en model unavailable: ${it.message}") }
        enToHi.downloadModelIfNeeded(conds)
            .addOnSuccessListener { enToHiReady.set(true); Log.i(TAG, "en->hi model ready") }
            .addOnFailureListener { Log.w(TAG, "en->hi model unavailable: ${it.message}") }
    }

    /**
     * Produce the string to actually speak for a freshly-read sign, plus the language it's in.
     *
     * @param text              raw OCR output
     * @param sourceIsDevanagari true if it came from the Devanagari recognizer (treated as Hindi)
     * @param wantHindi         the listener's preference (LanguagePrefs.speakHindi)
     * @param translateEnabled  LanguagePrefs.translateSigns
     * @param done              (spokenText, isHindi) — always called exactly once, on the main thread
     */
    fun localize(
        text: String,
        sourceIsDevanagari: Boolean,
        wantHindi: Boolean,
        translateEnabled: Boolean,
        done: (String, Boolean) -> Unit,
    ) {
        // Script already matches the preference, or translation is off -> speak as printed.
        if (sourceIsDevanagari == wantHindi || !translateEnabled) {
            done(text, sourceIsDevanagari)
            return
        }
        val (translator, ready) =
            if (sourceIsDevanagari) hiToEn to hiToEnReady else enToHi to enToHiReady
        if (!ready.get()) {
            // Model pair not on device yet — read the native script rather than stay silent.
            done(text, sourceIsDevanagari)
            return
        }
        translator.translate(text)
            .addOnSuccessListener { out ->
                val clean = out.trim()
                if (clean.isEmpty()) done(text, sourceIsDevanagari) else done(clean, wantHindi)
            }
            .addOnFailureListener {
                Log.w(TAG, "translate failed: ${it.message}")
                done(text, sourceIsDevanagari)
            }
    }

    fun close() {
        runCatching { hiToEn.close() }
        runCatching { enToHi.close() }
    }

    private companion object {
        const val TAG = "SecondSense/translate"
    }
}
