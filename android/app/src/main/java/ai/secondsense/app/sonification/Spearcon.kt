package ai.secondsense.app.sonification

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.io.File
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

/**
 * Ticket #20 — Spearcon fallback: a single-word, time-compressed speech token for
 * object classes that don't have a bespoke [AuditoryIcon]. Bible §5.1 keeps spearcons
 * specifically because the auditory-display literature shows short sped-up speech can
 * match or beat icons on raw accuracy/reaction time for less-common classes.
 *
 * HOW IT WORKS
 *   1. Android on-device TTS synthesizes the word to a WAV (offline; no network).
 *   2. We resample it faster (time-compress ~1.8x) into a short PCM buffer — the
 *      "sped-up speech" that makes it a spearcon rather than a spoken word.
 *   3. The buffer is cached per word so we synthesize each class only once.
 *
 * OFFLINE NOTE: on-device TTS voices must be installed for airplane-mode operation
 * (Bible §16 / #31). If a word isn't ready yet, callers fall back to the neutral
 * unknown-tick so the system NEVER goes silent while a spearcon is still baking.
 *
 * This class deliberately does the heavy lifting (synthesis) LAZILY and OFF the audio
 * thread; the cue loop only ever reads a finished buffer.
 */
class Spearcon(context: Context) {

    private val appContext = context.applicationContext
    private val cacheDir = File(appContext.cacheDir, "spearcons").apply { mkdirs() }

    @Volatile private var ready = false
    private var tts: TextToSpeech? = null

    /** word -> compressed PCM (mono, 44.1k, -1f..1f). Ready-to-mix. */
    private val buffers = ConcurrentHashMap<String, FloatArray>()
    /** words currently being synthesized, to avoid double work. */
    private val inFlight = ConcurrentHashMap.newKeySet<String>()

    fun initialize() {
        tts = TextToSpeech(appContext) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                ready = true
            }
        }
    }

    /**
     * Return a ready spearcon buffer for [word], or null if it isn't baked yet.
     * Non-blocking: if null, it kicks off synthesis and the caller uses a fallback
     * this cue; next time the word will be ready.
     */
    fun get(word: String): FloatArray? {
        buffers[word]?.let { return it }
        maybeSynthesize(word)
        return null
    }

    private fun maybeSynthesize(word: String) {
        if (!ready) return
        if (buffers.containsKey(word) || !inFlight.add(word)) return
        val t = tts ?: run { inFlight.remove(word); return }

        val out = File(cacheDir, "$word.wav")
        val id = "spearcon_$word"
        t.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) {
                try {
                    if (utteranceId == id && out.exists()) {
                        val pcm = WavIo.readMonoFloat(out)
                        if (pcm != null) {
                            buffers[word] = TimeStretch.compress(pcm, factor = 1.8f)
                        }
                    }
                } catch (_: Throwable) {
                    // leave it unbaked; caller keeps using the fallback tick
                } finally {
                    inFlight.remove(word)
                }
            }
            @Deprecated("required override") override fun onError(utteranceId: String?) {
                inFlight.remove(word)
            }
        })
        @Suppress("DEPRECATION")
        t.synthesizeToFile(word, null, out, id)
    }

    fun release() {
        tts?.run { stop(); shutdown() }
        tts = null
        ready = false
    }
}
