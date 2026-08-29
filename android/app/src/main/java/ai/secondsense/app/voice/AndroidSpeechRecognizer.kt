package ai.secondsense.app.voice

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer as SysSpeechRecognizer
import android.util.Log
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Phase 4 ASR — INTERIM non-QNN path using Android's built-in on-device speech recognition
 * (`SpeechRecognizer.createOnDeviceSpeechRecognizer`, API 33+). No bundled model, no downloads.
 *
 * Trade-off vs the sherpa-onnx path: this depends on the device having an on-device speech
 * service (Google's, present on GMS devices). It works offline once that service's language
 * pack is installed; if the pack is missing, the first use may need network once. The
 * sherpa-onnx recognizer (`SherpaKwsRecognizer`, `-PenableSherpa`) is the guaranteed-offline
 * replacement and drops in behind this same [SpeechRecognizer] interface with no other change.
 *
 * SELF-CAPTURING: the system recognizer runs its own mic session, so [VoiceCommandCapture]
 * skips its own PCM recording for this recognizer (see [selfCaptures]) and the `pcm` arg to
 * [transcribe] is ignored.
 */
class AndroidSpeechRecognizer(context: Context) : SpeechRecognizer {

    private val appContext = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    @Volatile private var available = false

    override fun selfCaptures(): Boolean = true

    override fun isReady(): Boolean = available

    override fun initialize() {
        available = try {
            SysSpeechRecognizer.isRecognitionAvailable(appContext) ||
                (Build.VERSION.SDK_INT >= 33 &&
                    SysSpeechRecognizer.isOnDeviceRecognitionAvailable(appContext))
        } catch (t: Throwable) {
            Log.w(TAG, "availability check failed: ${t.message}"); false
        }
        Log.i(TAG, "ready=$available")
    }

    /**
     * Ignores [pcm] — runs a fresh recognition session and BLOCKS the calling (background)
     * thread until a result, an error, or ~8 s elapse. Returns the top transcript or null.
     */
    override fun transcribe(pcm: ShortArray, sampleRate: Int): String? {
        if (!available) return null
        val latch = CountDownLatch(1)
        val out = AtomicReference<String?>(null)
        val recRef = AtomicReference<SysSpeechRecognizer?>(null)

        main.post {
            val rec = try {
                if (Build.VERSION.SDK_INT >= 33 &&
                    SysSpeechRecognizer.isOnDeviceRecognitionAvailable(appContext)
                ) {
                    SysSpeechRecognizer.createOnDeviceSpeechRecognizer(appContext)
                } else {
                    SysSpeechRecognizer.createSpeechRecognizer(appContext)
                }
            } catch (t: Throwable) {
                Log.w(TAG, "create failed: ${t.message}"); latch.countDown(); return@post
            }
            recRef.set(rec)
            rec.setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle) {
                    out.set(
                        results.getStringArrayList(SysSpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                    )
                    latch.countDown()
                }
                override fun onError(error: Int) {
                    Log.w(TAG, "onError $error"); latch.countDown()
                }
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            }
            try {
                rec.startListening(intent)
            } catch (t: Throwable) {
                Log.w(TAG, "startListening failed: ${t.message}"); latch.countDown()
            }
        }

        val got = try {
            latch.await(8, TimeUnit.SECONDS)
        } catch (_: InterruptedException) {
            false
        }
        main.post { recRef.getAndSet(null)?.let { runCatching { it.destroy() } } }
        return if (got) out.get() else null
    }

    private companion object {
        const val TAG = "SecondSense/voice"
    }
}
