package ai.secondsense.app.audio

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.concurrent.thread
import kotlin.math.exp

/**
 * Ticket #33 — continuous hazard-sound detection (car horns, sirens, alarms) via YamNet.
 * A DIFFERENT sensing modality entirely from vision — catches hazards outside the camera's
 * field of view (a car approaching from behind, a siren around a corner), which nothing else
 * in the pipeline can. Real, cloud-profiled export: 0.2ms NPU inference, 521-class AudioSet
 * output, PSNR 65.97dB on-device-vs-reference (excellent). See MelSpectrogram.kt for the
 * offline-validated preprocessing this depends on.
 *
 * Runs on ~1-second non-overlapping audio chunks — matches YAMNetParams.PATCH_HOP_SECONDS=1.0
 * in the reference implementation ("for our inference, don't need overlapping windows").
 *
 * YamNet's raw output is LOGITS, not probabilities (confirmed offline: real outputs ranged
 * roughly -9..+5) — sigmoid is applied here before any thresholding, matching
 * YAMNetParams.CLASSIFIER_ACTIVATION = 'sigmoid' from the model's own published spec.
 *
 * HONEST LIMITATION: the preprocessing math and the model itself are validated (offline test
 * correctly classified "Speech" then "Whistling" in a real reference recording). The specific
 * HAZARD_KEYWORDS matching and scoreThreshold have NOT been validated against a real
 * horn/siren/alarm recording — no such sample was available this session. Treat the
 * threshold as a reasonable starting point, not a tuned value.
 */
class HazardSoundDetector(
    private val context: Context,
    private val modelAsset: String = "models/yamnet.tflite",
    private val labelsAsset: String = "models/yamnet_labels.txt",
    private val scoreThreshold: Float = 0.3f,
) {
    companion object {
        private const val TAG = "SecondSense/hazard"
        // Curated from YamNet's 521 AudioSet classes (see the exported labels.txt for the
        // full list). Matched by substring so minor label wording differences across model
        // versions don't silently break detection.
        private val HAZARD_KEYWORDS = listOf(
            "vehicle horn", "car horn", "honking", "car alarm", "air horn", "truck horn",
            "emergency vehicle", "police car (siren)", "ambulance (siren)",
            "fire engine, fire truck (siren)", "train horn", "civil defense siren",
            "smoke detector, smoke alarm", "fire alarm", "explosion", "gunshot, gunfire",
        )
        // Ticket #34 — voice auto-ducking. Matched the same way as HAZARD_KEYWORDS: scan the
        // FULL 521-class score array (not just the single top-1 class), so speech ducks the
        // cue volume even on a frame where something else scored marginally higher overall.
        private val SPEECH_KEYWORDS = listOf(
            "speech", "conversation", "narration, monologue", "child speech", "shout",
            "whispering",
        )
        private const val SPEECH_SCORE_THRESHOLD = 0.3f
    }

    data class Hazard(val label: String, val score: Float)

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var audioRecord: AudioRecord? = null
    @Volatile private var running = false
    private var captureThread: Thread? = null

    val isReady: Boolean get() = interpreter != null

    fun initialize() {
        try {
            interpreter = Interpreter(loadModel(modelAsset), Interpreter.Options().apply { setNumThreads(2) })
            labels = context.assets.open(labelsAsset).bufferedReader().readLines()
            Log.i(TAG, "ready. ${labels.size} labels loaded")
        } catch (t: Throwable) {
            Log.w(TAG, "initialize failed: ${t.message}")
        }
    }

    /**
     * Starts continuous background capture + classification. No-op if already running or not
     * initialized. [onSample] fires every ~1s with the single top class (any class, hazard or
     * not) for HUD/debug visibility; [onHazard] fires only for HAZARD_KEYWORDS matches above
     * [scoreThreshold].
     */
    fun start(
        onHazard: (Hazard) -> Unit,
        onSample: ((topLabel: String, topScore: Float) -> Unit)? = null,
        onSpeechChanged: ((Boolean) -> Unit)? = null,
    ) {
        val itp = interpreter ?: return
        if (running) return
        val minBuf = AudioRecord.getMinBufferSize(
            MelSpectrogram.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(MelSpectrogram.SAMPLE_RATE) // at least 1s of headroom
        val record = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC, MelSpectrogram.SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, minBuf,
            )
        } catch (t: Throwable) {
            Log.w(TAG, "AudioRecord init failed (mic permission missing?): ${t.message}")
            return
        }
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            Log.w(TAG, "AudioRecord failed to initialize")
            return
        }
        audioRecord = record
        running = true
        record.startRecording()

        captureThread = thread(name = "hazard-sound-capture") {
            val chunkSamples = MelSpectrogram.SAMPLE_RATE // 1.0s non-overlapping chunks
            val shortBuf = ShortArray(chunkSamples)
            while (running) {
                var filled = 0
                while (running && filled < chunkSamples) {
                    val n = record.read(shortBuf, filled, chunkSamples - filled)
                    if (n > 0) filled += n else break
                }
                if (!running || filled < chunkSamples) continue
                val pcm = FloatArray(chunkSamples) { shortBuf[it] / 32768f }
                try {
                    val patches = MelSpectrogram.toLogMelPatches(pcm)
                    for (patch in patches) {
                        val inputBuf = ByteBuffer.allocateDirect(patch.size * 4).order(ByteOrder.nativeOrder())
                        patch.forEach { inputBuf.putFloat(it) }
                        inputBuf.rewind()
                        val outputBuf = ByteBuffer.allocateDirect(labels.size * 4).order(ByteOrder.nativeOrder())
                        itp.run(inputBuf, outputBuf)
                        outputBuf.rewind()
                        // sigmoid: raw output is logits, not probabilities (see class doc).
                        val scores = FloatArray(labels.size) { 1f / (1f + exp(-outputBuf.float)) }
                        val topIdx = scores.indices.maxByOrNull { scores[it] }
                        if (topIdx != null) onSample?.invoke(labels.getOrElse(topIdx) { "?" }, scores[topIdx])
                        checkHazards(scores).forEach(onHazard)
                        onSpeechChanged?.invoke(isSpeechPresent(scores))
                    }
                } catch (t: Throwable) {
                    Log.w(TAG, "classify failed: ${t.message}")
                }
            }
        }
    }

    private fun isSpeechPresent(scores: FloatArray): Boolean {
        for (i in scores.indices) {
            if (scores[i] < SPEECH_SCORE_THRESHOLD) continue
            val label = labels.getOrNull(i)?.lowercase() ?: continue
            if (SPEECH_KEYWORDS.any { label.contains(it) }) return true
        }
        return false
    }

    private fun checkHazards(scores: FloatArray): List<Hazard> {
        val found = mutableListOf<Hazard>()
        for (i in scores.indices) {
            if (scores[i] < scoreThreshold) continue
            val label = labels.getOrNull(i) ?: continue
            val lower = label.lowercase()
            if (HAZARD_KEYWORDS.any { lower.contains(it) }) {
                found += Hazard(label, scores[i])
            }
        }
        return found
    }

    fun stop() {
        running = false
        captureThread?.join(500)
        captureThread = null
        audioRecord?.let { runCatching { it.stop() }; runCatching { it.release() } }
        audioRecord = null
    }

    fun close() {
        stop()
        interpreter?.close()
        interpreter = null
    }

    private fun loadModel(asset: String): ByteBuffer {
        context.assets.openFd(asset).use { fd ->
            fd.createInputStream().channel.use { ch ->
                return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }
}
