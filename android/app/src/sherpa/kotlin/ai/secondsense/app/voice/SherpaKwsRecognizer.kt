package ai.secondsense.app.voice

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.KeywordSpotter
import com.k2fsa.sherpa.onnx.KeywordSpotterConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig

/**
 * Phase 4 — the ASR half, non-QNN path. Offline keyword spotting via sherpa-onnx
 * (next-gen Kaldi + onnxruntime), replacing [WhisperQnnRecognizer] (Whisper has no usable
 * TFLite export; the Hexagon NPU is access-gated on this device).
 *
 * COMPILED ONLY WITH `-PenableSherpa=true` — this whole source tree (src/sherpa/kotlin) is
 * excluded from the default build (see app/build.gradle.kts), so the app builds green
 * without the sherpa wrapper/native lib. [VoiceRecognizers.create] resolves this class by
 * name and falls back to the QNN stub when it isn't present.
 *
 * REQUIRES (see android/app/src/sherpa/README.md for exact download URLs):
 *   1. sherpa-onnx Kotlin wrapper sources -> src/sherpa/kotlin/com/k2fsa/sherpa/onnx/*.kt
 *   2. libsherpa-onnx-jni.so (+ its onnxruntime deps) per ABI -> src/main/jniLibs/<abi>/
 *   3. a streaming KWS transducer model -> src/main/assets/kws/ containing an encoder /
 *      decoder / joiner .onnx (matched by substring below), tokens.txt, and keywords.txt.
 *      Reference model: sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 (English, ~15 MB).
 *
 * SCOPE: closed-vocabulary. keywords.txt should list the object-class nouns GoalGrounding can
 * resolve against yolo26s (chair, person, bottle, cup, laptop, ...). The spotted keyword IS
 * the target noun — TargetNoun.extract still runs downstream and is a harmless passthrough
 * for a single word.
 *
 * VERSION NOTE: sherpa-onnx's Kotlin config field names have shifted slightly across
 * releases. If compilation fails, the likely spots are the `KeywordSpotterConfig` /
 * `OnlineModelConfig` / `OnlineTransducerModelConfig` constructor argument names — check them
 * against the wrapper .kt files you copied in.
 */
class SherpaKwsRecognizer(private val context: Context) : SpeechRecognizer {

    private var spotter: KeywordSpotter? = null

    override fun isReady(): Boolean = spotter != null

    override fun initialize() {
        if (spotter != null) return
        try {
            val am = context.assets
            val files = am.list(ASSET_DIR)?.toList().orEmpty()
            fun pick(part: String): String? =
                files.firstOrNull { it.endsWith(".onnx") && it.contains(part) }?.let { "$ASSET_DIR/$it" }

            val encoder = pick("encoder")
            val decoder = pick("decoder")
            val joiner = pick("joiner")
            val tokens = if (files.contains("tokens.txt")) "$ASSET_DIR/tokens.txt" else null
            val keywords = if (files.contains(KEYWORDS_FILE)) "$ASSET_DIR/$KEYWORDS_FILE" else null
            if (encoder == null || decoder == null || joiner == null || tokens == null || keywords == null) {
                Log.w(TAG, "KWS assets incomplete in assets/$ASSET_DIR (need encoder/decoder/joiner .onnx + tokens.txt + $KEYWORDS_FILE); staying not-ready")
                return
            }

            val config = KeywordSpotterConfig(
                featConfig = FeatureConfig(sampleRate = 16000, featureDim = 80),
                modelConfig = OnlineModelConfig(
                    transducer = OnlineTransducerModelConfig(
                        encoder = encoder,
                        decoder = decoder,
                        joiner = joiner,
                    ),
                    tokens = tokens,
                    numThreads = 1,
                    provider = "cpu",
                ),
                keywordsFile = keywords,
            )
            spotter = KeywordSpotter(assetManager = am, config = config)
            Log.i(TAG, "ready (encoder=$encoder)")
        } catch (t: Throwable) {
            Log.w(TAG, "initialize failed: ${t.message}")
            spotter = null
        }
    }

    /**
     * Feed one captured utterance (from [VoiceCommandCapture]) through the spotter in a single
     * pass and return the first keyword that fired, or null.
     */
    override fun transcribe(pcm: ShortArray, sampleRate: Int): String? {
        val s = spotter ?: return null
        return try {
            val samples = FloatArray(pcm.size) { pcm[it] / 32768f }
            val stream = s.createStream()
            stream.acceptWaveform(samples, sampleRate)
            // flush: sherpa expects a trailing tail of silence to end the segment
            stream.acceptWaveform(FloatArray(sampleRate / 2), sampleRate)
            var spotted: String? = null
            while (s.isReady(stream)) {
                s.decode(stream)
                val kw = s.getResult(stream).keyword
                if (kw.isNotBlank()) {
                    spotted = kw.trim()
                    s.reset(stream)
                    break
                }
            }
            stream.release()
            spotted
        } catch (t: Throwable) {
            Log.w(TAG, "transcribe failed: ${t.message}")
            null
        }
    }

    override fun release() {
        spotter?.release()
        spotter = null
    }

    private companion object {
        const val TAG = "SecondSense/sherpa"
        const val ASSET_DIR = "kws"
        const val KEYWORDS_FILE = "keywords.txt"
    }
}
