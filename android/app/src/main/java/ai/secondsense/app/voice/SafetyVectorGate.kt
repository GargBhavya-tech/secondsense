package ai.secondsense.app.voice

import android.content.Context

/**
 * Tier-2 of the safety gate (deep-research "primary defence"): a multilingual sentence-embedding
 * classifier that decides, PURELY BY MEANING, whether a spoken query is asking about safety /
 * whether it's OK to move — in English, Hindi, or Kannada, in any phrasing. If it fires, the
 * generative LLM is bypassed entirely and MainActivity speaks the deterministic
 * [MainActivity.speakSafety] deflection.
 *
 * The real implementation embeds the transcript with `multilingual-e5-small` (INT8 ONNX, ~39 MB,
 * static seq-64 / batch-1, run on the Hexagon NPU via the QNN EP or ONNX Runtime) and takes the
 * max cosine similarity against the pre-embedded [SafetyAnchors.PHRASES] matrix; a hit above an
 * empirically tuned threshold (~0.82, tuned for 0% false-negatives on explicit safety intents)
 * returns true. It lives in a flag-gated source set like [MediaPipeLlmAssistant] and is resolved
 * here BY NAME; until it's built, [SafetyVectorGates.create] returns [NoopSafetyVectorGate] and
 * the Tier-1 keyword fast-path + LLM deflect-flag + green-light veto carry the load.
 *
 * Caller: MainActivity (query in startVoiceCapture BEFORE IntentInterpreter).
 */
interface SafetyVectorGate {
    fun initialize() {}
    fun isReady(): Boolean
    /** True if [transcript] semantically asks "is it safe / can I move / is the way clear". */
    fun isSafetyQuery(transcript: String): Boolean
    fun close() {}
}

object SafetyVectorGates {
    fun create(context: Context): SafetyVectorGate {
        val impl = runCatching {
            Class.forName("ai.secondsense.app.voice.MultilingualE5SafetyGate")
                .getConstructor(Context::class.java)
                .newInstance(context.applicationContext) as SafetyVectorGate
        }.getOrNull()
        return impl ?: NoopSafetyVectorGate
    }
}

/** Default: no embedding model present. Tier-1 handles safety on its own. */
object NoopSafetyVectorGate : SafetyVectorGate {
    override fun isReady(): Boolean = false
    override fun isSafetyQuery(transcript: String): Boolean = false
}
