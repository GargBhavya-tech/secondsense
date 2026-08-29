package ai.secondsense.app.voice

import android.content.Context

/**
 * Phase 4 — the on-device reasoning fallback. When the deterministic [IntentInterpreter] grammar
 * can't place a transcript ([VoiceIntent.Unknown]), it comes here: a small quantized LLM
 * (Llama-3.2-1B INT4 via MediaPipe's LLM Inference API) that either
 *  - maps the free-form request onto the SAME closed action set ([LlmResolution.Action]), or
 *  - answers a question about what the app currently sees ([LlmResolution.Speak]).
 *
 * The model file (~1 GB `.task`) is NEVER bundled — it is side-loaded to the device and the
 * MediaPipe-backed implementation lives in a flag-gated source set (`-PenableLlm=true`, see
 * app/build.gradle.kts), resolved BY NAME here exactly like the sherpa recognizer. With no
 * model / no flag, [LlmAssistants.create] returns [StubLlmAssistant] and callers degrade
 * honestly — the grammar still handles every common command, the LLM just isn't there for the
 * long tail.
 *
 * Caller: MainActivity (construct once, query on the Unknown branch off the main thread).
 */
interface LlmAssistant {
    /** One-time model load. Heavy (seconds) — call off the main thread. Safe no-op on the stub. */
    fun initialize() {}

    fun isReady(): Boolean

    /**
     * @param transcript the raw thing the user said that the grammar couldn't place.
     * @param scene      a snapshot of what the app perceives right now, for grounded answers.
     * @return a resolved action / spoken answer, or null if not ready or the model declined.
     *         Blocking; call off the main thread.
     */
    fun resolve(transcript: String, scene: SceneBrief): LlmResolution?

    fun close() {}
}

/**
 * A compact, spoken-language-ready snapshot of the live pipeline state, handed to the LLM as
 * grounding context. Deliberately strings, not model objects — this is what goes into a prompt.
 */
data class SceneBrief(
    val context: String,                 // activity context, e.g. "walking"
    val objectsAhead: List<String>,       // e.g. ["person ahead", "door to the left"]
    val hazard: String?,                  // e.g. "possible drop-off ahead", or null
    val batteryPct: Int,                  // 0..100, -1 unknown
    val camera: String,                   // "ok" / "dim" / "blocked" / "angle"
    val lastSpoken: String?,              // the last thing the app told the user
)

/** What the LLM decided to do with an otherwise-unhandled request. */
sealed interface LlmResolution {
    /** The request maps onto the existing closed action set — run it through the normal handler. */
    data class Action(val intent: VoiceIntent) : LlmResolution
    /** A one-off spoken reply (scene Q&A: "is there a chair nearby?" -> "Yes, to your right."). */
    data class Speak(val text: String) : LlmResolution
    /**
     * The model flagged this as a safety / movement question (or was told to deflect). The
     * caller MUST ignore any model text and speak its own deterministic, hazard-grounded
     * deflection instead. Layer 4 of the safety gate.
     */
    object Defer : LlmResolution
}

object LlmAssistants {
    /**
     * [MediaPipeLlmAssistant] if the flag-gated class compiled in AND finds a model file;
     * otherwise [StubLlmAssistant]. Never throws.
     */
    fun create(context: Context): LlmAssistant {
        val impl = runCatching {
            Class.forName("ai.secondsense.app.voice.MediaPipeLlmAssistant")
                .getConstructor(Context::class.java)
                .newInstance(context.applicationContext) as LlmAssistant
        }.getOrNull()
        return impl ?: StubLlmAssistant
    }
}

/** Default: the LLM isn't present. Every call is a safe no-op. */
object StubLlmAssistant : LlmAssistant {
    override fun isReady(): Boolean = false
    override fun resolve(transcript: String, scene: SceneBrief): LlmResolution? = null
}
