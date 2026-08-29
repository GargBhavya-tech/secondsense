package ai.secondsense.app.voice

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInference.LlmInferenceOptions
import java.io.File

/**
 * Phase 4 — Llama-3.2-1B INT4 on-device via MediaPipe's LLM Inference API.
 *
 * Compiled ONLY with `-PenableLlm=true` (app/build.gradle.kts adds this source dir + the
 * `com.google.mediapipe:tasks-genai` dependency then). [LlmAssistants.create] finds this class
 * by name; without the flag the app has zero MediaPipe footprint and falls back to the stub.
 *
 * The model file is NEVER in the APK (~1 GB). Side-load it:
 *   adb push llama3.2-1b-it-int4.task /data/local/tmp/secondsense-llm/model.task
 * or drop it at <filesDir>/llm/model.task. Missing file => isReady() == false, and the
 * IntentInterpreter grammar keeps handling every common command on its own.
 */
class MediaPipeLlmAssistant(private val appContext: Context) : LlmAssistant {

    @Volatile private var engine: LlmInference? = null
    @Volatile private var triedInit = false

    private fun modelFile(): File? {
        val candidates = listOf(
            File(appContext.filesDir, "llm/model.task"),
            File("/data/local/tmp/secondsense-llm/model.task"),
        )
        return candidates.firstOrNull { it.isFile && it.length() > 1_000_000L }
    }

    @Synchronized
    override fun initialize() {
        if (triedInit) return
        triedInit = true
        val f = modelFile() ?: run {
            Log.i(TAG, "no model file — LLM disabled (grammar-only)")
            return
        }
        engine = runCatching {
            LlmInference.createFromOptions(
                appContext,
                LlmInferenceOptions.builder()
                    .setModelPath(f.absolutePath)
                    .setMaxTokens(512)
                    .setMaxTopK(40)
                    .build(),
            )
        }.onFailure { Log.w(TAG, "LlmInference init failed", it) }.getOrNull()
        Log.i(TAG, "LLM ready=${engine != null} (${f.absolutePath}, ${f.length() / (1024 * 1024)} MB)")
    }

    override fun isReady(): Boolean = engine != null

    override fun resolve(transcript: String, scene: SceneBrief): LlmResolution? {
        val e = engine ?: return null
        val prompt = LlmPrompt.build(transcript, scene)
        val reply = runCatching { e.generateResponse(prompt) }
            .onFailure { Log.w(TAG, "generateResponse failed", it) }
            .getOrNull()
        Log.i(TAG, "q=\"$transcript\" -> \"${reply?.take(160)}\"")
        return LlmPrompt.parse(reply)
    }

    @Synchronized
    override fun close() {
        runCatching { engine?.close() }
        engine = null
    }

    private companion object {
        const val TAG = "SecondSense/llm"
    }
}
