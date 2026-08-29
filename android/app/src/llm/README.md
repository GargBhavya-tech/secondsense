# Phase 4 — on-device LLM (optional)

The `IntentInterpreter` grammar (Phase 3) already handles every common spoken command offline.
This adds a small quantized LLM as the **fallback** for anything the grammar can't place
(`VoiceIntent.Unknown`): it either maps the request onto the same closed action set, or answers
a question about the current scene.

Everything here is **off by default**. `gradlew assembleDebug` has zero MediaPipe footprint and
`LlmAssistants.create()` returns `StubLlmAssistant`.

## Enable

1. Build with the flag (one online build to fetch `com.google.mediapipe:tasks-genai`):

   ```
   gradle.bat :app:assembleDebug -PenableLlm=true
   ```

   This compiles `src/llm/kotlin/MediaPipeLlmAssistant.kt` and packages the MediaPipe GenAI
   native libs (arm64-v8a + x86_64). Adds ~120 MB to the APK.

2. Side-load a model `.task` (NOT bundled — ~1 GB). Llama-3.2-1B INT4 recommended:

   ```
   adb shell mkdir -p /data/local/tmp/secondsense-llm
   adb push llama3.2-1b-it-int4.task /data/local/tmp/secondsense-llm/model.task
   ```

   Alternative location: `<filesDir>/llm/model.task`. Either path with a file > 1 MB is picked
   up at `MediaPipeLlmAssistant.initialize()`.

3. Launch. `logcat -s SecondSense/llm` prints `LLM ready=true` once the model loads (a few
   seconds, off the main thread). Without a model file it prints
   `no model file — LLM disabled (grammar-only)` and the app runs exactly as the default build.

## How it's wired

- `voice/LlmAssistant.kt` — interface + `SceneBrief` + `LlmResolution` + `LlmAssistants.create()`
  factory (resolves `MediaPipeLlmAssistant` by name, like `VoiceRecognizers` does for sherpa).
- `voice/LlmPrompt.kt` — pure prompt builder + lenient reply parser (one JSON line →
  `VoiceIntent` / spoken answer). No `org.json` (not mocked in JVM tests); unit-tested in
  `LlmPromptTest`.
- `MainActivity.handleUnknown()` — on `VoiceIntent.Unknown`, if `llm.isReady()` it builds a
  `SceneBrief` from the live `FrameResult` + context + battery and calls `llm.resolve()` on a
  worker thread, then applies the result through the normal `handleVoiceIntent` path.
