package ai.secondsense.app.voice

import android.content.Context

/**
 * Picks the Phase-4 speech recognizer at runtime.
 *
 *  - `SherpaKwsRecognizer` (offline keyword spotting via sherpa-onnx, no QNN) IF the optional
 *    sherpa module is compiled in (`-PenableSherpa=true`, see app/build.gradle.kts). It is
 *    resolved BY NAME so the default build has no compile-time dependency on the sherpa
 *    wrapper. If its model assets are missing at runtime it still constructs but reports
 *    `isReady() == false`.
 *  - Otherwise [fallback] — the QNN Whisper stub, which reports notReady honestly until the
 *    native bridge exists.
 *
 * Callers only ever see a [SpeechRecognizer]; dropping the real QNN Whisper back in later
 * changes nothing above this line.
 */
object VoiceRecognizers {
    fun create(context: Context, fallback: SpeechRecognizer): SpeechRecognizer = try {
        Class.forName("ai.secondsense.app.voice.SherpaKwsRecognizer")
            .getConstructor(Context::class.java)
            .newInstance(context.applicationContext) as SpeechRecognizer
    } catch (_: Throwable) {
        fallback
    }
}
