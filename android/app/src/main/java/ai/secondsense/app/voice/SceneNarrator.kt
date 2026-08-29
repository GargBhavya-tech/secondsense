package ai.secondsense.app.voice

import ai.secondsense.app.inference.FrameResult
import ai.secondsense.app.inference.decode.HazardState

/**
 * "What's around me?" — double-tap scene description.
 *
 * Closes the describe-the-scene gap with Seeing AI / Google Lookout, but fully OFFLINE and
 * instant, with NO language model (the Bible lists Llama-3.2-1B as "cut first"). It just
 * assembles one sentence from the current [FrameResult]: the nearest few named objects with
 * a coarse distance + direction, plus any active drop-off. Spoken via on-device TTS.
 */
object SceneNarrator {

    fun describe(result: FrameResult?): String {
        if (result == null) return "Scene not ready"
        val parts = ArrayList<String>()

        when (result.hazardState) {
            HazardState.DROP_CONFIRMED -> parts += "drop-off ahead"
            HazardState.POSSIBLE_DROP -> parts += "possible drop-off ahead"
            HazardState.SCENE_NOT_TRAVERSABLE -> parts += "path blocked ahead"
            else -> {}
        }

        val named = result.detections
            .filter { !it.label.isNullOrBlank() }
            .sortedByDescending { it.proximity }
            .take(3)

        for (d in named) {
            val dist = when {
                d.proximity >= 0.7f -> "very close"
                d.proximity >= 0.45f -> "close"
                d.proximity >= 0.25f -> "a few steps away"
                else -> "far"
            }
            val dir = when {
                d.box.centerX < 0.4f -> "to your left"
                d.box.centerX > 0.6f -> "to your right"
                else -> "straight ahead"
            }
            parts += "${d.label} $dist $dir"
        }

        return if (parts.isEmpty()) "Nothing detected ahead" else parts.joinToString(", ")
    }
}
