package ai.secondsense.app.sonification

import ai.secondsense.app.inference.ConfidenceTier
import ai.secondsense.app.inference.Detection

/**
 * The single resolved target the [CueEngine] sonifies at any moment.
 *
 * This is the clean handoff between the VISION half (targeting, #15) and the AUDIO
 * half (sonification, #18–#22). Targeting decides WHICH one thing to cue; the cue
 * engine decides HOW it sounds/feels. Keeping this as a tiny value type is what lets
 * each channel be tested in isolation against synthetic CueTargets (the build map's
 * "build each channel standalone against synthetic inputs first").
 *
 * The three orthogonal dimensions (Bible §5.1) map to three fields — and nothing here
 * lets one field drive two channels:
 *   azimuth   -> DIRECTION (pan, #18)
 *   proximity -> DISTANCE  (pulse repetition rate, #19) + haptic intensity (#21)
 *   label     -> IDENTITY  (auditory icon / spearcon, #20)
 * tier shapes TEXTURE only (§5.3), never identity or distance.
 */
data class CueTarget(
    /** 0f = hard left, 0.5f = center, 1f = hard right. */
    val azimuth: Float,
    /** 0f = far, 1f = touching. RELATIVE proximity, never metres. */
    val proximity: Float,
    /** object class, or null = identity unknown (RED tier: proximity-only). */
    val label: String?,
    /** confidence tier — shapes audio TEXTURE, not the other channels. */
    val tier: ConfidenceTier,
    /** +ve = getting closer; reserved for urgency shaping. */
    val approaching: Float = 0f,
) {
    companion object {
        /** Lift a targeting [Detection] (#15 output) into a CueTarget. */
        fun from(d: Detection): CueTarget = CueTarget(
            azimuth = d.box.centerX,
            proximity = d.proximity,
            label = d.label,
            tier = d.tier,
            approaching = d.approaching,
        )
    }
}
