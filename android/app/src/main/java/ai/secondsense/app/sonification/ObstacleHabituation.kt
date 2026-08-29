package ai.secondsense.app.sonification

import kotlin.math.abs

/**
 * Habituation for the continuous obstacle cue.
 *
 * The problem: standing in front of (or beside) a static object — a wall, a parked chair —
 * the cue engine keeps beeping forever even though nothing is changing and the user has
 * clearly registered it. That's fatiguing and drowns out signals that DO matter.
 *
 * The rule: alert normally for [alertHoldMs] after a situation first appears or changes, then
 * fall SILENT for that same obstacle unless the situation *worsens*:
 *   - the user is closing on it (smoothed approach signal, or proximity climbing above the
 *     level it settled at), or
 *   - the target is a different obstacle / the user has turned toward something else.
 * A genuinely imminent obstacle ([imminentProximity]+, i.e. about to be walked into) is never
 * fully silenced — it drops to a faint slow "still here" pulse instead.
 *
 * Only gates the obstacle-spine [CueTarget]. The voice goal cue, the memory-nav cue, and the
 * edge-triggered drop-off / overhead HAPTICS all bypass this and are unaffected.
 *
 * Stateful — one instance per session; [reset] on mode change / pause.
 */
class ObstacleHabituation(
    private val alertHoldMs: Long = 2_500L,
    private val approachResumeThresh: Float = 0.06f,
    private val proximityResumeDelta: Float = 0.06f,
    private val sameAzimuthGate: Float = 0.16f,
    private val imminentProximity: Float = 0.85f,
    private val faintProximity: Float = 0.33f,
    private val approachEmaAlpha: Float = 0.35f,
) {
    /** True when the current static obstacle has been habituated away (for the HUD). */
    var muted: Boolean = false
        private set

    private var lastAzimuth: Float? = null
    private var lastLabel: String? = null
    private var settledProximity: Float = 0f
    private var situationSinceMs: Long = 0L
    private var approachEma: Float = 0f

    fun filter(target: CueTarget?, walking: Boolean, nowMs: Long): CueTarget? {
        if (target == null) {
            reset()
            return null
        }
        approachEma = approachEmaAlpha * target.approaching + (1f - approachEmaAlpha) * approachEma

        val sameSituation = lastAzimuth?.let { az ->
            abs(target.azimuth - az) <= sameAzimuthGate &&
                (lastLabel == target.label || lastLabel == null || target.label == null)
        } ?: false

        if (!sameSituation) {
            // New obstacle, or the user turned toward a different one — alert.
            lastAzimuth = target.azimuth
            lastLabel = target.label
            settledProximity = target.proximity
            situationSinceMs = nowMs
            approachEma = target.approaching
            muted = false
            return target
        }

        lastAzimuth = target.azimuth   // follow slow drift without treating it as new
        lastLabel = target.label

        val worsening = approachEma > approachResumeThresh ||
            target.proximity > settledProximity + proximityResumeDelta ||
            (walking && approachEma > 0.02f)

        if (worsening) {
            settledProximity = maxOf(settledProximity, target.proximity)
            situationSinceMs = nowMs
            muted = false
            return target
        }

        // Stable or receding. Track it moving away so a later re-approach re-triggers cleanly.
        if (target.proximity < settledProximity) settledProximity = target.proximity

        if (nowMs - situationSinceMs < alertHoldMs) return target   // initial alert window

        muted = true
        return if (target.proximity >= imminentProximity) {
            target.copy(proximity = faintProximity)                 // faint "still here" pulse
        } else {
            null                                                    // silence
        }
    }

    fun reset() {
        muted = false
        lastAzimuth = null
        lastLabel = null
        settledProximity = 0f
        situationSinceMs = 0L
        approachEma = 0f
    }
}
