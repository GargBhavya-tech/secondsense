package ai.secondsense.app.sonification

/**
 * Ticket #7 — One-tap calibration.
 *
 * Monocular depth is RELATIVE, not metric (Bible §5.4). Calibration is what gives the
 * relative-proximity ordering a meaningful zero: the user holds the phone level looking
 * forward at normal walking clearance and taps once, and we snapshot the current
 * center-of-frame proximity as the BASELINE. From then on, proximity is re-referenced
 * against that baseline — "closer than my calibrated forward clearance" is what drives
 * urgency, which is exactly the claim the pitch makes (a one-tap baseline, not a false
 * metric range, not full gyroscope auto-correction).
 *
 * OFF BY DEFAULT: with no baseline captured, [apply] is an identity passthrough, so the
 * app behaves exactly as before until the user opts in. Tapping again clears it. Thread-safe
 * for the single writer (UI tap) / single reader (frame loop) pattern.
 */
class Calibration {

    @Volatile private var baseline: Float? = null

    val isCalibrated: Boolean get() = baseline != null
    val baselineValue: Float? get() = baseline

    /** Snapshot the current center proximity as the forward-clearance baseline (#7 tap). */
    fun capture(currentProximity: Float) {
        // Cap well below 1f: apply() divides by (1f - baseline), so a baseline captured with
        // something right against the lens (~0.95+) would make the denominator tiny and slam
        // every later reading to max urgency. 0.85 keeps real headroom above the baseline.
        baseline = currentProximity.coerceIn(0f, 0.85f)
    }

    fun clear() { baseline = null }

    /**
     * Re-reference a raw proximity against the baseline. Objects at exactly the baseline read
     * ~0 urgency; anything nearer than the baseline scales up toward 1. Uncalibrated → raw.
     */
    fun apply(rawProximity: Float): Float {
        val b = baseline ?: return rawProximity
        val denom = (1f - b).coerceAtLeast(1e-3f)
        return ((rawProximity - b) / denom).coerceIn(0f, 1f)
    }
}
