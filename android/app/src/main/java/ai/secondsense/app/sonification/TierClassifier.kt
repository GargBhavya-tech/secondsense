package ai.secondsense.app.sonification

import ai.secondsense.app.inference.ConfidenceTier

/**
 * Ticket #23 — the self-trust / uncertainty layer.
 *
 * Turns raw detector signal into a WHITE/BLUE/RED confidence tier (Bible §5.3). This
 * is deliberately SEPARATE from where the mock happens to set a tier: the mock scripts
 * tiers for demo variety, but the REAL system must DERIVE the tier from the actual
 * signal — detection score + whether depth is present — so it behaves identically when
 * QnnInferenceEngine replaces the mock. That's what this class does.
 *
 * The tiers, and the honest claim each makes (§5.3):
 *   WHITE — "I see it and I know what it is."   score high, depth present.
 *   BLUE  — "Something's there, I'm not sure what." score low-but-real, depth present.
 *   RED   — "Something's there, I can't name it."   no reliable class, depth only.
 *
 * The system SOUNDS unsure on BLUE and ADMITS it doesn't know on RED — it never goes
 * silent and never fakes crisp confidence. "The device knows when it doesn't know" is
 * the defensible stage claim; this class is where that claim is actually true.
 *
 * HYSTERESIS: raw per-frame scores jitter around the thresholds, which would make the
 * audio texture stutter between clean and grainy every few frames — audibly worse than
 * either. So promotions (toward WHITE) require clearing a slightly HIGHER bar than
 * demotions, and a tier must persist a couple of frames before it's allowed to change.
 * This mirrors the temporal-smoothing intent of #16, applied to the tier itself.
 */
class TierClassifier(
    /** at/above this score with depth -> eligible for WHITE. */
    private val whiteEnter: Float = 0.62f,
    /** must drop below this to leave WHITE (hysteresis gap). */
    private val whiteExit: Float = 0.52f,
    /** at/above this score with depth -> at least BLUE (real, but unsure). */
    private val blueEnter: Float = 0.32f,
    /** must drop below this to leave BLUE toward RED. */
    private val blueExit: Float = 0.24f,
    /** frames a *new* tier must persist before it's accepted. */
    private val stabilityFrames: Int = 2,
) {
    private var currentTier: ConfidenceTier = ConfidenceTier.RED
    private var pendingTier: ConfidenceTier? = null
    private var pendingCount = 0

    /**
     * @param score          raw detector confidence 0f..1f (ignored for identity if it lands RED).
     * @param depthAvailable is a proximity reading present this frame?
     * @param hasClassLabel  did the detector return a usable class name?
     * @return the smoothed tier to use for this cue.
     */
    fun classify(score: Float, depthAvailable: Boolean = true, hasClassLabel: Boolean = true): ConfidenceTier {
        val raw = rawTier(score, depthAvailable, hasClassLabel)
        return smooth(raw)
    }

    /** The instantaneous tier before hysteresis, using directional thresholds. */
    private fun rawTier(score: Float, depth: Boolean, hasLabel: Boolean): ConfidenceTier {
        // No depth at all = nothing trustworthy to say. Treated as RED (proximity-only
        // is itself unavailable, but the cue engine will simply have nothing to pulse).
        if (!depth) return ConfidenceTier.RED
        // Depth present but no class -> honest RED (proximity-only, no identity claim).
        if (!hasLabel) return ConfidenceTier.RED

        // Direction-aware thresholds (hysteresis): the bar to ENTER a higher tier is
        // higher than the bar to STAY, so we don't oscillate at the boundary.
        return when (currentTier) {
            ConfidenceTier.WHITE -> when {
                score >= whiteExit -> ConfidenceTier.WHITE
                score >= blueExit -> ConfidenceTier.BLUE
                else -> ConfidenceTier.RED
            }
            ConfidenceTier.BLUE -> when {
                score >= whiteEnter -> ConfidenceTier.WHITE
                score >= blueExit -> ConfidenceTier.BLUE
                else -> ConfidenceTier.RED
            }
            ConfidenceTier.RED -> when {
                score >= whiteEnter -> ConfidenceTier.WHITE
                score >= blueEnter -> ConfidenceTier.BLUE
                else -> ConfidenceTier.RED
            }
        }
    }

    /** Require a new tier to persist [stabilityFrames] before committing to it. */
    private fun smooth(raw: ConfidenceTier): ConfidenceTier {
        if (raw == currentTier) {
            pendingTier = null
            pendingCount = 0
            return currentTier
        }
        if (raw == pendingTier) {
            pendingCount++
            if (pendingCount >= stabilityFrames) {
                currentTier = raw
                pendingTier = null
                pendingCount = 0
            }
        } else {
            pendingTier = raw
            pendingCount = 1
        }
        return currentTier
    }

    /** Reset on target loss so a new target starts fresh (called when cue target is null). */
    fun reset() {
        currentTier = ConfidenceTier.RED
        pendingTier = null
        pendingCount = 0
    }

    val tier: ConfidenceTier get() = currentTier
}
