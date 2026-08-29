package ai.secondsense.app.sonification

import ai.secondsense.app.inference.ConfidenceTier

/**
 * Ticket #24 — Graceful-Degradation Ladder. Formalizes §5.3 / §13.3.
 *
 * #23 gave us WHITE/BLUE/RED as a signal-quality judgement. #24 turns that (plus
 * proximity and depth availability) into an explicit, TOTAL rung decision for the cue
 * engine — "total" meaning every possible input lands on exactly one defined rung, so
 * there is never a dead end and never silence-by-omission.
 *
 * The three rungs (Bible §13.3):
 *   FULL     — identity (icon) + direction (pan) + distance (pulse rate). WHITE tier.
 *   PROXIMITY— drop identity, keep proximity-pulse + uncertainty texture. BLUE/RED tier.
 *   PANIC    — haptic-only threshold when something is very close (< the panic proximity),
 *              regardless of what audio can say. This is the floor: it can fire UNDER any
 *              audio state, including when there's no usable audio cue at all.
 *
 * KEY DESIGN POINT — PANIC is not "the bottom of a chain you fall through"; it's an
 * always-checked FLOOR. A close object at RED tier is both PROXIMITY (for audio: bare
 * pulse) AND PANIC (for haptics: hard buzz). The ladder therefore returns the audio rung
 * separately from a panic flag, so the two never cancel each other. That's what "no dead
 * end — each rung degrades to the next automatically" means in practice: the system can
 * always still buzz you away from a wall even when it has nothing intelligent to say.
 */
enum class LadderRung {
    /** Full three-channel cue. */
    FULL,
    /** Proximity pulse + uncertainty texture, no identity claim. */
    PROXIMITY,
    /** No usable audio cue at all this frame (e.g. no depth AND no class). */
    SILENT_AUDIO,
}

/**
 * The resolved degradation decision for one frame.
 * @param audioRung which audio rung to render (FULL / PROXIMITY / SILENT_AUDIO).
 * @param panic     if true, fire the haptic panic buzz IN ADDITION, independent of audio.
 */
data class LadderDecision(
    val audioRung: LadderRung,
    val panic: Boolean,
)

object DegradationLadder {

    /**
     * Panic threshold (Bible §13.3 rung 3: "<0.5m"). Proximity is RELATIVE (0..1), not
     * metres, so this is a relative-proximity threshold calibrated against the #7 baseline
     * — i.e. "close enough that we buzz you regardless of whether we can name it."
     */
    const val PANIC_PROXIMITY = 0.80f

    /**
     * Decide the rung for this frame. TOTAL over all inputs.
     *
     * @param tier            the smoothed confidence tier (#23).
     * @param proximity       relative proximity 0..1 (or null if no depth this frame).
     * @param depthAvailable  did depth return anything this frame?
     * @param hasLabel        did a class label survive to this point?
     */
    fun decide(
        tier: ConfidenceTier,
        proximity: Float?,
        depthAvailable: Boolean,
        hasLabel: Boolean,
    ): LadderDecision {
        val prox = proximity ?: -1f
        val panic = depthAvailable && prox >= PANIC_PROXIMITY

        val audioRung = when {
            // Rung 1: full cue only when we're confident AND have both identity + depth.
            tier == ConfidenceTier.WHITE && depthAvailable && hasLabel -> LadderRung.FULL
            // Rung 2: any real depth signal -> proximity pulse + texture, no identity claim.
            depthAvailable -> LadderRung.PROXIMITY
            // Nothing audible to say this frame. Haptic panic may STILL fire above.
            else -> LadderRung.SILENT_AUDIO
        }
        return LadderDecision(audioRung = audioRung, panic = panic)
    }
}
