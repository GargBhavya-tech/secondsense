package ai.secondsense.app.voice

import ai.secondsense.app.inference.BBox
import ai.secondsense.app.inference.ConfidenceTier
import ai.secondsense.app.sonification.CueTarget
import kotlin.math.abs

/**
 * Ticket #28 — Vector-to-Goal cueing. The single clearest differentiator: *they describe, we
 * vector*. Once a spoken goal is grounded to a box each frame (#27), this turns that box into
 * a continuous steering [CueTarget] — pan toward it (direction), pulse/haptics as you close in
 * (distance) — reusing the exact same cue engine as obstacle avoidance, and signals ARRIVAL
 * when the goal is centered and close.
 *
 * Pure logic over a grounded box, so it's testable off-device and identical on any backend.
 * The only Phase-4-specific dependency (the OWL-ViT grounder) lives upstream; this just steers.
 */
class VectorToGoalController(
    /** how centered the goal must be to count as "reached" (|cx-0.5| <= this). */
    private val arriveCenter: Float = 0.12f,
    /** how close (relative proximity) the goal must be to count as "reached". */
    private val arriveProximity: Float = 0.75f,
) {
    @Volatile private var goalWord: String? = null

    val activeGoal: String? get() = goalWord
    val isActive: Boolean get() = goalWord != null

    /** Set (or clear, with null) the current spoken goal. */
    fun setGoal(word: String?) { goalWord = word }

    /**
     * Turn this frame's grounded [box] (+ its relative [proximity]) into a steering cue toward
     * the goal, or null if there's no active goal or the goal isn't visible this frame.
     */
    fun cueFor(box: BBox?, proximity: Float): CueTarget? {
        val g = goalWord ?: return null
        if (box == null) return null
        return CueTarget(
            azimuth = box.centerX,      // DIRECTION — pan toward the goal
            proximity = proximity,      // DISTANCE — pulse rate + haptics as you approach
            label = g,                  // IDENTITY — the named goal (icon/spearcon)
            tier = ConfidenceTier.WHITE,
        )
    }

    /** True once the goal is both centered and close — fire the "arrived" confirmation. */
    fun hasArrived(box: BBox?, proximity: Float): Boolean {
        if (goalWord == null || box == null) return false
        return abs(box.centerX - 0.5f) <= arriveCenter && proximity >= arriveProximity
    }
}
