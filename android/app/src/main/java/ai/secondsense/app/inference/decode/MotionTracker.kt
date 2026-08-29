package ai.secondsense.app.inference.decode

import ai.secondsense.app.inference.BBox
import ai.secondsense.app.inference.Detection
import kotlin.math.abs
import kotlin.math.hypot

/**
 * Fills in the COARSE motion signals the pipeline expects — [Detection.moving] (#15's
 * static/dynamic split) and [Detection.approaching] (rate-of-approach, #13) — by matching
 * each detection to the previous frame's detection of the same class.
 *
 * DELIBERATELY COARSE (Bible §11/§13.2): this is a "is this box moving" flag and a signed
 * proximity delta, NOT velocity tracking and NOT Time-to-Collision. Full frame-to-frame
 * TTC physics is a rejected stretch goal. One frame of memory, nearest-center matching
 * within a gate — cheap, and enough to let a walking person outrank a parked chair.
 *
 * Stateful: one instance per engine. Runtime-agnostic — the QNN engine reuses it as-is.
 */
class MotionTracker(
    /** center-distance gate (normalized) for matching a det to last frame's. */
    private val matchGate: Float = 0.12f,
    /** center shift above this (normalized/frame) counts as "moving". */
    private val movingThreshold: Float = 0.02f,
    /** scale factor turning a proximity delta into the reported approaching value. */
    private val approachGain: Float = 4f,
    /**
     * Research candidate (secondsense_research_candidates_v1.md §3, item 5) — scale factor
     * turning ego-motion-COMPENSATED 2D shift into an urgency BOOST added on top of the
     * depth-based approaching value. WHY: a single 2D box-centroid flow vector can't tell us
     * the SIGN of depth change on its own (lateral motion looks the same as no motion in a
     * flat projection) — only depth-delta can say "getting closer" vs "getting farther".
     * What flow magnitude CAN add: fast independent motion near you (a cyclist crossing your
     * path, unpredictable) is itself hazard-relevant even before the depth signal catches up
     * (depth updates only every Nth frame — see depthEveryN in TfliteInferenceEngine — so it
     * lags a fast-moving object more than a slow proximity change). This is deliberately only
     * ever additive on an already-nonnegative depth signal (see [annotate]) — it boosts
     * urgency, it never flips "receding" into "approaching".
     */
    private val flowApproachGain: Float = 3f,
) {
    private data class Prev(val label: String?, val cx: Float, val cy: Float, val proximity: Float)

    private var prev: List<Prev> = emptyList()

    /**
     * Return copies of [dets] with `moving` and `approaching` filled from motion vs the
     * last frame. Call once per frame, in order; it updates internal state.
     *
     * @param egoMotion the estimated CAMERA motion this frame, in the same normalized
     *   (0f..1f-per-frame) units as box centers — from [OpticalFlow.estimateEgoMotion],
     *   converted to normalized units by the caller. Subtracted from each box's raw shift
     *   before comparing to [movingThreshold], so panning your head/chest while walking no
     *   longer reads as "the object is moving" — only genuinely independent motion does.
     *   Defaults to (0,0) (old behavior) so callers that don't compute optical flow still work.
     */
    fun annotate(dets: List<Detection>, egoMotion: Pair<Float, Float> = 0f to 0f): List<Detection> {
        val (egoDx, egoDy) = egoMotion
        val out = dets.map { d ->
            val match = nearestPrev(d.label, d.box)
            if (match == null) {
                d.copy(moving = false, approaching = 0f)
            } else {
                val rawDx = d.box.centerX - match.cx
                val rawDy = d.box.centerY - match.cy
                // ego-motion-compensated shift: what's left after subtracting camera motion.
                val shift = hypot((rawDx - egoDx).toDouble(), (rawDy - egoDy).toDouble()).toFloat()
                val moving = shift > movingThreshold
                val depthApproaching = ((d.proximity - match.proximity) * approachGain).coerceIn(-1f, 1f)
                // Only boost urgency on top of a non-receding depth signal — see
                // flowApproachGain's doc comment for why this never flips the sign.
                val approaching = if (depthApproaching >= 0f) {
                    (depthApproaching + shift * flowApproachGain).coerceIn(-1f, 1f)
                } else {
                    depthApproaching
                }
                d.copy(moving = moving, approaching = approaching)
            }
        }
        prev = out.map { Prev(it.label, it.box.centerX, it.box.centerY, it.proximity) }
        return out
    }

    /** Reset on target loss / mode change so stale motion doesn't leak across a gap. */
    fun reset() { prev = emptyList() }

    private fun nearestPrev(label: String?, box: BBox): Prev? {
        var best: Prev? = null
        var bestD = matchGate
        for (p in prev) {
            if (p.label != label) continue
            val dist = hypot((box.centerX - p.cx).toDouble(), (box.centerY - p.cy).toDouble()).toFloat()
            if (dist < bestD) { bestD = dist; best = p }
        }
        return best
    }
}
