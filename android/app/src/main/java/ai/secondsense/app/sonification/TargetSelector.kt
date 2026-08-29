package ai.secondsense.app.sonification

import ai.secondsense.app.inference.Detection
import ai.secondsense.app.inference.FrameResult
import kotlin.math.abs

/**
 * Tickets #14 + #15 — turn a frame's detection list into the single target to cue.
 *
 * This is the "#15 output" that CueEngine (#22) integrates against. Built here (rather
 * than waiting on the live YOLO loop #12) because it's pure logic over [Detection]s and
 * runs identically on mock or real data — so the whole sonification spine is testable now.
 *
 * Rules, in order:
 *   #14 CENTER-CROP: ignore anything outside the center band. Flow mode stays sparse.
 *   #15 CLOSEST-IN-CENTER: among centered targets, the nearest (highest proximity) wins.
 *   #15 STATIC/DYNAMIC: a MOVING target beats a STATIC one at ~equal proximity — an
 *       approaching person outranks a parked chair at the same range. A coarse flag,
 *       not velocity tracking (Bible §13.2).
 */
class TargetSelector(
    /** half-width of the center band, 0.15 => center 30% (Bible #14). */
    private val centerHalfWidth: Float = 0.15f,
    /** proximity tie window within which the moving/static flag decides (#15). */
    private val equalProximityWindow: Float = 0.12f,
) {

    fun select(frame: FrameResult): CueTarget? =
        selectDetection(frame.detections)?.let { CueTarget.from(it) }

    /**
     * Selection + tier derivation (#23). Runs [classifier] on the chosen detection's
     * RAW signal (score + depth + whether a class came back) and stamps the SMOOTHED
     * tier onto the returned CueTarget — overriding any tier the upstream engine guessed.
     *
     * This is the path the app uses in production: the tier is a DERIVED signal, so it
     * behaves identically once QnnInferenceEngine replaces the mock. On target loss the
     * classifier is reset so a new target starts fresh (no stale hysteresis carryover).
     *
     * @param depthAvailable frame-level depth presence (FrameResult.depthAvailable).
     */
    fun selectWithTier(
        frame: FrameResult,
        classifier: TierClassifier,
    ): CueTarget? {
        val d = selectDetection(frame.detections)
        if (d == null) {
            classifier.reset()
            return null
        }
        val tier = classifier.classify(
            score = d.score,
            depthAvailable = frame.depthAvailable,
            hasClassLabel = d.label != null,
        )
        // On RED the identity claim is dropped, honestly (§5.3): null the label.
        val honestLabel = if (tier == ai.secondsense.app.inference.ConfidenceTier.RED) null else d.label
        return CueTarget.from(d).copy(tier = tier, label = honestLabel)
    }

    /** Exposed for unit-style testing against synthetic detection lists. */
    fun selectDetection(detections: List<Detection>): Detection? {
        val centered = detections.filter { isCentered(it.box.centerX) }
        if (centered.isEmpty()) return null

        // Base ordering: closest first.
        val sorted = centered.sortedByDescending { it.proximity }
        var best = sorted.first()

        // Static/dynamic override: if a moving target is within the equal-proximity
        // window of the current best and best is static, the moving one wins.
        for (cand in sorted) {
            if (cand === best) continue
            val within = abs(cand.proximity - best.proximity) <= equalProximityWindow
            if (within && cand.moving && !best.moving) {
                best = cand
            }
        }
        return best
    }

    private fun isCentered(centerX: Float): Boolean =
        abs(centerX - 0.5f) <= centerHalfWidth
}
