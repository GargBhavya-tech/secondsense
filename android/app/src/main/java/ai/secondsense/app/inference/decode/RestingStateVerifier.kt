package ai.secondsense.app.inference.decode

import ai.secondsense.app.inference.Detection
import ai.secondsense.app.inference.SettledSighting

/**
 * Gate that only lets an object into episodic memory once it has *come to rest* — so a bottle
 * swinging in someone's hand, or a person walking past, never gets logged as "where your
 * bottle is". Adapted from the research doc §2C (temporal-spatial variance), image-space +
 * metric-distance instead of full 3D world variance (we have no world frame without SLAM).
 *
 * An object (keyed by label) is SETTLED when, over the last [minFrames] observations:
 *  - its box centre barely moves (variance of normalized cx/cy below [posVarThresh]), and
 *  - its metric distance is changing slower than [maxApproachMps], and
 *  - the window spans at least [minWindowMs].
 *
 * It emits one [SettledSighting] on that transition, then goes quiet for that label until the
 * object leaves view for [forgetMs] (so a re-placed object logs a fresh position, but a
 * stationary one doesn't spam).
 */
class RestingStateVerifier(
    private val minFrames: Int = 8,
    private val posVarThresh: Float = 0.0016f,
    private val maxApproachMps: Float = 0.15f,
    private val minWindowMs: Long = 500L,
    private val minScore: Float = 0.50f,
    private val forgetMs: Long = 1_500L,
    private val requeueAfterMs: Long = 6_000L,
    private val hfovDeg: Float = 60f,
) {
    private class Track {
        val cx = ArrayDeque<Float>()
        val cy = ArrayDeque<Float>()
        val dist = ArrayDeque<Float>()
        val ts = ArrayDeque<Long>()
        var lastSeenMs = 0L
        var mutedUntilMs = 0L
    }

    private val tracks = HashMap<String, Track>()

    /** @param observed (detection, roughMetres) for every NAMED detection this frame. */
    fun update(observed: List<Pair<Detection, Float>>, nowMs: Long): List<SettledSighting> {
        val out = ArrayList<SettledSighting>()

        for ((det, distM) in observed) {
            val label = det.label ?: continue
            if (det.score < minScore) continue
            val tr = tracks.getOrPut(label) { Track() }
            tr.lastSeenMs = nowMs
            tr.cx.addLast(det.box.centerX); tr.cy.addLast(det.box.centerY)
            tr.dist.addLast(distM); tr.ts.addLast(nowMs)
            while (tr.cx.size > minFrames) { tr.cx.removeFirst(); tr.cy.removeFirst(); tr.dist.removeFirst(); tr.ts.removeFirst() }

            if (nowMs < tr.mutedUntilMs) continue
            if (tr.cx.size < minFrames) continue
            val span = tr.ts.last() - tr.ts.first()
            if (span < minWindowMs) continue

            if (variance(tr.cx) > posVarThresh || variance(tr.cy) > posVarThresh) continue
            val dDist = kotlin.math.abs(tr.dist.last() - tr.dist.first())
            val approachMps = dDist / (span / 1000f).coerceAtLeast(0.1f)
            if (approachMps > maxApproachMps) continue

            val meanDist = tr.dist.average().toFloat()
            val meanCx = tr.cx.average().toFloat()
            val bearingDeg = (meanCx - 0.5f) * hfovDeg
            out.add(SettledSighting(label = label, distanceM = meanDist, bearingDeg = bearingDeg))
            tr.mutedUntilMs = nowMs + requeueAfterMs
        }

        // Forget tracks whose object has been gone a while (lets a moved object re-log).
        val it = tracks.entries.iterator()
        while (it.hasNext()) {
            if (nowMs - it.next().value.lastSeenMs > forgetMs) it.remove()
        }
        return out
    }

    fun reset() = tracks.clear()

    private fun variance(xs: ArrayDeque<Float>): Float {
        val m = xs.average().toFloat()
        var s = 0f
        for (x in xs) { val d = x - m; s += d * d }
        return s / xs.size
    }
}
