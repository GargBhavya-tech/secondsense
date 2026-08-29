package ai.secondsense.app.inference.decode

import kotlin.math.abs

/**
 * Research candidate (secondsense_research_candidates_v2.md §8) — V-disparity + RANSAC
 * ground-plane fitting, a SECOND, independent drop-off signal alongside [DropOffDetector]'s
 * Sobel-gradient approach (never a replacement — see below).
 *
 * VALIDATED OFFLINE (debug_vdisparity.py) against all 4 real test photos used this session,
 * including the one case [DropOffDetector] V2 is KNOWN to miss:
 *   - bottle1.jpeg (no hazard): max |deviation| 0.031 — correctly silent
 *   - stairs1.jpeg (ascending stairs, not our hazard class): max |deviation| 0.022 — silent
 *   - stairs.jpg (real hazard, V2 already catches this): max |deviation| 0.234 — flagged
 *   - stairs2.jpeg (real hazard, V2 MISSES — the open bug from earlier this session):
 *     max |deviation| 0.118, growing monotonically toward the bottom of frame — FLAGGED.
 * This is the one case in the whole session where an offline test caught something the
 * shipped detector genuinely cannot.
 *
 * METHOD: fit a line to the "V-disparity profile" (median proximity per row) over a TRUSTED
 * reference band (55%-85% of frame height), then compare the actual profile near the feet
 * (90%-100%) against what that line predicts. A real flat/gently-sloped floor extrapolates
 * predictably; a drop-off (or, per the stairs.jpg case, ANY structural break between the
 * reference band and the near field) does not — flagged on ABSOLUTE deviation, not just
 * "farther than predicted," because a real geometry case (viewing a staircase from across a
 * room, not walking toward it) showed the opposite sign but was equally anomalous.
 *
 * WHY THIS DOESN'T REPLACE DropOffDetector: only 4 real photos were used to validate this —
 * strong evidence, not exhaustive proof. Both detectors independently triggering the same
 * haptic (dropOff fires if EITHER fires — see the engine wiring) is safer than swapping one
 * validated-but-imperfect detector for another.
 */
class GroundPlaneAnalyzer(
    /** half-width of the sampled center column. */
    private val centerHalfWidth: Float = 0.15f,
    private val fitBandTop: Float = 0.55f,
    private val fitBandBottom: Float = 0.85f,
    private val checkBandTop: Float = 0.90f,
    /** flag if |actual - predicted| exceeds this, at the WORST (max-deviation) row checked. */
    private val deviationThreshold: Float = 0.08f,
    private val ransacIters: Int = 60,
    private val ransacInlierTolerance: Float = 0.04f,
) {
    /** Where the worst deviation from the fitted ground plane is, and how strong it reads. */
    data class Edge(val rowFraction: Float, val deviation: Float)

    fun detect(frame: DepthSampler.Frame): Edge? {
        if (!frame.valid) return null
        val w = frame.w
        val h = frame.h
        val range = (frame.hi - frame.lo).takeIf { it > 1e-6f } ?: return null
        val x1 = ((0.5f - centerHalfWidth) * w).toInt().coerceIn(0, w - 1)
        val x2 = ((0.5f + centerHalfWidth) * w).toInt().coerceIn(x1, w - 1)

        val fitY1 = (h * fitBandTop).toInt().coerceIn(0, h - 1)
        val fitY2 = (h * fitBandBottom).toInt().coerceIn(fitY1 + 1, h)
        if (fitY2 - fitY1 < 4) return null

        // V-disparity profile: median normalized proximity per row, over the fit band only.
        val rows = ArrayList<Float>(fitY2 - fitY1)
        val proxs = ArrayList<Float>(fitY2 - fitY1)
        val rowBuf = FloatArray(x2 - x1 + 1)
        for (y in fitY1 until fitY2) {
            for (x in x1..x2) rowBuf[x - x1] = (frame.map[y * w + x] - frame.lo) / range
            rowBuf.sort()
            rows += y.toFloat() / h
            proxs += rowBuf[rowBuf.size / 2] // median
        }

        val (a, b) = ransacLineFit(rows, proxs) ?: return null

        val checkY1 = (h * checkBandTop).toInt().coerceIn(0, h - 1)
        var worstRow = -1
        var worstDeviation = 0f
        for (y in checkY1 until h) {
            for (x in x1..x2) rowBuf[x - x1] = (frame.map[y * w + x] - frame.lo) / range
            rowBuf.sort()
            val actual = rowBuf[rowBuf.size / 2]
            val predicted = a * (y.toFloat() / h) + b
            val deviation = actual - predicted
            if (abs(deviation) > abs(worstDeviation)) {
                worstDeviation = deviation
                worstRow = y
            }
        }
        if (worstRow < 0 || abs(worstDeviation) < deviationThreshold) return null
        return Edge(rowFraction = worstRow.toFloat() / h, deviation = worstDeviation)
    }

    /**
     * V3 drop-off plan §5 — depth as an EVIDENCE channel, not a veto. Same RANSAC ground-plane
     * fit as [detect], but samples multiple columns across the corridor and requires their
     * sign to AGREE before trusting the result at all — VALIDATED OFFLINE (debug_v3_fusion.py)
     * to be more discriminating than the single-center-strip [detect]: on stairs2.jpeg (the
     * hard case with a wrong-signed depth read in the original V2 detector), this multi-column
     * agreement check actually recovers a genuine SUPPORTS verdict where a naive single-strip
     * read was ambiguous — sampling several columns and requiring agreement turns out to
     * average out exactly the kind of localized sign noise that broke the single-strip version.
     *
     * @return SUPPORTS/CONTRADICTS when several sampled columns agree AND the deviation is
     *         large enough to trust; UNRELIABLE when the depth map is too locally flat to
     *         mean anything, or when columns disagree (itself a sign the read isn't trustworthy).
     */
    fun depthEvidence(frame: DepthSampler.Frame, corridor: TraversableCorridor): Pair<DepthVerdict, Float> {
        if (!frame.valid) return DepthVerdict.UNRELIABLE to 0f
        val w = frame.w
        val h = frame.h
        val range = (frame.hi - frame.lo).takeIf { it > 1e-6f } ?: return DepthVerdict.UNRELIABLE to 0f

        val cx1 = (corridor.x1 * w).toInt().coerceIn(0, w - 1)
        val cx2 = (corridor.x2 * w).toInt().coerceIn(cx1, w - 1)
        val fitY1 = (h * corridor.y1).toInt().coerceIn(0, h - 1)
        val fitY2 = (h * 0.85f).toInt().coerceIn(fitY1 + 1, h)
        if (fitY2 - fitY1 < 4 || cx2 - cx1 < 4) return DepthVerdict.UNRELIABLE to 0f

        val centerX1 = (cx1 + (cx2 - cx1) * 0.35f).toInt()
        val centerX2 = (cx1 + (cx2 - cx1) * 0.65f).toInt().coerceAtLeast(centerX1 + 1)
        val rows = ArrayList<Float>(fitY2 - fitY1)
        val proxs = ArrayList<Float>(fitY2 - fitY1)
        val rowBuf = FloatArray(centerX2 - centerX1 + 1)
        for (y in fitY1 until fitY2) {
            for (x in centerX1..centerX2) rowBuf[x - centerX1] = (frame.map[y * w + x] - frame.lo) / range
            rowBuf.sort()
            rows += y.toFloat() / h
            proxs += rowBuf[rowBuf.size / 2]
        }
        if (proxs.distinct().size < 3) return DepthVerdict.UNRELIABLE to 0f // locally flat map

        val (a, b) = ransacLineFit(rows, proxs) ?: return DepthVerdict.UNRELIABLE to 0f

        val checkY1 = (h * 0.90f).toInt().coerceIn(0, h - 1)
        var worstDeviation = 0f
        val colSigns = ArrayList<Float>()
        val numSamples = 5
        for (i in 0 until numSamples) {
            val x0 = (cx1 + (cx2 - cx1) * i / (numSamples - 1).coerceAtLeast(1)).coerceIn(cx1, cx2)
            var colWorst = 0f
            var colSum = 0f
            var colN = 0
            for (y in checkY1 until h) {
                val actual = (frame.map[y * w + x0] - frame.lo) / range
                val predicted = a * (y.toFloat() / h) + b
                val deviation = predicted - actual
                colSum += deviation
                colN++
                if (kotlin.math.abs(deviation) > kotlin.math.abs(colWorst)) colWorst = deviation
            }
            if (colN > 0) {
                colSigns += kotlin.math.sign(colSum / colN)
                if (kotlin.math.abs(colWorst) > kotlin.math.abs(worstDeviation)) worstDeviation = colWorst
            }
        }
        if (colSigns.isEmpty()) return DepthVerdict.UNRELIABLE to 0f

        val agreement = kotlin.math.abs(colSigns.average().toFloat())
        if (kotlin.math.abs(worstDeviation) < 0.06f) return DepthVerdict.UNRELIABLE to worstDeviation
        if (agreement < 0.5f) return DepthVerdict.UNRELIABLE to worstDeviation // columns disagree -> don't trust either way
        return (if (worstDeviation > 0) DepthVerdict.SUPPORTS else DepthVerdict.CONTRADICTS) to worstDeviation
    }

    /** Fit proximity = a*rowFrac + b via RANSAC (2-point sampling, inlier-count selection, least-squares refit on inliers). */
    private fun ransacLineFit(xs: List<Float>, ys: List<Float>): Pair<Float, Float>? {
        val n = xs.size
        if (n < 4) return null
        var bestInlierIdx: List<Int> = emptyList()
        val rnd = java.util.Random(0) // deterministic — matches the offline validation's fixed seed
        repeat(ransacIters) {
            val i1 = rnd.nextInt(n)
            var i2 = rnd.nextInt(n)
            if (i2 == i1) i2 = (i2 + 1) % n
            val x1 = xs[i1]; val x2 = xs[i2]
            if (x1 == x2) return@repeat
            val a = (ys[i2] - ys[i1]) / (x2 - x1)
            val b = ys[i1] - a * x1
            val inliers = ArrayList<Int>()
            for (i in 0 until n) {
                val pred = a * xs[i] + b
                if (abs(pred - ys[i]) <= ransacInlierTolerance) inliers += i
            }
            if (inliers.size > bestInlierIdx.size) bestInlierIdx = inliers
        }
        if (bestInlierIdx.size < 2) return null
        // least-squares refit on the inlier set
        var sumX = 0.0; var sumY = 0.0; var sumXY = 0.0; var sumXX = 0.0
        for (i in bestInlierIdx) {
            val x = xs[i].toDouble(); val y = ys[i].toDouble()
            sumX += x; sumY += y; sumXY += x * y; sumXX += x * x
        }
        val cnt = bestInlierIdx.size
        val denom = cnt * sumXX - sumX * sumX
        if (abs(denom) < 1e-9) return null
        val a = ((cnt * sumXY - sumX * sumY) / denom).toFloat()
        val b = ((sumY - a * sumX) / cnt).toFloat()
        return a to b
    }
}
