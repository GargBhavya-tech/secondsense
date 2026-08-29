package ai.secondsense.app.inference.decode

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.tan
import kotlin.math.PI

/**
 * V3 drop-off plan §4 — classical RGB edge lattice, no training data. Finds several roughly
 * parallel, roughly-horizontal, evenly-spaced lines inside a corridor (candidate stair
 * nosings), and scores how convincingly they form a real periodic lattice vs. random
 * scene clutter (shelf edges, table lines, floor seams).
 *
 * VALIDATED OFFLINE (debug_v3_fusion.py) against the same 4 photos GroundPlaneAnalyzer used —
 * this is the SECOND algorithm tried: a naive per-row gradient-energy version was tried FIRST
 * and demonstrably failed (missed the real target case, stairs2.jpeg, entirely — see the
 * script's git history/comments). This angle-restricted Hough-line version is what actually
 * passed: correctly SAFE on bottle.jpeg (flat floor) and stairs1.jpeg (ascending stairs, the
 * plan's explicit hard-negative), while finding real lattice structure on both stairs.jpg and
 * stairs2.jpeg (the hard descending case with a wrong-signed depth read).
 *
 * WHY NOT OpenCV: this project has zero CV-library dependency by design (GroundPlaneAnalyzer,
 * DropOffDetector, OpticalFlow are all hand-rolled pure-Kotlin/array math) — adding the OpenCV
 * Android SDK for one feature would be a large new native dependency for a narrow angle-range
 * line search that's genuinely cheap to hand-roll instead.
 *
 * DEPENDENCY-FREE DESIGN: since only near-horizontal lines matter here (a stair nosing is
 * never steeply diagonal after rough camera leveling), this restricts the Hough angle search to
 * a narrow band instead of the usual full 180° sweep — far cheaper than general-purpose Hough,
 * parameterized by (angle bin, row-intercept-at-corridor-center bin) rather than the usual
 * (rho, theta) polar form, which is a simpler accumulator for this narrow use case.
 */
object EdgeLattice {

    data class Result(
        /** 0..1 — how convincingly a periodic stair-like lattice was found. */
        val score: Float,
        /** Row fraction (0..1, full-frame) of the nearest (bottom-most) lattice line, or null. */
        val nearestRowFraction: Float?,
        val lineCount: Int,
    )

    // REVISED after a real on-device false positive: a keyboard's rows satisfied the original
    // (±22°, 10% width) thresholds just as well as real stair nosings do — tightened per the
    // user-supplied fix list. Angle range narrowed (a keyboard's rows have more angular spread
    // in practice — off-axis phone angle — than a genuine stair nosing after rough leveling)
    // and the minimum horizontal support raised from 10% to 35% of corridor width (a keyboard's
    // individual key-rows are much narrower runs than a real nosing spanning most of a stair's
    // width).
    private const val ANGLE_MIN_DEG = -12f
    private const val ANGLE_MAX_DEG = 12f
    private const val ANGLE_STEP_DEG = 3f
    private const val MIN_LINES_FOR_LATTICE = 3
    private const val MIN_ROW_SEPARATION_FRAC = 0.02f // merge candidate rows closer than this
    private const val MIN_ROW_SUPPORT_FRAC = 0.35f // was 0.10 — see the revision note above

    /**
     * @param gray full-frame grayscale buffer (e.g. from [OpticalFlow.toGrayscale])
     * @param corridor the IMU-stabilized traversable region (normalized 0..1 rect) to search
     *        inside — see ImuTracker/TraversableCorridor. Evidence outside decays rather than
     *        being hard-cropped, matching the plan's "weighting mask, not a hard crop" note;
     *        here that's approximated by simply not searching outside it (the cheapest correct
     *        version — a soft decay would need per-pixel weighting, not justified yet).
     */
    fun detect(gray: FloatArray, w: Int, h: Int, corridor: TraversableCorridor): Result {
        val x1 = (corridor.x1 * w).toInt().coerceIn(1, w - 2)
        val x2 = (corridor.x2 * w).toInt().coerceIn(x1 + 1, w - 1)
        val y1 = (corridor.y1 * h).toInt().coerceIn(1, h - 2)
        val y2 = (corridor.y2 * h).toInt().coerceIn(y1 + 1, h - 1)
        if (x2 - x1 < 8 || y2 - y1 < 8) return Result(0f, null, 0)

        // Sobel-Y-ish vertical gradient magnitude (edge strength for near-horizontal edges),
        // thresholded to a binary edge map — same spirit as DropOffDetector's Sobel usage.
        val cw = x2 - x1
        val ch = y2 - y1
        val grad = FloatArray(cw * ch)
        var sum = 0.0
        var sumSq = 0.0
        for (yy in 1 until ch - 1) {
            for (xx in 0 until cw) {
                val gx = x1 + xx
                val gyTop = y1 + yy - 1
                val gyBot = y1 + yy + 1
                val g = abs(gray[gyBot * w + gx] - gray[gyTop * w + gx])
                grad[yy * cw + xx] = g
                sum += g
                sumSq += g.toDouble() * g
            }
        }
        val n = (cw * ch).toDouble()
        val mean = sum / n
        val std = kotlin.math.sqrt(max(0.0, sumSq / n - mean * mean))
        val threshold = (mean + 0.75 * std).toFloat()

        // Angle-restricted Hough: for each edge pixel and each candidate angle, vote into a
        // (angle bin, row-intercept-at-corridor-center bin) accumulator. Cheap: narrow angle
        // range, and the accumulator is tiny (a handful of angle bins x corridor height).
        val angleBins = ((ANGLE_MAX_DEG - ANGLE_MIN_DEG) / ANGLE_STEP_DEG).toInt() + 1
        val votes = Array(angleBins) { IntArray(ch) }
        val cx = cw / 2f
        for (yy in 0 until ch) {
            for (xx in 0 until cw) {
                if (grad[yy * cw + xx] <= threshold) continue
                for (ai in 0 until angleBins) {
                    val angleDeg = ANGLE_MIN_DEG + ai * ANGLE_STEP_DEG
                    val slope = tan(angleDeg * PI / 180.0).toFloat()
                    // b = row-intercept at corridor center for a line of this slope through (xx, yy)
                    val b = (yy - slope * (xx - cx)).toInt()
                    if (b in 0 until ch) votes[ai][b]++
                }
            }
        }

        // Collapse the angle dimension: for each row-intercept bin, keep the best angle's vote
        // count (we only care about "is there a strong near-horizontal line through here", not
        // which exact angle).
        val rowVotes = IntArray(ch)
        for (b in 0 until ch) {
            var best = 0
            for (ai in 0 until angleBins) best = max(best, votes[ai][b])
            rowVotes[b] = best
        }

        val voteThreshold = (MIN_ROW_SUPPORT_FRAC * cw).toInt().coerceAtLeast(4)
        val minSepPx = max(1, (MIN_ROW_SEPARATION_FRAC * h).toInt())
        val peaks = ArrayList<Int>()
        var i = 0
        while (i < ch) {
            if (rowVotes[i] > voteThreshold) {
                var bestIdx = i
                var bestVal = rowVotes[i]
                var j = i
                while (j < min(i + minSepPx, ch)) {
                    if (rowVotes[j] > bestVal) { bestVal = rowVotes[j]; bestIdx = j }
                    j++
                }
                peaks += bestIdx
                i = bestIdx + minSepPx
            } else {
                i++
            }
        }

        if (peaks.size < MIN_LINES_FOR_LATTICE) return Result(0f, null, peaks.size)

        val rowsFrac = peaks.map { (y1 + it).toFloat() / h }.sorted()
        val spacings = FloatArray(rowsFrac.size - 1) { rowsFrac[it + 1] - rowsFrac[it] }
        val meanSpacing = spacings.average().toFloat()
        val spacingStd = kotlin.math.sqrt(
            spacings.sumOf { val d = it - meanSpacing; (d * d).toDouble() } / spacings.size
        ).toFloat()
        val spacingConsistency = (1f - min(1f, spacingStd / max(meanSpacing, 1e-3f))).coerceIn(0f, 1f)
        val countSupport = min(1f, peaks.size / 5f)
        val score = (0.75f * spacingConsistency + 0.25f * countSupport).coerceIn(0f, 1f)

        return Result(score, rowsFrac.max(), peaks.size)
    }
}
