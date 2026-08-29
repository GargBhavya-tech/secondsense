package ai.secondsense.app.inference.decode

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Problem Statement 3 ("The Specular Trap") — Veto B: is a candidate drop-off edge actually a
 * FLAT reflective/low-texture surface (a puddle, wet marble, polished tile) rather than a
 * physical void?
 *
 * When the wearer walks forward, every point on the real, planar ground moves in the image by
 * an (approximately) affine flow field. A physical hole breaks that — points beyond the edge
 * are on a *lower* plane and move slower. But a puddle / wet floor / glossy marble is
 * physically coplanar with the ground even though the monocular depth net hallucinates a void
 * in it: its surface points move EXACTLY with the surrounding floor.
 *
 * Method (research §3.1, homography residual, done data-driven so it needs no metric scale):
 *  1. Lucas-Kanade track a grid of points in the KNOWN-GOOD ground region (below the horizon,
 *     above the candidate edge).
 *  2. Least-squares fit an affine flow model A to those vectors.
 *  3. Track points in the candidate-edge band; residual = ||observed - A*predicted||.
 *  4. If most band points have LOW residual, the "hole" is moving with the floor -> it is
 *     coplanar -> not a drop. High [coplanarConfidence].
 *
 * Requires ego-motion (no parallax when standing still) and enough trackable texture — both
 * of which fail gracefully to 0f (no veto). Grayscale grid is the small GRAY_W x GRAY_H one
 * SceneAnalyzer already downsamples.
 *
 * Caller: SceneAnalyzer (feeds RawEvidence.groundCoplanar); unit test.
 */
object GroundFlowConsistency {

    private const val MIN_EGO = 0.004f
    private const val MIN_GROUND_PTS = 6
    private const val MIN_BAND_PTS = 4
    private const val INLIER_PX = 1.1f

    fun coplanarConfidence(
        prevGray: FloatArray?,
        curGray: FloatArray,
        w: Int,
        h: Int,
        corridor: TraversableCorridor,
        edgeRowFraction: Float,
        egoMotionXY: Pair<Float, Float>,
    ): Float {
        if (prevGray == null) return 0f
        val egoMag = sqrt(egoMotionXY.first * egoMotionXY.first + egoMotionXY.second * egoMotionXY.second)
        if (egoMag < MIN_EGO) return 0f

        val x0 = (corridor.x1.coerceIn(0f, 1f) * w).toInt().coerceIn(1, w - 2)
        val x1 = (corridor.x2.coerceIn(0f, 1f) * w).toInt().coerceIn(x0 + 1, w - 1)
        val edgeY = edgeRowFraction.coerceIn(0f, 1f) * h
        val gTop = (corridor.y1 + 0.45f * (corridor.y2 - corridor.y1)) * h
        val gBot = edgeY - 0.06f * h
        if (gBot - gTop < 0.08f * h) return 0f
        val bTop = edgeY - 0.02f * h
        val bBot = (edgeY + 0.06f * h).coerceAtMost(h - 2f)

        val gx = ArrayList<Float>(); val gy = ArrayList<Float>()
        val gdx = ArrayList<Float>(); val gdy = ArrayList<Float>()
        var yy = gTop
        val xStep = (x1 - x0) / 8f + 1f
        while (yy < gBot) {
            var xx = x0.toFloat()
            while (xx < x1) {
                OpticalFlow.trackPoint(prevGray, curGray, w, h, xx, yy)?.let { (dx, dy) ->
                    gx.add(xx); gy.add(yy); gdx.add(dx); gdy.add(dy)
                }
                xx += xStep
            }
            yy += (gBot - gTop) / 6f + 1f
        }
        if (gx.size < MIN_GROUND_PTS) return 0f

        val ax = fitAffine(gx, gy, gdx) ?: return 0f
        val ay = fitAffine(gx, gy, gdy) ?: return 0f

        var inliers = 0; var band = 0
        yy = bTop
        while (yy < bBot) {
            var xx = x0.toFloat()
            while (xx < x1) {
                OpticalFlow.trackPoint(prevGray, curGray, w, h, xx, yy)?.let { (dx, dy) ->
                    val px = ax[0] * xx + ax[1] * yy + ax[2]
                    val py = ay[0] * xx + ay[1] * yy + ay[2]
                    val r = sqrt((dx - px) * (dx - px) + (dy - py) * (dy - py))
                    band++
                    if (r < INLIER_PX) inliers++
                }
                xx += xStep
            }
            yy += (bBot - bTop) / 4f + 1f
        }
        if (band < MIN_BAND_PTS) return 0f
        val inlierFrac = inliers.toFloat() / band
        return ((inlierFrac - 0.55f) / 0.35f).coerceIn(0f, 1f)
    }

    /** Least-squares [c0,c1,c2] for  v ~= c0*x + c1*y + c2  via 3x3 normal equations. */
    private fun fitAffine(xs: List<Float>, ys: List<Float>, vs: List<Float>): FloatArray? {
        var sxx = 0.0; var sxy = 0.0; var sx = 0.0; var syy = 0.0; var sy = 0.0
        var sxv = 0.0; var syv = 0.0; var sv = 0.0; val n = xs.size.toDouble()
        for (i in xs.indices) {
            val x = xs[i].toDouble(); val y = ys[i].toDouble(); val v = vs[i].toDouble()
            sxx += x * x; sxy += x * y; sx += x; syy += y * y; sy += y
            sxv += x * v; syv += y * v; sv += v
        }
        val m = arrayOf(
            doubleArrayOf(sxx, sxy, sx, sxv),
            doubleArrayOf(sxy, syy, sy, syv),
            doubleArrayOf(sx, sy, n, sv),
        )
        for (col in 0..2) {
            var piv = col
            for (r in col + 1..2) if (abs(m[r][col]) > abs(m[piv][col])) piv = r
            if (abs(m[piv][col]) < 1e-9) return null
            val t = m[col]; m[col] = m[piv]; m[piv] = t
            for (r in 0..2) {
                if (r == col) continue
                val f = m[r][col] / m[col][col]
                for (k in col..3) m[r][k] -= f * m[col][k]
            }
        }
        return floatArrayOf(
            (m[0][3] / m[0][0]).toFloat(),
            (m[1][3] / m[1][1]).toFloat(),
            (m[2][3] / m[2][2]).toFloat(),
        )
    }
}
