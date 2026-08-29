package ai.secondsense.app.inference.decode

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.floor

/**
 * Research-candidate item (secondsense_research_candidates_v1.md §3) — sparse Lucas-Kanade
 * optical flow, used to separate CAMERA motion (panning your head/chest while walking) from
 * OBJECT motion (something actually moving in the scene). Validated offline before writing
 * this (debug_optical_flow.py): tracked a known synthetic 4.0/-2.5 pixel shift and recovered
 * 3.98/-2.50 (median) — matches almost exactly.
 *
 * WHY THIS MATTERS: MotionTracker's `moving` flag previously compared raw box-position shift
 * against a threshold — but a box shifts just as much when YOU turn your head as when the
 * OBJECT moves. That's a real false-positive source this fixes: estimate the background's
 * median flow (a robust proxy for camera ego-motion, since most of a frame is usually static
 * background) via a grid of tracked points, then subtract it from each detection's own
 * tracked flow before deciding if it's genuinely moving.
 *
 * Deliberately single-scale (no image pyramid) — validated to work well for the SMALL
 * frame-to-frame motion expected between two consecutive analyzed camera frames at normal
 * walking pace; a fast whip-pan or large inter-frame jump would need pyramidal LK for
 * robustness, which is real added complexity not justified by this project's actual demo
 * scenario (walking, not running).
 */
object OpticalFlow {

    private const val WINDOW_RADIUS = 7
    private const val MAX_ITERS = 5
    private const val CONVERGE_EPS = 0.01f

    /** Downsample a Bitmap to a small grayscale float array (0..255) for cheap flow tracking. */
    fun toGrayscale(bitmap: Bitmap, targetW: Int, targetH: Int): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true)
        val out = FloatArray(targetW * targetH)
        val pixels = IntArray(targetW * targetH)
        scaled.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH)
        if (scaled !== bitmap) scaled.recycle()
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            out[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }
        return out
    }

    /**
     * Track ONE point via single-scale iterative Lucas-Kanade.
     * @return (dx, dy) flow in pixels, or null if the window is untrackable (out of bounds,
     *         or too little texture — an ill-conditioned gradient system).
     */
    fun trackPoint(prevGray: FloatArray, curGray: FloatArray, w: Int, h: Int, x0: Float, y0: Float): Pair<Float, Float>? {
        val xi = x0.toInt()
        val yi = y0.toInt()
        val x1 = (xi - WINDOW_RADIUS).coerceAtLeast(1)
        val x2 = (xi + WINDOW_RADIUS + 1).coerceAtMost(w - 1)
        val y1 = (yi - WINDOW_RADIUS).coerceAtLeast(1)
        val y2 = (yi + WINDOW_RADIUS + 1).coerceAtMost(h - 1)
        if (x2 - x1 < 3 || y2 - y1 < 3) return null

        fun at(g: FloatArray, x: Int, y: Int) = g[y * w + x]
        fun gx(x: Int, y: Int) = (at(prevGray, x + 1, y) - at(prevGray, x - 1, y)) / 2f
        fun gy(x: Int, y: Int) = (at(prevGray, x, y + 1) - at(prevGray, x, y - 1)) / 2f

        // Build the 2x2 gradient system ONCE (it doesn't change across iterations).
        var sxx = 0.0; var sxy = 0.0; var syy = 0.0
        for (y in y1 until y2) for (x in x1 until x2) {
            val ix = gx(x, y); val iy = gy(x, y)
            sxx += ix * ix; sxy += ix * iy; syy += iy * iy
        }
        val det = sxx * syy - sxy * sxy
        if (abs(det) < 1e-3) return null // flat/textureless window, unreliable

        var u = 0f
        var v = 0f
        iterations@ for (iter in 0 until MAX_ITERS) {
            var sxt = 0.0
            var syt = 0.0
            for (y in y1 until y2) {
                for (x in x1 until x2) {
                    val warped = bilinear(curGray, w, h, x + u, y + v) ?: break@iterations
                    val it = warped - at(prevGray, x, y)
                    val ix = gx(x, y); val iy = gy(x, y)
                    sxt += ix * it; syt += iy * it
                }
            }
            val du = ((syy * -sxt) - (sxy * -syt)) / det
            val dv = ((sxx * -syt) - (sxy * -sxt)) / det
            u += du.toFloat(); v += dv.toFloat()
            if (abs(du) < CONVERGE_EPS && abs(dv) < CONVERGE_EPS) break@iterations
        }
        return u to v
    }

    /** Inlier radius (px, at the flow-tracking resolution) for the RANSAC ego-motion consensus below. */
    private const val EGO_MOTION_INLIER_PX = 1.5f

    /**
     * RANSAC-style consensus flow over a background grid, replacing a plain per-axis median.
     *
     * WHY NOT MEDIAN: a per-axis median can produce a "phantom" vector that doesn't match ANY
     * actually-tracked point — if roughly half the grid points land on a large object moving
     * one way and half land on background moving another way, the x-median and y-median can
     * each get pulled toward a DIFFERENT one of those two clusters independently, yielding a
     * combined (medianX, medianY) that corresponds to neither the object's motion nor the
     * camera's. Confirmed as a real limitation (research candidate, secondsense_research_
     * candidates_v1.md §3): "median breaks if a large object dominates the frame."
     *
     * WHAT THIS DOES INSTEAD: exhaustively try each tracked point's OWN flow as a hypothesis
     * (cheap — the grid is small, ~9-16 points, so this is O(n^2) not a performance concern),
     * count how many other points agree with it within [EGO_MOTION_INLIER_PX], and return the
     * MEAN of the largest such consensus cluster. This picks an actual, physically coherent
     * cluster of agreeing points rather than an independently-computed-per-axis blend.
     *
     * HONEST LIMITATION (unchanged from median, and inherent to any majority-vote approach):
     * if the moving object covers MORE than half the background grid points (very close, very
     * large in frame), the largest consensus cluster could legitimately be the object's motion,
     * not the camera's. Fixing that needs a genuinely different signal (e.g. depth-aware point
     * selection, weighting distant/background-likely points higher) — out of scope here.
     */
    fun estimateEgoMotion(prevGray: FloatArray, curGray: FloatArray, w: Int, h: Int): Pair<Float, Float> {
        val margin = WINDOW_RADIUS + 2
        val stepX = ((w - 2 * margin) / 3).coerceAtLeast(1)
        val stepY = ((h - 2 * margin) / 3).coerceAtLeast(1)
        val flows = mutableListOf<Pair<Float, Float>>()
        var y = margin
        while (y < h - margin) {
            var x = margin
            while (x < w - margin) {
                trackPoint(prevGray, curGray, w, h, x.toFloat(), y.toFloat())?.let { flows += it }
                x += stepX
            }
            y += stepY
        }
        if (flows.isEmpty()) return 0f to 0f
        if (flows.size == 1) return flows[0]

        var bestInliers: List<Pair<Float, Float>> = listOf(flows[0])
        for (hyp in flows) {
            val inliers = flows.filter { (u, v) ->
                val du = u - hyp.first; val dv = v - hyp.second
                (du * du + dv * dv) <= EGO_MOTION_INLIER_PX * EGO_MOTION_INLIER_PX
            }
            if (inliers.size > bestInliers.size) bestInliers = inliers
        }
        val meanU = bestInliers.sumOf { it.first.toDouble() }.toFloat() / bestInliers.size
        val meanV = bestInliers.sumOf { it.second.toDouble() }.toFloat() / bestInliers.size
        return meanU to meanV
    }

    private fun bilinear(g: FloatArray, w: Int, h: Int, x: Float, y: Float): Float? {
        val x0 = floor(x).toInt(); val y0 = floor(y).toInt()
        val x1 = x0 + 1; val y1 = y0 + 1
        if (x0 < 0 || y0 < 0 || x1 >= w || y1 >= h) return null
        val fx = x - x0; val fy = y - y0
        val v00 = g[y0 * w + x0]; val v01 = g[y0 * w + x1]
        val v10 = g[y1 * w + x0]; val v11 = g[y1 * w + x1]
        return v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy + v11 * fx * fy
    }
}
