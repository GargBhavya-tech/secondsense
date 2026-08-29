package ai.secondsense.app.inference.decode

import kotlin.math.tan

/**
 * Turns Depth-Anything-V2's affine-invariant relative depth into a ROUGH metric distance, by
 * anchoring the unknown scale/shift to the one thing whose geometry we know: the floor.
 *
 * Method (from the SecondSense episodic-memory research doc, §2A):
 *  1. For floor pixels inside the traversable corridor, the true camera-frame depth is fixed
 *     by ray/ground-plane intersection: Z_geom = -H_mount / (n_cam · r), where n_cam is the
 *     ground normal from IMU pitch/roll and r = K⁻¹[u,v,1].
 *  2. Depth-Anything emits d_raw ∝ 1/Z up to an unknown affine (s, t): 1/Z = s·d_raw + t.
 *  3. Least-squares fit (s, t) over those floor samples, then apply to any pixel.
 *
 * HONESTY: this is an ESTIMATE with wide error bars, NOT a range-finder. Depends on an
 * assumed camera FOV ([assumedHfovDeg] — uncalibrated), a fixed mount height, and the floor
 * actually being visible and flat. The project's core invariant (depth is RELATIVE, Bible
 * §5.4) is untouched — `Detection.proximity` stays relative; this metres value is used ONLY
 * by the object-memory feature, and its spoken output is bucketed ("a few steps"), never
 * read out as a precise number. When the floor fit fails, [metersForBox] degrades to a
 * monotonic guess from relative proximity.
 */
class MetricDepthScaler(
    private val mountHeightM: Float = 1.30f,
    private val assumedHfovDeg: Float = 60f,
    private val minFloorSamples: Int = 14,
    /** frames a good fit stays trusted before it's considered stale. */
    private val fitTtlFrames: Int = 30,
) {
    private var haveFit = false
    private var s = 0f
    private var t = 0f
    private var fitAgeFrames = Int.MAX_VALUE

    /** Recompute the affine fit from this frame's floor, if enough floor is visible. */
    fun updateFloorFit(
        frame: DepthSampler.Frame,
        corridor: TraversableCorridor,
        pitchDeg: Float,
        rollDeg: Float,
    ) {
        fitAgeFrames++
        if (!frame.valid) return

        val w = frame.w
        val h = frame.h
        val fx = (w / 2f) / tan(assumedHfovDeg / 2f * DEG2RAD)
        val fy = fx // square letterboxed input -> ~equal focal lengths
        val cx = w / 2f
        val cy = h / 2f

        val pr = pitchDeg * DEG2RAD
        val rr = rollDeg * DEG2RAD
        // Ground normal in a +Y-down camera frame (level look -> [0,-1,0]).
        val nx = Math.sin(rr.toDouble()).toFloat()
        val ny = (-Math.sin(pr.toDouble()) * Math.cos(rr.toDouble())).toFloat()
        val nz = (-Math.cos(pr.toDouble()) * Math.cos(rr.toDouble())).toFloat()

        // Sample the lower part of the corridor — most likely to be actual floor.
        val yLo = ((maxOf(corridor.y1, 0.60f)) * h).toInt().coerceIn(0, h - 1)
        val yHi = (corridor.y2 * h).toInt().coerceIn(0, h - 1)
        val xLo = (corridor.x1 * w).toInt().coerceIn(0, w - 1)
        val xHi = (corridor.x2 * w).toInt().coerceIn(0, w - 1)

        var sxx = 0.0; var sx = 0.0; var sxy = 0.0; var sy = 0.0; var n = 0
        var y = yLo
        while (y < yHi) {
            val row = y * w
            var x = xLo
            while (x < xHi) {
                val dRaw = frame.map[row + x]
                if (dRaw > 1e-4f) {
                    val rx = (x - cx) / fx
                    val ry = (y - cy) / fy
                    val denom = nx * rx + ny * ry + nz
                    if (denom < -1e-3f) { // ray actually meets the floor below
                        val zGeom = -mountHeightM / denom
                        if (zGeom in 0.3f..8f) {
                            val invZ = 1.0 / zGeom
                            sxx += dRaw.toDouble() * dRaw
                            sx += dRaw.toDouble()
                            sxy += dRaw.toDouble() * invZ
                            sy += invZ
                            n++
                        }
                    }
                }
                x += 3
            }
            y += 3
        }

        if (n < minFloorSamples) return
        val det = sxx * n - sx * sx
        if (kotlin.math.abs(det) < 1e-9) return
        val sFit = ((sxy * n - sx * sy) / det).toFloat()
        val tFit = ((sxx * sy - sx * sxy) / det).toFloat()
        if (sFit <= 0f) return // Depth-Anything: larger d_raw must map to larger 1/Z (nearer)

        s = sFit; t = tFit; haveFit = true; fitAgeFrames = 0
    }

    /** Rough metric distance (m) to the object in [box]; always returns something usable. */
    fun metersForBox(frame: DepthSampler.Frame, box: ai.secondsense.app.inference.BBox): Float {
        val rawMed = medianRawIn(frame, box)
        if (haveFit && fitAgeFrames <= fitTtlFrames && rawMed != null) {
            val invZ = s * rawMed + t
            if (invZ > 1e-3f) return (1f / invZ).coerceIn(0.3f, 10f)
        }
        // Fallback: monotonic guess from this-frame relative proximity.
        val prox = if (rawMed != null && frame.valid) {
            ((rawMed - frame.lo) / (frame.hi - frame.lo)).coerceIn(0f, 1f)
        } else 0.4f
        return (0.55f / prox.coerceAtLeast(0.07f)).coerceIn(0.3f, 9f)
    }

    val hasMetricFit: Boolean get() = haveFit && fitAgeFrames <= fitTtlFrames

    fun reset() {
        haveFit = false; s = 0f; t = 0f; fitAgeFrames = Int.MAX_VALUE
    }

    private fun medianRawIn(frame: DepthSampler.Frame, box: ai.secondsense.app.inference.BBox): Float? {
        if (!frame.valid) return null
        val w = frame.w; val h = frame.h
        val hx = (box.right - box.left) * 0.3f
        val hy = (box.bottom - box.top) * 0.3f
        val x1 = ((box.centerX - hx) * w).toInt().coerceIn(0, w - 1)
        val x2 = ((box.centerX + hx) * w).toInt().coerceIn(0, w - 1)
        val y1 = ((box.centerY - hy) * h).toInt().coerceIn(0, h - 1)
        val y2 = ((box.centerY + hy) * h).toInt().coerceIn(0, h - 1)
        val vals = ArrayList<Float>((x2 - x1 + 1) * (y2 - y1 + 1))
        for (yy in y1..y2) {
            val row = yy * w
            for (xx in x1..x2) vals.add(frame.map[row + xx])
        }
        if (vals.isEmpty()) return null
        vals.sort()
        return vals[vals.size / 2]
    }

    private companion object {
        const val DEG2RAD = 0.017453292f
    }
}
