package ai.secondsense.app.inference.decode

/**
 * Ticket #17 — Drop-off / negative-obstacle detection.
 *
 * A white cane finds the ground; the camera's job is to find the ABSENCE of ground — the
 * case the cane's structural blind spot and a plain "object in the way" detector both miss
 * (Bible §8). Curbs, downward stairs, unguarded platform edges and potholes are life-safety
 * critical in Indian streets, and nothing else in the pipeline answers the downward case.
 *
 * V2 — REPLACES the original fixed-band version after live-phone testing on a real staircase
 * showed the fixed-band comparison (hardcoded near/mid-ground zones) simply never fires
 * whenever the framing/geometry doesn't match its assumed camera-close-to-ground setup — it
 * missed a real staircase entirely (validated against a real photo: band diff was -0.09,
 * nowhere near the fire threshold). This version is adaptive instead of fixed-geometry:
 *
 *   1. Scan the lower half of the depth map for the SHARPEST vertical gradient (Sobel-style
 *      central difference) in the center column — wherever it actually is, not a fixed band.
 *   2. Confirm it's a real drop-off and not just any object boundary (a desk edge, a chair
 *      back) by checking the SIGN of the proximity change right around that row: proximity
 *      must read farther just past the edge than just before it — the actual physical
 *      signature of "the ground disappeared here." An object boundary usually reads the
 *      OPPOSITE way (something nearer sits behind it, e.g. floor visible under a desk).
 *      Validated offline: a desk-edge photo showed local_diff=-0.084 (correctly rejected)
 *      while the real staircase showed local_diff=+0.058 (correctly fires).
 *
 * Both checks must pass. This is deliberately still conservative — a flat floor has no sharp
 * gradient at all (step 1 alone rejects it) — but no longer BLIND to hazards outside a
 * hardcoded zone. No fabricated metres (Bible §5.4): [Edge.rowFraction] is frame-relative
 * position only. Runtime-agnostic: consumes a [DepthSampler.Frame] directly (no [DepthSampler]
 * instance needed), so the TFLite engine and the future QNN engine share it unchanged.
 */
class DropOffDetector(
    /** half-width of the sampled center column (0.16 => center ~32%). */
    private val centerHalfWidth: Float = 0.16f,
    /** minimum gradient magnitude to count as a real discontinuity, not noise. */
    private val edgeStrengthThreshold: Float = 0.10f,
    /** how much farther proximity must read just past the edge vs just before it. */
    private val localDropDelta: Float = 0.05f,
) {

    /** Where the discontinuity is, and how sharp it reads. */
    data class Edge(
        /** 0f (top of frame) .. 1f (bottom) — how far down the discontinuity sits. */
        val rowFraction: Float,
        /** relative gradient magnitude at that row (unitless; for threshold/urgency use only). */
        val strength: Float,
    )

    /**
     * @return the located drop-off edge if the frame has one, else null (also null when the
     *         depth map itself is invalid).
     */
    fun detect(frame: DepthSampler.Frame): Edge? {
        if (!frame.valid) return null
        val w = frame.w
        val h = frame.h
        if (w < 3 || h < 3) return null
        val range = (frame.hi - frame.lo).takeIf { it > 1e-6f } ?: return null
        val x1 = ((0.5f - centerHalfWidth) * w).toInt().coerceIn(0, w - 1)
        val x2 = ((0.5f + centerHalfWidth) * w).toInt().coerceIn(x1, w - 1)
        val yStart = (h * 0.5f).toInt().coerceIn(1, h - 2) // lower half only — a floor/edge search zone

        // Step 1: find the sharpest vertical depth gradient — a TRUE 3x3 Sobel Gy kernel
        // (weights [-1,-2,-1 / 0,0,0 / 1,2,1]), not a plain row-1/row+1 difference. That
        // simpler difference under-weights by ~4x vs the real kernel and was validated to
        // MISS real staircases entirely — this exact weighted form is what was calibrated
        // against real photos (offline, via debug_dropoff_v2.py) to land the 0.10 threshold.
        fun norm(y: Int, x: Int) = (frame.map[y * w + x] - frame.lo) / range
        var bestRow = -1
        var bestStrength = 0f
        for (y in yStart until h - 1) {
            var sum = 0f
            for (x in x1..x2) {
                val xm1 = (x - 1).coerceIn(0, w - 1)
                val xp1 = (x + 1).coerceIn(0, w - 1)
                val gy = (-norm(y - 1, xm1) - 2f * norm(y - 1, x) - norm(y - 1, xp1)) +
                    (norm(y + 1, xm1) + 2f * norm(y + 1, x) + norm(y + 1, xp1))
                sum += kotlin.math.abs(gy)
            }
            val strength = sum / (x2 - x1 + 1)
            if (strength > bestStrength) {
                bestStrength = strength
                bestRow = y
            }
        }
        if (bestRow < 0 || bestStrength < edgeStrengthThreshold) return null

        // Step 2: confirm the SIGN — proximity must read farther just past the edge than just
        // before it (the ground disappearing), not nearer (an ordinary object boundary).
        val bandPx = maxOf(6, h / 40)
        fun avgProx(yRange: IntRange): Float {
            var sum = 0f
            var n = 0
            for (y in yRange) for (x in x1..x2) {
                sum += (frame.map[y * w + x] - frame.lo) / range
                n++
            }
            return if (n > 0) sum / n else 0.5f
        }
        val aboveProx = avgProx(maxOf(0, bestRow - bandPx) until bestRow)
        val belowProx = avgProx(bestRow until minOf(h, bestRow + bandPx))
        val localDiff = aboveProx - belowProx
        if (localDiff < localDropDelta) return null

        return Edge(rowFraction = bestRow.toFloat() / h, strength = bestStrength)
    }
}
