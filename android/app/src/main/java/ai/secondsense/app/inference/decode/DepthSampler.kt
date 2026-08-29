package ai.secondsense.app.inference.decode

import ai.secondsense.app.inference.BBox

/**
 * Turns Depth-Anything-V2's raw depth map into a single RELATIVE proximity value per box
 * (0f far .. 1f near), against a per-frame normalization. Runtime-agnostic (consumes a
 * [RawTensor]), so TFLite and QNN share it.
 *
 * HONESTY INVARIANT (Bible §5.4): this is RELATIVE proximity ordering, never metres. We
 * normalize each frame's map to 0..1 by robust percentiles and read the box region off
 * that — which is exactly the "reliable relative-proximity ordering" the pitch claims and
 * nothing more.
 *
 * NEAR/FAR ORDERING (read this): Depth-Anything-V2 outputs inverse-depth-like values —
 * LARGER = NEARER. So proximity = normalized value directly. If your specific export is
 * flipped (some depth heads emit distance, where larger = farther), set [nearIsHigh]=false
 * ONCE and the whole distance channel inverts correctly — getting this backwards silently
 * turns "approaching" into "receding", so it's a single, deliberate switch, logged at init.
 */
class DepthSampler(
    private val nearIsHigh: Boolean = true,
    /** shrink each box to its central fraction before sampling, to avoid edge bleed. */
    private val centralFraction: Float = 0.6f,
    /** stride when scanning the full map for percentiles (speed; 1 = every pixel). */
    private val percentileStride: Int = 4,
    /** robust normalization percentiles, to ignore a few outlier pixels. */
    private val loPct: Float = 0.02f,
    private val hiPct: Float = 0.98f,
) {

    /** Parsed depth map with its dimensions and this-frame normalization bounds. */
    class Frame(
        val map: FloatArray,
        val w: Int,
        val h: Int,
        val lo: Float,
        val hi: Float,
    ) {
        val valid: Boolean get() = hi > lo
    }

    /**
     * Parse the depth output tensor into a normalized [Frame]. Accepts [1,H,W],
     * [1,H,W,1], or [1,1,H,W]; anything else throws with the shape so you can adapt.
     */
    fun parse(depth: RawTensor): Frame {
        val (w, h) = resolveWh(depth.shape)
        val data = depth.data
        // robust min/max via sampled percentiles
        val sample = ArrayList<Float>(data.size / percentileStride + 1)
        var i = 0
        while (i < data.size) { sample += data[i]; i += percentileStride }
        sample.sort()
        val lo = sample[(sample.size * loPct).toInt().coerceIn(0, sample.size - 1)]
        val hi = sample[(sample.size * hiPct).toInt().coerceIn(0, sample.size - 1)]
        return Frame(data, w, h, lo, hi)
    }

    /** Relative proximity 0..1 for [box] (normalized frame coords). ~0.5 if map invalid. */
    fun proximityFor(frame: Frame, box: BBox): Float {
        if (!frame.valid) return 0.5f
        val cx = box.centerX; val cy = box.centerY
        val halfW = (box.right - box.left) * centralFraction / 2f
        val halfH = (box.bottom - box.top) * centralFraction / 2f
        val x1 = ((cx - halfW) * frame.w).toInt().coerceIn(0, frame.w - 1)
        val x2 = ((cx + halfW) * frame.w).toInt().coerceIn(0, frame.w - 1)
        val y1 = ((cy - halfH) * frame.h).toInt().coerceIn(0, frame.h - 1)
        val y2 = ((cy + halfH) * frame.h).toInt().coerceIn(0, frame.h - 1)

        // median of the central region is robust to a few bad pixels
        val vals = ArrayList<Float>((x2 - x1 + 1) * (y2 - y1 + 1))
        for (y in y1..y2) {
            val row = y * frame.w
            for (x in x1..x2) vals += frame.map[row + x]
        }
        if (vals.isEmpty()) return 0.5f
        vals.sort()
        val med = vals[vals.size / 2]
        return normalize(med, frame.lo, frame.hi)
    }

    /**
     * Find the single nearest region in the CENTER band of the map — used to synthesize a
     * proximity-only, identity-less detection when YOLO returns nothing but something is
     * clearly close (feeds the RED tier honestly, §5.3). Returns null if nothing is near
     * enough. The box is a small synthetic region around the nearest cell.
     */
    fun nearestCenterRegion(frame: Frame, minProximity: Float): Pair<BBox, Float>? {
        if (!frame.valid) return null
        val bandX1 = (frame.w * 0.35f).toInt(); val bandX2 = (frame.w * 0.65f).toInt()
        val bandY1 = (frame.h * 0.30f).toInt(); val bandY2 = (frame.h * 0.70f).toInt()
        var bestVal = Float.NEGATIVE_INFINITY; var bx = -1; var by = -1
        var y = bandY1
        while (y < bandY2) {
            val row = y * frame.w
            var x = bandX1
            while (x < bandX2) {
                val prox = normalize(frame.map[row + x], frame.lo, frame.hi)
                if (prox > bestVal) { bestVal = prox; bx = x; by = y }
                x += percentileStride
            }
            y += percentileStride
        }
        if (bx < 0 || bestVal < minProximity) return null
        val cx = bx.toFloat() / frame.w; val cy = by.toFloat() / frame.h
        val box = BBox(
            (cx - 0.08f).coerceIn(0f, 1f), (cy - 0.12f).coerceIn(0f, 1f),
            (cx + 0.08f).coerceIn(0f, 1f), (cy + 0.12f).coerceIn(0f, 1f),
        )
        return box to bestVal
    }

    private fun normalize(v: Float, lo: Float, hi: Float): Float {
        val t = ((v - lo) / (hi - lo)).coerceIn(0f, 1f)
        return if (nearIsHigh) t else 1f - t
    }

    private fun resolveWh(shape: IntArray): Pair<Int, Int> = when {
        shape.size == 3 -> shape[1] to shape[2]                    // [1,H,W]
        shape.size == 4 && shape[3] == 1 -> shape[1] to shape[2]   // [1,H,W,1]
        shape.size == 4 && shape[1] == 1 -> shape[2] to shape[3]   // [1,1,H,W]
        else -> throw IllegalStateException(
            "DepthSampler: unrecognized depth shape ${shape.joinToString("x")}. " +
                "Expected [1,H,W], [1,H,W,1], or [1,1,H,W]."
        )
    }
}
