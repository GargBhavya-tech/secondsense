package ai.secondsense.app.inference.decode

/**
 * V3 drop-off plan §2 — the IMU-stabilized traversable corridor: a normalized-coordinate
 * region of the frame representing "where the user could step next," shifted using the
 * phone's estimated pitch/roll (see [ImuTracker]) instead of a fixed lower-half band (V1's
 * known failure mode — see DropOffDetector's history).
 *
 * SIMPLIFICATION FROM THE PLAN: the plan describes a full trapezoid rotated by pitch AND roll.
 * This is an axis-aligned rectangle instead — GroundPlaneAnalyzer/EdgeLattice/DropOffDetector
 * all already operate on axis-aligned row/column ranges (no per-pixel rotation), so a rotated
 * trapezoid would need a much larger rewrite of every consumer for a benefit that's mostly
 * relevant at extreme roll angles, which [ImuTracker] already down-weights evidence during
 * (see HIGH_ROTATION handling). Pitch shifts the corridor vertically (looking down moves the
 * floor higher in-frame, so the corridor should start higher up); roll widens the corridor
 * slightly to cover the uncertainty a rotated rect can't otherwise represent.
 *
 * This is a WEIGHTING region as the plan specifies, not a hard crop in principle — but the
 * current consumers (EdgeLattice, GroundPlaneAnalyzer) treat it as a search bound rather than a
 * soft-decay mask, since per-pixel weighting would need reworking their row/column loops for a
 * marginal accuracy gain not yet justified by validation data.
 */
data class TraversableCorridor(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
) {
    companion object {
        private const val BASE_X1 = 0.15f
        private const val BASE_X2 = 0.85f
        private const val BASE_Y1 = 0.35f
        private const val BASE_Y2 = 1.0f

        /** Degrees of pitch that shift the corridor's top edge by 1% of frame height. */
        private const val PITCH_SHIFT_PER_DEGREE = 0.008f
        /** Degrees of roll that widen the corridor by 1% of frame width (each side). */
        private const val ROLL_WIDEN_PER_DEGREE = 0.004f

        val DEFAULT = TraversableCorridor(BASE_X1, BASE_Y1, BASE_X2, BASE_Y2)

        /**
         * @param pitchDeg positive = looking down (per [ImuTracker]'s convention)
         * @param rollDeg magnitude only matters (which side rolled doesn't change corridor width)
         */
        fun from(pitchDeg: Float, rollDeg: Float): TraversableCorridor {
            val yShift = pitchDeg * PITCH_SHIFT_PER_DEGREE
            val widen = kotlin.math.abs(rollDeg) * ROLL_WIDEN_PER_DEGREE
            return TraversableCorridor(
                x1 = (BASE_X1 - widen).coerceIn(0.02f, 0.4f),
                y1 = (BASE_Y1 - yShift).coerceIn(0.05f, 0.65f),
                x2 = (BASE_X2 + widen).coerceIn(0.6f, 0.98f),
                y2 = BASE_Y2,
            )
        }
    }
}
