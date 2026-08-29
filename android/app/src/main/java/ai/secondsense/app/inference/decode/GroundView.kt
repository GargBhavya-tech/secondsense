package ai.secondsense.app.inference.decode

import ai.secondsense.app.inference.Detection

/**
 * Object-mask suppression — the fix for the real on-device false positive (desk + laptop +
 * keyboard scene, both [EdgeLattice] and [GroundPlaneAnalyzer] fired on their own narrow
 * definitions). Uses the YOLO detections the engine already computes every frame — no new
 * model, no new data — to check whether a "drop-off" candidate is actually sitting on/behind a
 * desk-like object rather than open floor.
 *
 * SUPPRESSION VOCAB: [CocoLabels.toIconVocab] buckets dining table/tv/laptop/refrigerator/
 * oven/microwave into "furniture" and chair/couch/bench/bed/toilet into "chair"; keyboard and
 * mouse pass through unbucketed. All five of those icon-vocab labels are desk/seating-like
 * objects that commonly dominate a near-field view without being a walkable floor.
 */
object GroundView {
    private val SUPPRESSION_LABELS = setOf("furniture", "chair", "keyboard", "mouse")

    /**
     * Fraction (0..1) of [corridor]'s width covered (in x) by suppression-labeled detections
     * whose vertical center falls inside [yBandStart]..[yBandEnd] (both normalized 0..1,
     * full-frame). Cheap axis-aligned interval overlap, not pixel-exact — matches the
     * precision every other V3 component in this codebase already uses (row/column bands, not
     * per-pixel masks).
     */
    fun objectCoverage(
        detections: List<Detection>,
        corridor: TraversableCorridor,
        yBandStart: Float,
        yBandEnd: Float,
    ): Float {
        val corridorWidth = (corridor.x2 - corridor.x1).takeIf { it > 1e-4f } ?: return 0f
        var coveredWidth = 0f
        for (d in detections) {
            val label = d.label ?: continue
            if (label !in SUPPRESSION_LABELS) continue
            val boxCenterY = (d.box.top + d.box.bottom) / 2f
            if (boxCenterY !in yBandStart..yBandEnd) continue
            val overlapX1 = maxOf(corridor.x1, d.box.left)
            val overlapX2 = minOf(corridor.x2, d.box.right)
            if (overlapX2 > overlapX1) coveredWidth += (overlapX2 - overlapX1)
        }
        return (coveredWidth / corridorWidth).coerceIn(0f, 1f)
    }
}
