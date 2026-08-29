package ai.secondsense.app.inference.decode

import ai.secondsense.app.inference.Detection
import kotlin.math.hypot

/**
 * Multi-frame detection stabilization — raises EFFECTIVE accuracy with NO model change.
 *
 * A single-frame score is noisy: a real object flickers 0.28 / 0.41 / 0.33; a false positive
 * pops once at 0.55 then vanishes. This tracks each box across frames (nearest same-label
 * within a gate), keeps an EMA-smoothed score + a persistence count, and reports an adjusted
 * score so that:
 *   - a consistently re-seen detection gets a small confidence BOOST (recovers real recall),
 *   - a brand-new / one-frame detection is held DOWN until it proves itself (kills flicker
 *     false-positives, and makes the WHITE/BLUE/RED tier "warm up" honestly on fresh objects).
 *
 * Runtime-agnostic; call once per frame with the fused detection list, use the returned list.
 * Stateful — one instance per session; [reset] on a hard scene break.
 */
class DetectionStabilizer(
    private val matchGate: Float = 0.12f,
    private val scoreAlpha: Float = 0.5f,
    private val confirmHits: Int = 3,
    private val maxMissed: Int = 2,
    private val confirmedBoost: Float = 0.10f,
    private val unconfirmedPenalty: Float = 0.08f,
) {
    private class Track(
        var label: String?,
        var cx: Float,
        var cy: Float,
        var smScore: Float,
        var hits: Int,
        var missed: Int,
    )

    private val tracks = mutableListOf<Track>()

    fun update(dets: List<Detection>): List<Detection> {
        val matched = arrayOfNulls<Track>(dets.size)
        val trackUsed = BooleanArray(tracks.size)

        for ((di, d) in dets.withIndex()) {
            var best = -1
            var bestDist = matchGate
            for ((ti, t) in tracks.withIndex()) {
                if (trackUsed[ti] || t.label != d.label) continue
                val dist = hypot((d.box.centerX - t.cx).toDouble(), (d.box.centerY - t.cy).toDouble()).toFloat()
                if (dist < bestDist) { bestDist = dist; best = ti }
            }
            if (best >= 0) { trackUsed[best] = true; matched[di] = tracks[best] }
        }

        val out = ArrayList<Detection>(dets.size)
        for ((di, d) in dets.withIndex()) {
            val t = matched[di] ?: Track(d.label, d.box.centerX, d.box.centerY, d.score, 0, 0).also { tracks.add(it) }
            t.cx = d.box.centerX
            t.cy = d.box.centerY
            t.smScore = scoreAlpha * d.score + (1f - scoreAlpha) * t.smScore
            t.hits++
            t.missed = 0
            val adj = if (t.hits >= confirmHits) t.smScore + confirmedBoost else t.smScore - unconfirmedPenalty
            out += d.copy(score = adj.coerceIn(0f, 1f))
        }

        // Age only the pre-existing tracks (new ones added just above were matched this frame).
        for (ti in trackUsed.indices) if (!trackUsed[ti]) tracks[ti].missed++
        tracks.removeAll { it.missed > maxMissed }
        return out
    }

    fun reset() {
        tracks.clear()
    }
}
