package ai.secondsense.app.ar

import kotlin.math.sqrt

/**
 * The accumulating semantic layer of the room scan: object instances pinned to world
 * coordinates in ARCore's tracking frame.
 *
 * Per-frame detections are noisy AND ARCore's world frame drifts (a few cm normally, more
 * after a tracking blip), so the same chair seen twice can land far apart. [observe] fuses on
 * the FLOOR PLANE only (horizontal distance — vertical scatter from depth error is the worst
 * offender) within [mergeRadiusM], keyed by a folded label so chair/couch/bench don't split.
 * [prune] then periodically fuses confirmed duplicates that drift produced anyway and drops
 * unconfirmed ghosts that were never re-seen.
 *
 * In-memory only for now. Synchronized — written from a worker thread, read from GL/UI.
 */
class RoomMap(
    private val mergeRadiusM: Float = 0.9f,
    private val confirmHits: Int = 3,
    private val ghostTtlMs: Long = 20_000L,
) {
    data class SemanticPoint(
        val label: String,
        var x: Float, var y: Float, var z: Float,
        var confidence: Float,
        var hits: Int,
        var lastSeenMs: Long,
        var confirmAt: Int = 3,
    ) {
        val confirmed: Boolean get() = hits >= confirmAt
    }

    private val points = ArrayList<SemanticPoint>()

    /** @return true if this observation created a NEW point (first sighting). */
    @Synchronized
    fun observe(label: String, x: Float, y: Float, z: Float, score: Float, nowMs: Long): Boolean {
        val key = fold(label)
        val hit = nearest(key, x, z)
        return if (hit != null) {
            val a = 0.25f
            hit.x += a * (x - hit.x); hit.y += a * (y - hit.y); hit.z += a * (z - hit.z)
            hit.confidence = maxOf(hit.confidence, score)
            hit.hits++
            hit.lastSeenMs = nowMs
            false
        } else {
            points.add(SemanticPoint(key, x, y, z, score, 1, nowMs, confirmAt = confirmHits))
            true
        }
    }

    /**
     * Housekeeping — call every second or so:
     *  - fuse any two SAME-key points now within [mergeRadiusM] on the floor (drift makes
     *    duplicates that observe() couldn't catch at creation time), heavier one wins;
     *  - drop points that never reached confirmation and haven't been seen for [ghostTtlMs].
     */
    @Synchronized
    fun prune(nowMs: Long) {
        points.removeAll { !it.confirmed && nowMs - it.lastSeenMs > ghostTtlMs }
        var merged = true
        while (merged) {
            merged = false
            outer@ for (i in points.indices) {
                for (j in i + 1 until points.size) {
                    val a = points[i]; val b = points[j]
                    if (a.label == b.label && horiz(a.x, a.z, b.x, b.z) < mergeRadiusM) {
                        val w = b.hits.toFloat() / (a.hits + b.hits)
                        a.x += w * (b.x - a.x); a.y += w * (b.y - a.y); a.z += w * (b.z - a.z)
                        a.hits += b.hits
                        a.confidence = maxOf(a.confidence, b.confidence)
                        a.lastSeenMs = maxOf(a.lastSeenMs, b.lastSeenMs)
                        points.removeAt(j)   // j > i, so index i is undisturbed
                        merged = true
                        break@outer
                    }
                }
            }
        }
    }

    @Synchronized fun confirmed(): List<SemanticPoint> = points.filter { it.confirmed }.map { it.copy() }
    @Synchronized fun all(): List<SemanticPoint> = points.map { it.copy() }
    @Synchronized fun confirmedCount(): Int = points.count { it.confirmed }

    @Synchronized
    fun summary(): String {
        val byLabel = points.filter { it.confirmed }.groupingBy { it.label }.eachCount()
        if (byLabel.isEmpty()) return "no objects yet"
        return byLabel.entries.sortedByDescending { it.value }
            .joinToString(", ") { (l, n) -> if (n > 1) "$l x$n" else l }
    }

    @Synchronized fun clear() = points.clear()

    /** Insert an already-fused point from a saved map (bypasses the merge — caller pre-dedups). */
    @Synchronized
    fun addRaw(label: String, x: Float, y: Float, z: Float, confidence: Float, hits: Int, nowMs: Long) {
        points.add(SemanticPoint(fold(label), x, y, z, confidence, maxOf(hits, confirmHits), nowMs, confirmHits))
    }

    private fun nearest(key: String, x: Float, z: Float): SemanticPoint? {
        var best: SemanticPoint? = null
        var bestD = mergeRadiusM
        for (p in points) {
            if (p.label != key) continue
            val d = horiz(p.x, p.z, x, z)
            if (d < bestD) { bestD = d; best = p }
        }
        return best
    }

    private fun horiz(x1: Float, z1: Float, x2: Float, z2: Float): Float =
        sqrt((x1 - x2) * (x1 - x2) + (z1 - z2) * (z1 - z2))

    /** Collapse near-synonymous COCO labels so drift-split duplicates still merge. */
    private fun fold(label: String): String {
        val l = label.lowercase()
        return when (l) {
            "couch", "sofa", "bench", "loveseat" -> "chair"
            "desk", "dining table", "coffee table" -> "table"
            "tv", "tvmonitor", "television", "monitor" -> "tv"
            else -> l
        }
    }
}
