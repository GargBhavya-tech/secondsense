package ai.secondsense.app.memory

import ai.secondsense.app.inference.SettledSighting
import java.util.concurrent.ConcurrentHashMap

/**
 * Short-horizon episodic memory: "the last place I saw your <thing>".
 *
 * DELIBERATELY IN-MEMORY, not SQLite. The research doc proposed a persistent `sqlite-vec`
 * store, but persistence is only meaningful with a persistent *world frame* — and without
 * SLAM our coordinates live in [DeadReckoner]'s drifting local frame, which is reset every
 * app launch. Writing those to disk would just be a stale lie next session. A process-lifetime
 * map is the honest data structure for what this feature can actually deliver (~1 room /
 * ~1 minute). Swap in a real store here the day ORB-SLAM3 lands.
 *
 * Keyed by lowercased label; a newer sighting of the same class overwrites the older one.
 * Read from the camera-analysis thread (logging) and the voice thread (recall) — hence the
 * concurrent map.
 */
class ObjectMemory {

    data class Hit(
        val label: String,
        val distanceM: Float,
        /** -180..180, 0 = ahead, + = right, ±180 = directly behind. */
        val bearingDeg: Float,
        val ageMs: Long,
    )

    private class Entry(val wx: Float, val wz: Float, val tsMs: Long)

    private val entries = ConcurrentHashMap<String, Entry>()

    val size: Int get() = entries.size

    /** Commit a settled sighting, placing it in the local frame relative to [observer]. */
    fun remember(sighting: SettledSighting, observer: DeadReckoner.Pose, nowMs: Long) {
        val (wx, wz) = DeadReckoner.placeObject(observer, sighting.distanceM, sighting.bearingDeg)
        entries[sighting.label.trim().lowercase()] = Entry(wx, wz, nowMs)
    }

    /** Where is [label] now, relative to [observer]? Null if never seen this session. */
    fun recall(label: String?, observer: DeadReckoner.Pose, nowMs: Long): Hit? {
        val key = label?.trim()?.lowercase()?.ifEmpty { null } ?: return null
        val entry = entries[key]
            ?: entries.entries.firstOrNull { key.contains(it.key) || it.key.contains(key) }?.value
            ?: return null
        val (range, bearing) = DeadReckoner.relativeTo(observer, entry.wx, entry.wz)
        return Hit(key, range, bearing, (nowMs - entry.tsMs).coerceAtLeast(0L))
    }

    fun forget(label: String) { entries.remove(label.trim().lowercase()) }

    fun clear() = entries.clear()
}
