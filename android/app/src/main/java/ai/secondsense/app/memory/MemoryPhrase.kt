package ai.secondsense.app.memory

/**
 * Turns an [ObjectMemory.Hit] into a short spoken sentence — deliberately COARSE (buckets,
 * never "3.2 metres"), because the underlying distance is a rough estimate and the bearing
 * comes from drifting dead-reckoning. English only; the Hindi path runs this through the
 * on-device translator (same as scene narration).
 */
object MemoryPhrase {

    fun build(label: String, hit: ObjectMemory.Hit): String =
        "Your $label — last seen ${age(hit.ageMs)}, ${distance(hit.distanceM)}, ${direction(hit.bearingDeg)}."

    fun distance(m: Float): String = when {
        m < 1f -> "right next to you"
        m < 2.5f -> "a couple of steps away"
        m < 5f -> "a few steps away"
        else -> "across the room"
    }

    /** [bearingDeg] is -180..180 (0 = ahead, + = right). */
    fun direction(bearingDeg: Float): String {
        val a = ((bearingDeg % 360f) + 360f) % 360f
        return when {
            a < 25f || a > 335f -> "straight ahead"
            a < 70f -> "to your right"
            a < 110f -> "on your right"
            a < 160f -> "behind you on the right"
            a <= 200f -> "directly behind you"
            a < 250f -> "behind you on the left"
            a < 290f -> "on your left"
            else -> "to your left"
        }
    }

    fun age(ms: Long): String = when {
        ms < 45_000L -> "just now"
        ms < 300_000L -> "a few minutes ago"
        ms < 3_600_000L -> "a while ago"
        else -> "earlier"
    }
}
