package ai.secondsense.app.sonification

import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin

/**
 * Ticket #20 — Identity channel: auditory icons (+ spearcon fallback).
 *
 * Bible §5.1: identity rides on TIMBRE, never on the distance channel. Each object
 * class maps to a short, synthesized "auditory icon" — a caricature of the real-world
 * sound (dog-yip, footstep-thud, wooden knock) that a listener can name cold, with no
 * training on the common classes. §5.1 also keeps single-word SPEARCONS (sped-up speech)
 * as a fallback for classes without a bespoke icon.
 *
 * WHY SYNTHESIZED, NOT SAMPLE FILES: keeps the whole thing offline + asset-free, and —
 * more importantly — lets the icon inherit the frame's PAN and be rendered at the exact
 * moment the cue clock fires. A sample player would fight the timing loop.
 *
 * IMPORTANT INVARIANT: none of these icons encode distance. Distance is the caller's
 * pulse *rate* (#19). An icon's own pitch/character is a fixed identity signature.
 */
object AuditoryIcon {

    private const val SR = 44_100

    /** The bespoke-icon classes. Anything not here falls back to a spearcon. */
    val ICON_CLASSES = setOf("person", "chair", "dog", "vehicle", "door", "furniture")

    /**
     * Render one mono PCM icon for [label] at [durationMs]. Returns a Float array in
     * -1f..1f (the mixer applies pan + amplitude). Unknown / null labels return a
     * neutral "unknown" tick — the RED-tier honest signal (§5.3), never silence.
     */
    fun render(label: String?, durationMs: Int): FloatArray {
        val n = SR * durationMs / 1000
        return when (label) {
            "dog" -> dogYip(n)
            "person" -> footstepThud(n)
            "vehicle" -> lowRumble(n)
            "chair", "furniture" -> woodenKnock(n)
            "door" -> doorClack(n)
            null -> unknownTick(n)          // RED tier: depth present, no identity claim
            else -> unknownTick(n)          // caller should use spearcon() instead for these
        }
    }

    /** True if [label] has a bespoke icon; if false, the caller should use a spearcon. */
    fun hasIcon(label: String?): Boolean = label != null && label in ICON_CLASSES

    // ---- icon synthesizers -------------------------------------------------
    // Each is a caricature, tuned to be identifiable and short. Not hi-fi — legible.

    /** Dog: a quick rising yip — two fast chirps. */
    private fun dogYip(n: Int): FloatArray = buildArray(n) { i, t ->
        val f = 700.0 + 900.0 * t                       // rising
        val chirp = sin(2 * PI * f * i / SR)
        val gate = if (t < 0.4) 1.0 else if (t < 0.55) 0.0 else 1.0  // two yips
        chirp * gate * expDecay(t, 6.0)
    }

    /** Person: a soft low footstep-thud — filtered noise-ish body + low sine. */
    private fun footstepThud(n: Int): FloatArray = buildArray(n) { i, t ->
        val body = sin(2 * PI * 90.0 * i / SR)
        val click = sin(2 * PI * 180.0 * i / SR) * expDecay(t, 40.0) * 0.4
        (body + click) * expDecay(t, 9.0)
    }

    /** Vehicle: a low, slightly rough rumble. */
    private fun lowRumble(n: Int): FloatArray = buildArray(n) { i, t ->
        val a = sin(2 * PI * 70.0 * i / SR)
        val b = sin(2 * PI * 71.7 * i / SR)             // beating -> roughness
        (a + b) * 0.5 * plateau(t)
    }

    /** Furniture/chair: a dry wooden knock — short, mid, fast decay. */
    private fun woodenKnock(n: Int): FloatArray = buildArray(n) { i, t ->
        val f = 320.0
        (sin(2 * PI * f * i / SR) + 0.5 * sin(2 * PI * 2 * f * i / SR)) * expDecay(t, 22.0)
    }

    /** Door: a two-part clack (latch + panel). */
    private fun doorClack(n: Int): FloatArray = buildArray(n) { i, t ->
        val latch = sin(2 * PI * 500.0 * i / SR) * expDecay(t, 50.0)
        val panel = sin(2 * PI * 150.0 * i / SR) * expDecay((t - 0.15).coerceAtLeast(0.0), 18.0)
        (latch * 0.6 + panel)
    }

    /** RED-tier "I see something, I can't name it" — a neutral, honest tick. */
    private fun unknownTick(n: Int): FloatArray = buildArray(n) { i, t ->
        sin(2 * PI * 300.0 * i / SR) * expDecay(t, 30.0) * 0.6
    }

    // ---- helpers -----------------------------------------------------------

    private inline fun buildArray(n: Int, gen: (i: Int, t: Double) -> Double): FloatArray {
        val out = FloatArray(n)
        for (i in 0 until n) {
            val t = i.toDouble() / n
            out[i] = gen(i, t).toFloat().coerceIn(-1f, 1f)
        }
        return out
    }

    /** Exponential decay envelope, k = steepness. */
    private fun expDecay(t: Double, k: Double): Double = exp(-k * t)

    /** Soft attack + soft release plateau for sustained-ish icons. */
    private fun plateau(t: Double): Double = when {
        t < 0.15 -> t / 0.15
        t > 0.8 -> (1.0 - t) / 0.2
        else -> 1.0
    }.coerceIn(0.0, 1.0)
}
