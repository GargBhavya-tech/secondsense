package ai.secondsense.app.sonification

import ai.secondsense.app.inference.ConfidenceTier
import ai.secondsense.app.output.AudioOutput
import ai.secondsense.app.output.HapticOutput
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread
import kotlin.math.roundToLong

/**
 * Ticket #22 — Cue Engine Integration. The crown jewel.
 *
 * Runs a CONTINUOUS cue loop on its own thread, independent of the camera frame rate.
 * The vision side just posts the current [CueTarget] (via [update]); this loop decides
 * WHEN the next pulse fires and renders it across all channels AT ONCE, kept orthogonal:
 *
 *   DIRECTION (#18): AudioOutput pan, equal-power law, from target.azimuth.
 *   DISTANCE  (#19): the INTER-PULSE INTERVAL — faster pulses = closer. Pitch untouched.
 *   IDENTITY  (#20): the auditory icon (or spearcon) chosen by target.label — the timbre.
 *   PROXIMITY (#21): HapticOutput graded buzz in parallel — the PRIMARY how-close channel.
 *   TEXTURE   (§5.3): confidence tier shapes graininess/gain, never the above.
 *
 * The single hard rule this class exists to protect: no dimension bleeds into another's
 * channel. Distance changes the RATE, never the pitch. Identity changes the TIMBRE,
 * never the rate. Direction changes the PAN, nothing else.
 */
class CueEngine(
    private val audio: AudioOutput,
    private val haptics: HapticOutput,
    private val spearcon: Spearcon,
) {
    // The current target, swapped atomically by the vision thread. null = nothing to cue.
    private val current = AtomicReference<CueTarget?>(null)

    @Volatile private var running = false
    private var loopThread: Thread? = null

    // Ticket #34 — Voice Auto-Ducking, REVISED after a real safety concern: a flat 80% cut
    // applied to every cue regardless of urgency meant a close-range warning could go quiet
    // exactly when the user is distracted by conversation — audio can't tell "someone is
    // talking TO me" from "ambient chatter I'm not even part of," so a blanket duck in a
    // noisy street would near-silence navigational audio in exactly the environment that
    // needs it most. Two independent fixes, per that discussion:
    //   (a) max duck is now 50%, not 80% — never drops toward near-silent.
    //   (b) ducking fades OUT entirely as a target gets closer than [DUCK_SAFETY_PROXIMITY],
    //       and is fully disabled for RED tier — a close or identity-unknown-but-near
    //       warning is NEVER ducked, no matter what's being said nearby. Haptics remain
    //       completely unaffected either way (Bible §5.2 — the primary "how close" channel).
    @Volatile private var speechDetected: Boolean = false

    /**
     * @param ducked true while human speech is detected nearby (#34). Does NOT immediately
     *   cut volume by a fixed amount — see [duckFactorFor]: the actual reduction applied to
     *   each cue depends on how close/urgent that specific cue is. Safe to call from any thread.
     */
    fun setDucked(ducked: Boolean) {
        speechDetected = ducked
    }

    /** How much to multiply this cue's audio gain by, given the current speech-detected state. */
    private fun duckFactorFor(target: CueTarget): Float {
        if (!speechDetected) return 1f
        if (target.tier == ConfidenceTier.RED) return 1f // identity unknown + close -> never duck
        val p = target.proximity.coerceIn(0f, 1f)
        if (p >= DUCK_SAFETY_PROXIMITY) return 1f // close enough that ducking would be unsafe
        val t = p / DUCK_SAFETY_PROXIMITY // 0f (far) .. 1f (right at the safety floor)
        return DUCK_MIN_GAIN + (1f - DUCK_MIN_GAIN) * t
    }

    /** Cap on simultaneous cues (Bible §13.2 / #22): 1–2. This engine cues ONE primary. */
    // (A second, quieter secondary cue is a later enhancement; the spine cues the closest.)

    fun start() {
        if (running) return
        running = true
        loopThread = thread(name = "cue-loop", priority = Thread.MAX_PRIORITY) { loop() }
    }

    /** Post the latest resolved target. Called from the vision/targeting side each frame. */
    fun update(target: CueTarget?) {
        current.set(target)
    }

    fun stop() {
        running = false
        loopThread?.join(500)
        loopThread = null
    }

    // ---- the loop ----------------------------------------------------------

    private fun loop() {
        while (running) {
            val target = current.get()
            if (target == null) {
                // Nothing to cue — stay silent but responsive. Short sleep, re-check.
                Thread.sleep(40)
                continue
            }

            fireOnePulse(target)

            // DISTANCE CHANNEL (#19): the wait until the NEXT pulse is set by proximity.
            // Closer => shorter interval => faster pulse rate. Pitch is never touched here.
            val interval = intervalMsFor(target.proximity)
            sleepInterruptibly(interval)
        }
    }

    @Volatile private var lastPulseLogMs = 0L

    /** Render one pulse across audio (icon, panned) + haptics (graded), tier-shaped. */
    private fun fireOnePulse(target: CueTarget) {
        val nowMs = System.currentTimeMillis()
        if (nowMs - lastPulseLogMs >= 1000L) {
            lastPulseLogMs = nowMs
            android.util.Log.i(
                "SecondSense/cue",
                "pulse ${target.label ?: "?"} az=${"%.2f".format(target.azimuth)} " +
                    "prox=${"%.2f".format(target.proximity)} tier=${target.tier}",
            )
        }
        val dur = pulseDurationMs(target.proximity)
        val duck = duckFactorFor(target) // #34, urgency-aware — see duckFactorFor's doc comment

        // #24: resolve which rung we're on this frame. A CueTarget implies depth was
        // available (proximity comes from depth). The panic flag is decided in parallel.
        val decision = DegradationLadder.decide(
            tier = target.tier,
            proximity = target.proximity,
            depthAvailable = true,
            hasLabel = target.label != null,
        )

        // AUDIO — rung-dependent (#24 rungs 1 & 2).
        when (decision.audioRung) {
            LadderRung.FULL -> {
                // Rung 1: identity (#20) + direction (#18) + [distance via loop rate #19].
                val mono = when {
                    AuditoryIcon.hasIcon(target.label) -> AuditoryIcon.render(target.label, dur)
                    else -> target.label?.let { spearcon.get(it) } ?: AuditoryIcon.render(null, dur)
                }
                audio.playMono(applyTierTexture(mono, target.tier), pan = target.azimuth, gain = 1.0f * duck)
            }
            LadderRung.PROXIMITY -> {
                // Rung 2: drop identity, keep a proximity pulse + uncertainty texture.
                val bare = AuditoryIcon.render(null, dur)                 // no identity claim
                val gain = (if (target.tier == ConfidenceTier.BLUE) 0.85f else 0.7f) * duck
                audio.playMono(applyTierTexture(bare, target.tier), pan = target.azimuth, gain = gain)
            }
            LadderRung.SILENT_AUDIO -> {
                // No usable audio this frame. Haptics below may still fire. Never a hard stop.
            }
        }

        // PROXIMITY HAPTICS (#21): primary channel, always in parallel, independent modality.
        haptics.proximityPulse(target.proximity)

        // #24 rung 3 — PANIC FLOOR: an extra, unmistakable haptic hit when something is
        // very close, fired REGARDLESS of the audio rung (even under SILENT_AUDIO / RED).
        // This is the "always still buzz you away from a wall" guarantee.
        if (decision.panic) {
            haptics.panic()
        }
    }

    // ---- channel math ------------------------------------------------------

    /**
     * DISTANCE -> RATE mapping (#19). Far objects pulse slowly (~900ms apart), near
     * objects pulse rapidly (~120ms apart). The parking-sensor model. Monotonic,
     * so the ear reads "speeding up" as "getting closer" with zero training.
     */
    private fun intervalMsFor(proximity: Float): Long {
        val p = proximity.coerceIn(0f, 1f)
        val slow = 900.0
        val fast = 120.0
        return (slow + (fast - slow) * p).roundToLong()
    }

    /** Nearer cues are slightly shorter/snappier; keeps rapid pulses from smearing. */
    private fun pulseDurationMs(proximity: Float): Int {
        val p = proximity.coerceIn(0f, 1f)
        return (140 - 60 * p).toInt().coerceIn(70, 160)
    }

    /**
     * TEXTURE (§5.3). WHITE: clean. BLUE: add a grain of amplitude noise so it audibly
     * "sounds unsure". RED: already identity-less; leave the bare tick but roughen it a
     * little too. The system degrades audibly — it never goes silent, never fakes crisp
     * confidence it doesn't have.
     */
    private fun applyTierTexture(mono: FloatArray, tier: ConfidenceTier): FloatArray {
        if (tier == ConfidenceTier.WHITE) return mono
        val grain = if (tier == ConfidenceTier.BLUE) 0.18f else 0.12f
        val out = FloatArray(mono.size)
        var seed = 0x9E3779B9.toInt()
        for (i in mono.indices) {
            // cheap deterministic noise
            seed = seed * 1103515245 + 12345
            val n = ((seed ushr 16) and 0x7FFF) / 32768f * 2f - 1f
            out[i] = (mono[i] * (1f - grain) + n * grain).coerceIn(-1f, 1f)
        }
        return out
    }

    private fun sleepInterruptibly(ms: Long) {
        // Sleep in small slices so a proximity change re-rates quickly without waiting
        // out a long far-object interval.
        var remaining = ms
        val slice = 30L
        while (running && remaining > 0) {
            val s = minOf(slice, remaining)
            Thread.sleep(s)
            remaining -= s
            // If the target got much closer mid-wait, cut the wait short.
            val now = current.get() ?: return
            val freshInterval = intervalMsFor(now.proximity)
            if (freshInterval < remaining) remaining = freshInterval
        }
    }

    private companion object {
        /** Max reduction applied while ducked (0.5 = 50% cut, never near-silent). */
        const val DUCK_MIN_GAIN = 0.5f
        /** Proximity at/above which ducking is fully disabled — safety floor for close cues. */
        const val DUCK_SAFETY_PROXIMITY = 0.6f
    }
}
