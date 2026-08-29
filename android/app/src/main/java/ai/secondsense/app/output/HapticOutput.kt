package ai.secondsense.app.output

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

/**
 * The haptic output hook (build-map #6).
 *
 * Bible §5.2 is emphatic: haptics are a PRIMARY channel for "how close", not a
 * <0.5m panic backstop. So this class exposes GRADED intensity from day one, even
 * though #6 only needs a test buzz. Ticket #21 fills in the real proximity->waveform
 * mapping; the surface it targets is already here.
 *
 * On-device only. No laptop dependency.
 */
class HapticOutput(context: Context) {

    private val vibrator: Vibrator? = run {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val mgr = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
            mgr?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }
    }

    val hasAmplitudeControl: Boolean
        get() = vibrator?.hasAmplitudeControl() == true

    /** #6 done-condition: a single test buzz fires on a button tap. */
    fun testBuzz() {
        vibrate(durationMs = 200, amplitude = 200)
    }

    /**
     * GRADED proximity buzz — the primary "how close" channel (#21).
     * @param proximity 0f (far) .. 1f (touching). Maps to amplitude; callers at
     *                  #21 will also modulate rhythm/repetition for extra range.
     */
    fun proximityPulse(proximity: Float) {
        val p = proximity.coerceIn(0f, 1f)
        // amplitude 1..255; keep a floor so a real-but-distant object is still felt.
        val amp = (40 + p * 215).toInt().coerceIn(1, 255)
        val dur = (30 + (1f - p) * 90).toLong()   // closer = shorter, snappier pulses
        vibrate(durationMs = dur, amplitude = amp)
    }

    /**
     * #24 rung 3 — PANIC. A distinct, unmistakable double-hit at max amplitude, clearly
     * different from the graded proximity pulse. Fires when something is very close,
     * regardless of what audio can say — the always-available "away from the wall" floor.
     */
    fun panic() {
        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            // two sharp hits: buzz, gap, buzz — reads as urgent, not just "close".
            val timings = longArrayOf(0, 60, 40, 60)
            val amps = intArrayOf(0, 255, 0, 255)
            val effect = if (hasAmplitudeControl) {
                VibrationEffect.createWaveform(timings, amps, -1)
            } else {
                VibrationEffect.createWaveform(timings, -1)
            }
            v.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            v.vibrate(longArrayOf(0, 60, 40, 60), -1)
        }
    }

    /**
     * #17 — DROP-OFF hazard buzz. A downward negative obstacle (curb / step-down / edge) is
     * a distinct hazard from "object ahead", so it gets a distinct, unmistakable rhythm:
     * three escalating pulses (soft → medium → hard), clearly different from [panic]'s two
     * equal hits and from the graded [proximityPulse]. Fire-and-forget; safe from any thread.
     */
    /**
     * @param urgency 0f (far/soft) .. 1f (imminent/full intensity). Fed from the Sobel
     *                edge locator's row position (#17 enrichment) when available — an edge
     *                right at the feet reads more urgent than one still a few steps out.
     *                Defaults to 1f (old behavior) when no distance signal exists.
     */
    /**
     * V3 drop-off plan — a distinct, SUBDUED cue for POSSIBLE_DROP, deliberately different
     * from [dropOff]'s 3-pulse escalating pattern: a single short, moderate-amplitude tap.
     * "Something looks off, worth a beat of caution" is a different message than "the ground
     * is gone" — using the same escalating pattern for both (as the original V2-only wiring
     * did) trained the user to either over-react to weak evidence or tune out real drops.
     * Caller is responsible for cooldown-gating repeated calls (see MainActivity).
     */
    fun possibleDrop() {
        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val timings = longArrayOf(0, 90)
            val amps = intArrayOf(0, 130)
            val effect = if (hasAmplitudeControl) {
                VibrationEffect.createWaveform(timings, amps, -1)
            } else {
                VibrationEffect.createWaveform(timings, -1)
            }
            v.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            v.vibrate(longArrayOf(0, 90), -1)
        }
    }

    fun dropOff(urgency: Float = 1f) {
        val v = vibrator ?: return
        val u = urgency.coerceIn(0.3f, 1f) // floor at 0.3 so it's never imperceptibly soft
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            // buzz, gap, buzz, gap, buzz — escalating amplitude reads as "the ground is gone".
            val timings = longArrayOf(0, 70, 50, 70, 50, 110)
            val amps = intArrayOf(0, (110 * u).toInt(), 0, (180 * u).toInt(), 0, (255 * u).toInt())
            val effect = if (hasAmplitudeControl) {
                VibrationEffect.createWaveform(timings, amps, -1)
            } else {
                VibrationEffect.createWaveform(timings, -1)
            }
            v.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            v.vibrate(longArrayOf(0, 70, 50, 70, 50, 110), -1)
        }
    }

    /**
     * Ticket #33 — a distinct pattern for an AUDIO hazard (car horn, siren, alarm) detected
     * off-camera, outside the vision pipeline's field of view entirely. Deliberately
     * different rhythm from [dropOff] (3 escalating pulses) and [panic] (2 equal sharp hits)
     * so the user can tell "something behind/beside me made a hazard sound" apart from
     * "there's a visible obstacle" or "you're about to hit something" without needing to see
     * the HUD — four short, evenly-spaced taps reads as "listen up," not "step back."
     */
    fun hazardSound() {
        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val timings = longArrayOf(0, 50, 60, 50, 60, 50, 60, 50)
            val amps = intArrayOf(0, 200, 0, 200, 0, 200, 0, 200)
            val effect = if (hasAmplitudeControl) {
                VibrationEffect.createWaveform(timings, amps, -1)
            } else {
                VibrationEffect.createWaveform(timings, -1)
            }
            v.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            v.vibrate(longArrayOf(0, 50, 60, 50, 60, 50, 60, 50), -1)
        }
    }

    /**
     * Ticket #28 — the "arrived" confirmation fired when voice goal-seeking reaches its
     * target. Deliberately GENTLE and distinct from every hazard pattern: three soft,
     * evenly-spaced low-amplitude taps that read as "you're here / settle", not "step back"
     * (panic), "the ground is gone" (dropOff), or "listen up" (hazardSound).
     */
    fun arrived() {
        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val timings = longArrayOf(0, 45, 120, 45, 120, 45)
            val amps = intArrayOf(0, 90, 0, 90, 0, 90)
            val effect = if (hasAmplitudeControl) {
                VibrationEffect.createWaveform(timings, amps, -1)
            } else {
                VibrationEffect.createWaveform(timings, -1)
            }
            v.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            v.vibrate(longArrayOf(0, 45, 120, 45, 120, 45), -1)
        }
    }

    /**
     * Head-height / OVERHEAD hazard — the Bible's #1 differentiator (§3: "catches what the
     * cane structurally cannot"). A hard jolt then a soft one — reads as "something's at your
     * head, pull back" — deliberately unlike dropOff (3 escalating), panic (2 equal sharp),
     * hazardSound (4 even), arrived (3 gentle). Fired edge-triggered by MainActivity.
     */
    fun overhead() {
        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val timings = longArrayOf(0, 90, 40, 55)
            val amps = intArrayOf(0, 255, 0, 130)
            val effect = if (hasAmplitudeControl) {
                VibrationEffect.createWaveform(timings, amps, -1)
            } else {
                VibrationEffect.createWaveform(timings, -1)
            }
            v.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            v.vibrate(longArrayOf(0, 90, 40, 55), -1)
        }
    }

    private fun vibrate(durationMs: Long, amplitude: Int) {
        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val amp = if (hasAmplitudeControl) amplitude else VibrationEffect.DEFAULT_AMPLITUDE
            val effect = VibrationEffect.createOneShot(durationMs, amp)
            v.vibrate(effect)
        } else {
            @Suppress("DEPRECATION")
            v.vibrate(durationMs)
        }
    }

    fun release() { vibrator?.cancel() }
}
