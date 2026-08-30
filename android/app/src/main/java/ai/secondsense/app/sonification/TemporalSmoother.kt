package ai.secondsense.app.sonification

import kotlin.math.abs

/**
 * Ticket #16 — Temporal smoothing.
 *
 * Raw per-frame detections flicker: a target appears for one frame, a spurious box pops in
 * and out, identity jitters. If every such blip fired a cue, the audio/haptic stream would
 * stutter — the exact instability that made the thrown-object test too high-variance to keep
 * (Bible §19.1). This gate requires a target to PERSIST for [confirmFrames] consecutive
 * frames — same identity, roughly the same azimuth — before its cue is allowed to fire.
 *
 * Brief flickers (a hand waved through frame for a frame or two) never reach the cue engine;
 * a steady target confirms after a beat and then tracks live. Runtime-agnostic: it gates the
 * resolved [CueTarget], so it works identically on the mock, the TFLite engine, and QNN.
 *
 * Stateful — one instance per app session. Reset happens implicitly when the target is lost
 * (a null frame resets the streak so a new target starts fresh).
 */
class TemporalSmoother(
    /** frames a new target must persist before its cue fires (Bible §13.3). Two, not three:
     *  when the frame rate drops (thermal decimation, a slow reference delegate) a walking
     *  approach moves the target's azimuth enough between frames that a 3-long streak keeps
     *  resetting and the cue never fires. Two frames still rejects single-frame flicker. */
    private val confirmFrames: Int = 2,
    /** how far the azimuth may drift frame-to-frame and still count as "the same" target.
     *  Wide enough that a real approach (the target sliding across the center band as you
     *  close on it) stays "the same thing" instead of restarting the streak every frame. */
    private val azimuthGate: Float = 0.28f,
) {
    private var candidate: CueTarget? = null
    private var streak = 0
    private var confirmed = false

    /**
     * Feed the frame's resolved target (or null if nothing selected). Returns the target to
     * actually cue — null while a candidate is still being confirmed or on a flicker.
     */
    fun update(target: CueTarget?): CueTarget? {
        if (target == null) {
            reset()
            return null
        }
        val ref = candidate
        // "Same target" is primarily SPATIAL: something that stays at roughly the same azimuth
        // frame-to-frame is the same physical thing. Identity is only a secondary check — and
        // it must tolerate a null on EITHER side, because a confidence-tier flip to RED nulls
        // the label (TargetSelector.selectWithTier). Without that tolerance, a stable object
        // whose score hovers at the BLUE/RED boundary flips label present<->null every couple
        // frames, the streak keeps resetting, and its cue never fires.
        val sameTarget = ref != null &&
            abs(ref.azimuth - target.azimuth) <= azimuthGate &&
            (ref.label == target.label || ref.label == null || target.label == null)

        if (sameTarget) {
            streak++
            candidate = target                 // track the freshest values
            if (confirmed || streak >= confirmFrames) {
                confirmed = true
                return target
            }
            return null                        // still warming up — stay silent
        } else {
            // a different target (or first sighting): restart the streak, don't fire yet.
            candidate = target
            streak = 1
            confirmed = false
            return null
        }
    }

    fun reset() {
        candidate = null
        streak = 0
        confirmed = false
    }
}
