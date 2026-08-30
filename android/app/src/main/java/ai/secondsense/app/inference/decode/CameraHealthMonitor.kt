package ai.secondsense.app.inference.decode

import ai.secondsense.app.inference.CameraHealth
import kotlin.math.abs
import kotlin.math.sqrt

private fun angleDeltaDeg(a: Float, b: Float): Float {
    var d = a - b
    while (d > 180f) d -= 360f
    while (d < -180f) d += 360f
    return abs(d)
}

/**
 * Watches for the camera being tampered with, covered, or knocked off its mount — the
 * failure modes that silently wreck every downstream vision signal on a chest-worn rig.
 *
 * Two independent checks, run every frame off the grayscale the pipeline already computes:
 *
 *  1. OCCLUSION / LIGHT — a covered lens reads as a near-textureless frame: very low spatial
 *     std REGARDLESS of brightness (a hand in daylight, a pale shirt, a pocket all pin the
 *     std near zero even when the mean is nowhere near black), OR the classic dark-and-flat
 *     look of a lens covered inside a bag. A merely dim scene (dusk, dim room) is still
 *     low-mean but keeps structure -> [CameraHealth.DIM]. Always active, no setup needed.
 *
 *  2. MOUNT ANGLE — two ways in:
 *     (a) CALIBRATED: after the wearer taps Calibrate while vertical
 *         ([ImuTracker.calibrateMountingOffset] makes pitch/roll read ~0 at that pose), any
 *         sustained tilt beyond [pitchToleranceDeg] / [rollToleranceDeg] from vertical.
 *     (b) UNCALIBRATED: the monitor learns the rig's RESTING pitch/roll as a slow EMA from
 *         frames that already look healthy, then flags a sustained *change* of more than
 *         [autoTiltDeltaDeg] from that rest pose — i.e. "the camera has moved from where it
 *         was". This catches a knocked mount without any setup. The baseline only updates
 *         while the status is OK, so once it's flagged it stays flagged until the rig is put
 *         back. (The old auto-baseline was absolute, not a delta, and locked onto whatever
 *         tilt the phone had at app start — this one tracks the change instead.)
 *
 * Debounced: a bad status must persist [sustainMs] before it's reported, and OK must hold for
 * [clearMs] before recovery — so a stumble, a bend-down, or a hand briefly crossing the lens
 * doesn't trigger it.
 *
 * Grayscale values are 0..255 (see OpticalFlow.toGrayscale). Stateful; [reset] on session
 * restart. The caller wires the result into RawEvidence.sensorBlocked/lowLight (hazard fusion
 * -> SENSOR_BLOCKED, suppressing phantom drop-off) and FrameResult.cameraHealth (UI warns).
 */
class CameraHealthMonitor(
    private val darkMeanMax: Float = 20f,
    private val uniformStdMax: Float = 7f,
    /** covered-lens flatness, brightness-independent: below this std the frame has essentially
     *  no texture, which a real scene (even a blank wall — vignetting, noise, shadow gradient)
     *  never does, but a lens pressed against skin/cloth/pocket does. */
    private val flatStdMax: Float = 4.5f,
    private val dimMeanMax: Float = 42f,
    private val pitchToleranceDeg: Float = 14f,
    private val rollToleranceDeg: Float = 20f,
    /** uncalibrated fallback: sustained change from the learned rest pitch/roll that counts as
     *  "the camera has moved". Wider than the calibrated tolerances — it's a delta, and it has
     *  to clear normal walking bounce. */
    private val autoTiltDeltaDeg: Float = 17f,
    private val baselineAlpha: Float = 0.03f,
    private val sustainMs: Long = 1_500L,
    private val clearMs: Long = 900L,
) {
    private var stable = CameraHealth.OK
    private var pending = CameraHealth.OK
    private var pendingSinceMs = 0L

    // Uncalibrated mount-angle fallback: slow EMA of the resting pitch/roll, learned only from
    // healthy frames, plus a warm-up count so a knock in the first second can't poison it.
    private var basePitch = Float.NaN
    private var baseRoll = Float.NaN
    private var baselineSamples = 0

    /**
     * @param imuCalibrated true once the wearer has tapped Calibrate while vertical; only then
     *   are [pitchDeg]/[rollDeg] (which are then ~0 at the intended pose) judged for tilt.
     */
    fun update(
        gray: FloatArray,
        pitchDeg: Float,
        rollDeg: Float,
        imuCalibrated: Boolean,
        nowMs: Long,
    ): CameraHealth {
        if (gray.isEmpty()) return stable

        var sum = 0.0
        for (v in gray) sum += v
        val mean = (sum / gray.size).toFloat()
        var sq = 0.0
        for (v in gray) { val d = v - mean; sq += d * d }
        val std = sqrt(sq / gray.size).toFloat()

        val blocked = std < flatStdMax || (mean < darkMeanMax && std < uniformStdMax)

        val misaligned = if (imuCalibrated) {
            abs(pitchDeg) > pitchToleranceDeg || abs(rollDeg) > rollToleranceDeg
        } else {
            baselineSamples >= WARMUP_SAMPLES && !basePitch.isNaN() &&
                (angleDeltaDeg(pitchDeg, basePitch) > autoTiltDeltaDeg ||
                    angleDeltaDeg(rollDeg, baseRoll) > autoTiltDeltaDeg)
        }

        val instant = when {
            blocked -> CameraHealth.BLOCKED
            misaligned -> CameraHealth.MISALIGNED
            mean < dimMeanMax -> CameraHealth.DIM
            else -> CameraHealth.OK
        }

        if (instant != pending) { pending = instant; pendingSinceMs = nowMs }
        val needed = if (instant == CameraHealth.OK) clearMs else sustainMs
        if (pending != stable && nowMs - pendingSinceMs >= needed) stable = pending

        // Learn the rig's rest pose ONLY from frames that look healthy — never while blocked
        // or already flagged — so a knocked mount keeps reading as knocked until it's fixed.
        if (!imuCalibrated && stable == CameraHealth.OK && instant == CameraHealth.OK) {
            if (basePitch.isNaN()) { basePitch = pitchDeg; baseRoll = rollDeg } else {
                basePitch += baselineAlpha * angleSigned(pitchDeg, basePitch)
                baseRoll += baselineAlpha * angleSigned(rollDeg, baseRoll)
            }
            if (baselineSamples < WARMUP_SAMPLES) baselineSamples++
        }

        return stable
    }

    fun reset() {
        stable = CameraHealth.OK
        pending = CameraHealth.OK
        pendingSinceMs = 0L
        basePitch = Float.NaN
        baseRoll = Float.NaN
        baselineSamples = 0
    }

    private companion object {
        const val WARMUP_SAMPLES = 12
        /** signed wrapped (a - b), so the EMA can move either way across the ±180 seam. */
        fun angleSigned(a: Float, b: Float): Float {
            var d = a - b
            while (d > 180f) d -= 360f
            while (d < -180f) d += 360f
            return d
        }
    }
}
