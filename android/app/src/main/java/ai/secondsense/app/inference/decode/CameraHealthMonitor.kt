package ai.secondsense.app.inference.decode

import ai.secondsense.app.inference.CameraHealth
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Watches for the camera being tampered with, covered, or knocked off its mount — the
 * failure modes that silently wreck every downstream vision signal on a chest-worn rig.
 *
 * Two independent checks, run every frame off the grayscale the pipeline already computes:
 *
 *  1. OCCLUSION / LIGHT — a covered lens (clothing flap, hand, in a bag) or darkness reads as
 *     a near-flat, near-black frame: low luminance mean AND low spatial std. A merely dim
 *     scene (dusk, dim room) is still low-mean but keeps structure -> [CameraHealth.DIM].
 *     Always active, no setup needed.
 *
 *  2. MOUNT ANGLE — only judged AFTER the wearer has defined "level" for their rig by tapping
 *     Calibrate while holding the phone vertical ([ImuTracker.calibrateMountingOffset], which
 *     makes pitch/roll read ~0 at that pose). Then any sustained tilt beyond
 *     [pitchToleranceDeg] / [rollToleranceDeg] from vertical is [CameraHealth.MISALIGNED].
 *     Before calibration there is NO angle warning — the earlier auto-learned baseline was
 *     unreliable (it locked onto whatever tilt the phone happened to have at app start).
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
    private val dimMeanMax: Float = 42f,
    private val pitchToleranceDeg: Float = 14f,
    private val rollToleranceDeg: Float = 20f,
    private val sustainMs: Long = 1_500L,
    private val clearMs: Long = 900L,
) {
    private var stable = CameraHealth.OK
    private var pending = CameraHealth.OK
    private var pendingSinceMs = 0L

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

        val misaligned = imuCalibrated &&
            (abs(pitchDeg) > pitchToleranceDeg || abs(rollDeg) > rollToleranceDeg)

        val instant = when {
            mean < darkMeanMax && std < uniformStdMax -> CameraHealth.BLOCKED
            misaligned -> CameraHealth.MISALIGNED
            mean < dimMeanMax -> CameraHealth.DIM
            else -> CameraHealth.OK
        }

        if (instant != pending) { pending = instant; pendingSinceMs = nowMs }
        val needed = if (instant == CameraHealth.OK) clearMs else sustainMs
        if (pending != stable && nowMs - pendingSinceMs >= needed) stable = pending

        return stable
    }

    fun reset() {
        stable = CameraHealth.OK
        pending = CameraHealth.OK
        pendingSinceMs = 0L
    }
}
