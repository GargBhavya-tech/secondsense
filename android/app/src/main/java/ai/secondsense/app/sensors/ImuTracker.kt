package ai.secondsense.app.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.atan2
import kotlin.math.sqrt

private const val DEG_PER_RAD = 57.29578f

/**
 * V3 drop-off plan §1 — time-synchronized gyroscope + accelerometer fusion via a complementary
 * filter, producing pitch/roll for [ai.secondsense.app.inference.decode.TraversableCorridor]
 * and a high-rotation flag to down-weight vision evidence during fast motion (plan's explicit
 * "reject or down-weight visual updates during very fast rotation" guidance).
 *
 * WHY A COMPLEMENTARY FILTER (not a full Kalman filter, which the plan also mentions as an
 * option): the plan's own wording — "gyro for short-term rotation, gravity from the
 * accelerometer for drift correction" — describes a complementary filter exactly; it's the
 * standard, well-documented approach for this exact sensor pair and is what every public
 * Android sensor-fusion tutorial implements this way. A full Kalman filter would need a proper
 * process/measurement noise model tuned from real device data we don't have yet — not
 * justified over the simpler, well-proven approach for a first version.
 *
 * HONEST LIMITATION: the sign convention below (positive pitch = looking down) is the standard
 * formula for a phone held with its screen facing the user — but this project mounts the phone
 * on the CHEST (see bible §17), which may need a mounting rotation offset.
 * [calibrateMountingOffset] exists for exactly that: call it once while the phone is in its
 * actual mounted position and pointed level, and every subsequent pitch/roll reading is
 * corrected relative to that captured baseline. This has NOT been empirically verified on a
 * real chest-mounted phone yet — treat the sign/axis convention as needing a real on-device
 * check before trusting it blindly for the corridor shift direction.
 */
class ImuTracker(context: Context) : SensorEventListener {

    /** Positive = looking down toward the floor, negative = looking up. Degrees. */
    var pitchDeg: Float = 0f
        private set

    /** Positive = tilted to one side. Degrees. Magnitude is what matters downstream. */
    var rollDeg: Float = 0f
        private set

    /** deg/s magnitude of the gyroscope's rotation rate — used for the high-rotation gate. */
    var angularVelocityDegPerSec: Float = 0f
        private set

    /**
     * Heading / yaw in degrees, 0..360, monotonic with the user turning their body.
     *
     * Derived by projecting the gyro's angular-rate vector onto the gravity (down) axis and
     * integrating — i.e. "how fast are we rotating about vertical, right now". This is
     * MOUNT-AGNOSTIC: it works with the phone flat, upright on the chest, or anywhere between,
     * unlike getOrientation()'s azimuth which degenerates near vertical. No magnetometer, so
     * the ABSOLUTE value is arbitrary (no true north) and it drifts a few deg/min — only
     * *differences* over the ~1-minute object-memory window are used, so that's acceptable.
     */
    var headingDeg: Float = 0f
        private set

    /** True once a heading source has produced at least one sample. */
    var hasHeading: Boolean = false
        private set

    /** True during fast rotation (whip-pan, stumble) — callers should hold/degrade rather than
     * trust a fresh vision read, per the plan's explicit guidance. */
    val isHighRotation: Boolean get() = angularVelocityDegPerSec > HIGH_ROTATION_THRESHOLD_DEG_S

    /** True once at least one accelerometer + gyroscope sample pair has been fused. */
    var hasValidReading: Boolean = false
        private set

    /** True once [calibrateMountingOffset] has been called — i.e. "level" has been defined for
     * this mount, so pitch/roll now read ~0 at the intended vertical pose. The camera-health
     * monitor only judges mount angle after this. */
    var hasMountCalibration: Boolean = false
        private set

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    // Low-passed gravity direction (device frame) — the axis we project gyro yaw onto.
    private var gravX = 0f
    private var gravY = 0f
    private var gravZ = 9.81f

    private var lastGyroTimestampNs: Long = 0L
    private var accelPitchRaw = 0f
    private var accelRollRaw = 0f
    private var mountingPitchOffset = 0f
    private var mountingRollOffset = 0f

    fun start() {
        // Fastest stable rate available; the plan asks for 100-200Hz — SENSOR_DELAY_GAME is
        // the standard Android tier that lands in that range on most devices without the
        // battery/thermal cost of SENSOR_DELAY_FASTEST for a continuously-running foreground use.
        gyro?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        accel?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    /** Capture the CURRENT reading as "level" for the phone's actual mounted position — call
     * once while physically mounted and pointed at the intended neutral corridor direction. */
    fun calibrateMountingOffset() {
        mountingPitchOffset = pitchDeg + mountingPitchOffset // fold in any prior offset
        mountingRollOffset = rollDeg + mountingRollOffset
        hasMountCalibration = true
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                val (ax, ay, az) = event.values
                // Low-pass toward gravity so the yaw projection has a stable "down" axis.
                gravX += 0.1f * (ax - gravX)
                gravY += 0.1f * (ay - gravY)
                gravZ += 0.1f * (az - gravZ)
                // Standard phone-frame formula: pitch from forward/back tilt, roll from side tilt.
                accelPitchRaw = Math.toDegrees(
                    atan2(-ax.toDouble(), sqrt((ay * ay + az * az).toDouble()))
                ).toFloat()
                accelRollRaw = Math.toDegrees(
                    atan2(ay.toDouble(), az.toDouble())
                ).toFloat()
                if (!hasValidReading) {
                    // First sample: snap straight to the accelerometer reading rather than
                    // filtering from an arbitrary 0 baseline.
                    pitchDeg = accelPitchRaw - mountingPitchOffset
                    rollDeg = accelRollRaw - mountingRollOffset
                    hasValidReading = true
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                val (gx, gy, gz) = event.values
                angularVelocityDegPerSec = Math.toDegrees(
                    sqrt((gx * gx + gy * gy).toDouble())
                ).toFloat()

                if (lastGyroTimestampNs != 0L && hasValidReading) {
                    val dt = (event.timestamp - lastGyroTimestampNs) / 1_000_000_000f
                    if (dt in 0f..0.5f) { // ignore absurd gaps (sensor pause/resume)
                        // Heading: angular-rate projected onto the gravity (down) axis and
                        // integrated. Mount-agnostic — no getOrientation() gimbal issue.
                        val gMag = sqrt(
                            (gravX * gravX + gravY * gravY + gravZ * gravZ).toDouble()
                        ).toFloat()
                        if (gMag > 1f) {
                            val yawRate = (gx * gravX + gy * gravY + gz * gravZ) / gMag // rad/s about down
                            headingDeg = ((headingDeg + yawRate * dt * DEG_PER_RAD) % 360f + 360f) % 360f
                            hasHeading = true
                        }
                        // Integrate gyro for the short-term estimate, complementary-blend with
                        // the accelerometer's gravity-derived (drift-free but noisy) estimate.
                        val gyroPitch = pitchDeg + Math.toDegrees((gx * dt).toDouble()).toFloat()
                        val gyroRoll = rollDeg + Math.toDegrees((gy * dt).toDouble()).toFloat()
                        val accelPitch = accelPitchRaw - mountingPitchOffset
                        val accelRoll = accelRollRaw - mountingRollOffset
                        pitchDeg = COMPLEMENTARY_ALPHA * gyroPitch + (1 - COMPLEMENTARY_ALPHA) * accelPitch
                        rollDeg = COMPLEMENTARY_ALPHA * gyroRoll + (1 - COMPLEMENTARY_ALPHA) * accelRoll
                    }
                }
                lastGyroTimestampNs = event.timestamp
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private companion object {
        /** Weight given to the gyro's short-term (drift-prone) estimate vs the accelerometer's
         * gravity-derived (noisy but drift-free) estimate — standard complementary-filter value. */
        const val COMPLEMENTARY_ALPHA = 0.98f
        const val HIGH_ROTATION_THRESHOLD_DEG_S = 90f
    }
}
