package ai.secondsense.app.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.sqrt

/**
 * Counts footsteps for coarse dead-reckoning distance (object-memory feature).
 *
 * Deliberately does its OWN step detection off the raw accelerometer rather than the
 * hardware [Sensor.TYPE_STEP_DETECTOR] / [Sensor.TYPE_STEP_COUNTER] — those need the
 * ACTIVITY_RECOGNITION runtime permission (API 29+), and if it isn't granted they silently
 * deliver nothing, which is exactly the "it always says the object is straight ahead" bug
 * (position never advances). Raw accelerometer needs no permission and works everywhere.
 *
 * Detection: low-pass the acceleration magnitude to track gravity, take the residual, and
 * fire on each upward zero-ish crossing above [peakThresh] that clears the [refractoryMs]
 * gap (≈ max human step rate). Good enough for a rough ~1-room distance; [strideMeters] is a
 * fixed average.
 */
class PedometerTracker(
    context: Context,
    val strideMeters: Float = 0.72f,
    private val peakThresh: Float = 1.8f,   // m/s² of dynamic (gravity-removed) acceleration
    private val resetThresh: Float = 0.6f,
    private val refractoryMs: Long = 280L,
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    val available: Boolean get() = accel != null

    /** Monotonic step count since [start]. */
    @Volatile var stepCount: Long = 0L
        private set

    /** Wall-clock of the most recent detected step. */
    @Volatile var lastStepAtMs: Long = 0L
        private set

    /** True while the user appears to be actively walking (a step within the last ~1.4 s). */
    val isWalking: Boolean get() = System.currentTimeMillis() - lastStepAtMs < 1_400L

    /** Invoked on the sensor thread each time a step is registered. */
    var onStep: (() -> Unit)? = null

    private var gravityMag = 9.81f
    private var armed = true
    private var lastStepMs = 0L

    fun start() {
        gravityMag = 9.81f; armed = true; lastStepMs = 0L
        accel?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_ACCELEROMETER) return
        val (x, y, z) = event.values
        val mag = sqrt(x * x + y * y + z * z)
        gravityMag += 0.12f * (mag - gravityMag)
        val dyn = mag - gravityMag

        if (armed && dyn > peakThresh) {
            val now = System.currentTimeMillis()
            if (now - lastStepMs > refractoryMs) {
                lastStepMs = now
                lastStepAtMs = now
                stepCount++
                onStep?.invoke()
            }
            armed = false
        } else if (!armed && dyn < resetThresh) {
            armed = true
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
