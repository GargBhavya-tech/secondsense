package ai.secondsense.app.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

/**
 * Research-candidate item (secondsense_research_candidates_v1.md, §7) — cross-checks a
 * vision-only drop-off detection against the phone's barometer, entirely independent of the
 * camera/depth model. This targets a REAL, CONFIRMED bug: Depth-Anything-V2 produced an
 * inverted/wrong-sign gradient on a genuine descending staircase photo this session
 * (stairs2.jpeg, validated via debug_dropoff_v2.py) — a barometer can't be fooled by texture
 * or lighting the way a monocular depth model can, so it's a genuinely independent signal.
 *
 * PHYSICS: near sea level, atmospheric pressure increases by roughly 0.12 hPa per 1 metre of
 * descent (the standard barometric approximation, dP/dh ≈ -0.12 hPa/m at sea-level conditions
 * — this is an approximation, not exact at all altitudes/temperatures, but good enough for
 * "is the user's altitude dropping over the last few seconds," which is all we need).
 *
 * DELIBERATELY NOT in the decode/ layer: DropOffDetector is runtime-agnostic (TFLite/QNN
 * share it, no Android dependency by design). This class is an app-layer ENRICHMENT that
 * MainActivity consults alongside the vision result — it never replaces the vision-based
 * detect(), only confirms or informs it, keeping the pure-algorithm layer untouched.
 *
 * Most phones expose Sensor.TYPE_PRESSURE; if absent (some budget devices lack a barometer),
 * [isAvailable] is false and every check degrades to "unknown" rather than crashing.
 */
class BarometerMonitor(context: Context) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
    private val pressureSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_PRESSURE)

    val isAvailable: Boolean get() = pressureSensor != null

    // Rolling window of (timestampMs, pressureHpa), oldest-first. A few seconds is enough to
    // catch "the last few steps" without being so long it reacts to weather/AC drafts.
    private data class Sample(val atMs: Long, val hpa: Float)
    private val window = ArrayDeque<Sample>()
    private val windowMs = 3_000L

    fun start() {
        val sensor = pressureSensor ?: return
        sensorManager?.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL)
    }

    fun stop() {
        sensorManager?.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_PRESSURE) return
        val now = System.currentTimeMillis()
        synchronized(window) {
            window.addLast(Sample(now, event.values[0]))
            while (window.isNotEmpty() && now - window.first().atMs > windowMs) {
                window.removeFirst()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    /**
     * @return the pressure trend over the rolling window, in hPa (positive = pressure rising
     *         = altitude dropping = descending; negative = ascending). Null if the barometer
     *         is unavailable or we don't have enough samples yet to say anything meaningful.
     */
    fun pressureTrendHpa(): Float? = synchronized(window) {
        if (window.size < 2) return null
        window.last().hpa - window.first().hpa
    }

    /**
     * @return true if the recent pressure trend is consistent with the user physically
     *         descending (stairs, a ramp, a curb) right now. A conservative threshold
     *         (0.03 hPa ≈ ~25cm of real descent) so normal barometric sensor noise doesn't
     *         false-positive — genuine stair descent over a few steps produces a much larger,
     *         sustained trend than sensor jitter does.
     */
    fun descendingConfirmed(minDeltaHpa: Float = 0.03f): Boolean {
        val trend = pressureTrendHpa() ?: return false
        return trend >= minDeltaHpa
    }

    /** Rough, approximate metres descended over the window — for HUD/dashboard display only, never a safety threshold on its own. */
    fun approxMetresDescended(): Float? = pressureTrendHpa()?.let { it / 0.12f }
}
