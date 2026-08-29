package ai.secondsense.app.perf

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.util.Log

/**
 * Keeps inference latency deterministic inside a non-ventilated chest harness.
 *
 * The SoC, strapped against ~37 C body heat with no airflow, hits its thermal limit in
 * 15-20 min; Android then throttles CPU/GPU by 40-60% and per-frame latency jumps ~40ms ->
 * ~200ms. For a drop-off warning that is an extra ~20-30cm of travel before the user hears it.
 *
 * This governor watches four signals and fuses them into a [ThermalTier]:
 *  1. PowerManager thermal STATUS (NONE..CRITICAL) — Android's own verdict, most authoritative.
 *  2. PowerManager thermal HEADROOM (0..1+, API 30) — a forecast; >0.9 means throttle imminent.
 *  3. Battery TEMPERATURE (deg C) — the "skin" proxy the problem statement calls out.
 *  4. Observed p90 inference latency — ground truth that throttling has ALREADY started.
 *
 * Escalation is immediate; de-escalation waits [deescalateMs] of sustained calm (hysteresis)
 * so the policy doesn't oscillate. On every tier / walking-state change it recomputes a
 * [PerfPolicy] and, if it changed, fires [onPolicy]; [onNotice] carries a short spoken line
 * on the first escalate and the return to nominal (honest degradation, Bible 5.3).
 *
 * Caller: MainActivity (start/stop in lifecycle, feed onInferenceMs, apply onPolicy).
 */
class ThermalGovernor(
    private val walkingSupplier: () -> Boolean,
    private val deescalateMs: Long = 20_000L,
    private val pollMs: Long = 4_000L,
) {
    var onPolicy: ((PerfPolicy) -> Unit)? = null
    var onNotice: ((String) -> Unit)? = null

    @Volatile var tier: ThermalTier = ThermalTier.NOMINAL
        private set
    @Volatile var batteryTempC: Float = Float.NaN
        private set
    @Volatile var headroom: Float = Float.NaN
        private set
    @Volatile var p90Ms: Long = 0L
        private set
    @Volatile var policy: PerfPolicy = PerfPolicy.policyFor(ThermalTier.NOMINAL, walking = true)
        private set

    private var power: PowerManager? = null
    private var appContext: Context? = null
    private val handler = Handler(Looper.getMainLooper())
    private var running = false

    private var statusTier: ThermalTier = ThermalTier.NOMINAL
    private var coolSinceMs: Long = 0L
    private var announcedTier: ThermalTier = ThermalTier.NOMINAL
    private var lastWalking = true

    private val latency = LongArray(30)
    private var latIdx = 0
    private var latFilled = 0
    private var windowsComputed = 0
    /** this device's own steady-state p90, captured once after warmup; latency escalation is
     *  RELATIVE to it (our QNN path is ~200ms/frame — an absolute ms threshold is meaningless). */
    private var baselineMs = 0L

    private val thermalListener = PowerManager.OnThermalStatusChangedListener { status ->
        statusTier = when {
            status >= PowerManager.THERMAL_STATUS_SEVERE -> ThermalTier.CRITICAL
            status >= PowerManager.THERMAL_STATUS_MODERATE -> ThermalTier.HOT
            status >= PowerManager.THERMAL_STATUS_LIGHT -> ThermalTier.WARM
            else -> ThermalTier.NOMINAL
        }
        Log.i(TAG, "thermal status=$status -> $statusTier")
        reassess()
    }

    private val poll = object : Runnable {
        override fun run() {
            if (!running) return
            sampleHeadroom()
            sampleBatteryTemp()
            reassess()
            handler.postDelayed(this, pollMs)
        }
    }

    fun start(context: Context) {
        if (running) return
        running = true
        appContext = context.applicationContext
        power = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
        runCatching { power?.addThermalStatusListener(thermalListener) }
        handler.post(poll)
    }

    /** Feed every frame's measured inference time (ms). Cheap; keeps a 30-sample p90. */
    fun onInferenceMs(ms: Long) {
        if (ms <= 0) return
        latency[latIdx] = ms
        latIdx = (latIdx + 1) % latency.size
        if (latFilled < latency.size) latFilled++
        if (latIdx == 0) {
            val sorted = latency.copyOf(latFilled).sortedArray()
            p90Ms = sorted[(sorted.size * 9 / 10).coerceIn(0, sorted.size - 1)]
            windowsComputed++
            // Capture (and only ever ratchet DOWN) the device's steady-state p90 during a
            // moment the OTHER signals agree is cool — so a warm-start doesn't bake a
            // throttled number in as "normal", and throttle detection stays sensitive.
            if (windowsComputed >= 3 && p90Ms in 15..500) {
                val coolNow = statusTier == ThermalTier.NOMINAL &&
                    batteryTier() == ThermalTier.NOMINAL &&
                    (headroom.isNaN() || headroom < 0.85f)
                if (coolNow && (baselineMs == 0L || p90Ms < baselineMs)) {
                    baselineMs = p90Ms
                    Log.i(TAG, "latency baseline = ${baselineMs}ms")
                }
            }
        }
    }

    private fun sampleHeadroom() {
        if (Build.VERSION.SDK_INT < 30) return
        headroom = runCatching { power?.getThermalHeadroom(60) ?: Float.NaN }.getOrDefault(Float.NaN)
    }

    private fun sampleBatteryTemp() {
        val ctx = appContext ?: return
        val sticky: Intent? = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tenths = sticky?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        if (tenths > 0) batteryTempC = tenths / 10f
    }

    private fun latencyTier(): ThermalTier = when {
        // Ignore cold-start warmup, and only judge once a baseline has been captured.
        windowsComputed < 3 || baselineMs <= 0L -> ThermalTier.NOMINAL
        p90Ms > baselineMs * 2.0 -> ThermalTier.HOT     // throttling has clearly landed
        p90Ms > baselineMs * 1.5 -> ThermalTier.WARM
        else -> ThermalTier.NOMINAL
    }

    fun stop() {
        running = false
        windowsComputed = 0; latIdx = 0; latFilled = 0; baselineMs = 0L
        runCatching { power?.removeThermalStatusListener(thermalListener) }
        handler.removeCallbacks(poll)
    }

    private fun headroomTier(): ThermalTier = when {
        headroom.isNaN() -> ThermalTier.NOMINAL
        headroom >= 0.98f -> ThermalTier.CRITICAL
        headroom >= 0.93f -> ThermalTier.HOT
        headroom >= 0.85f -> ThermalTier.WARM
        else -> ThermalTier.NOMINAL
    }

    private fun batteryTier(): ThermalTier = when {
        batteryTempC.isNaN() -> ThermalTier.NOMINAL
        batteryTempC >= 46f -> ThermalTier.CRITICAL
        batteryTempC >= 44f -> ThermalTier.HOT
        batteryTempC >= 42f -> ThermalTier.WARM
        else -> ThermalTier.NOMINAL
    }

    @Synchronized
    private fun reassess() {
        val raw = maxOf(statusTier, headroomTier(), batteryTier(), latencyTier())
        val now = System.currentTimeMillis()

        val next: ThermalTier
        if (raw.ordinal > tier.ordinal) {
            next = raw
            coolSinceMs = 0L
        } else if (raw.ordinal < tier.ordinal) {
            if (coolSinceMs == 0L) coolSinceMs = now
            next = if (now - coolSinceMs >= deescalateMs) {
                coolSinceMs = 0L
                ThermalTier.values()[tier.ordinal - 1]
            } else tier
        } else {
            coolSinceMs = 0L
            next = tier
        }

        val walking = walkingSupplier()
        if (next != tier || walking != lastWalking) {
            tier = next
            lastWalking = walking
            val p = PerfPolicy.policyFor(tier, walking)
            if (p != policy) {
                policy = p
                Log.i(TAG, "policy -> ${p.label}  (tier=$tier walking=$walking p90=${p90Ms}ms headroom=$headroom battC=$batteryTempC)")
                onPolicy?.invoke(p)
            }
        }

        if (tier.ordinal > announcedTier.ordinal && tier.ordinal >= ThermalTier.HOT.ordinal) {
            announcedTier = tier
            onNotice?.invoke("Device is warm. Running in power save — updates a little slower.")
        } else if (tier == ThermalTier.NOMINAL && announcedTier != ThermalTier.NOMINAL) {
            announcedTier = ThermalTier.NOMINAL
            onNotice?.invoke("Cooled down. Back to full speed.")
        }
    }

    private companion object {
        const val TAG = "SecondSense/thermal"
    }
}
