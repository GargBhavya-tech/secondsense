package ai.secondsense.app.context

import android.os.Handler
import android.os.Looper
import android.util.Log

/**
 * Phase 2 of the activity-context system: infer [AppContext] from cheap motion signals and
 * feed it to [ContextManager.suggest], which debounces it (15 s of agreement) and yields to
 * anything the user set by voice / menu / swipe for [ContextManager]'s sticky window.
 *
 * Signals, all permission-free (raw accelerometer only — no ACTIVITY_RECOGNITION):
 *  - [walkingSupplier]   — [ai.secondsense.app.sensors.PedometerTracker.isWalking], a step in
 *                          the last ~1.4 s.
 *  - [vibrationSupplier] — [ai.secondsense.app.sensors.PedometerTracker.vibrationLevel], slow
 *                          EMA of dynamic-accel energy; a moving vehicle keeps it elevated with
 *                          no steps.
 *
 * What it can and cannot tell apart:
 *  - WALKING   — steps right now.
 *  - TRANSIT   — no steps for [vehicleSettleMs], but sustained vibration above
 *                [vehicleVibration] (bus / train / car floor).
 *  - STANDING  — no steps, low vibration. This is the generic "stopped" suggestion; SITTING,
 *                HOME and CONVERSATION need context the motion sensors don't carry (a chair vs
 *                the feet, a known room, a face-to-face), so those stay user-driven.
 *
 * The detector never calls [ContextManager.set] — only [ContextManager.suggest]. It cannot
 * override the user, only nudge when the user hasn't spoken recently.
 *
 * Caller: MainActivity (start in onResume, stop in onPause).
 */
class ContextAutoDetector(
    private val manager: ContextManager,
    private val walkingSupplier: () -> Boolean,
    private val vibrationSupplier: () -> Float,
    private val pollMs: Long = 2_000L,
    private val vehicleVibration: Float = 0.22f,
    private val vehicleSettleMs: Long = 4_000L,
) {
    // lazy so the pure classify() path + construction touch no Android framework (unit-testable)
    private val handler by lazy { Handler(Looper.getMainLooper()) }
    private var running = false
    private var lastWalkingAtMs = 0L

    private val poll = object : Runnable {
        override fun run() {
            if (!running) return
            val now = System.currentTimeMillis()
            if (walkingSupplier()) lastWalkingAtMs = now
            val guess = classify(
                walking = walkingSupplier(),
                vibration = vibrationSupplier(),
                msSinceWalking = now - lastWalkingAtMs,
            )
            if (guess != null) manager.suggest(guess, now)
            handler.postDelayed(this, pollMs)
        }
    }

    fun start() {
        if (running) return
        running = true
        lastWalkingAtMs = System.currentTimeMillis()   // assume in-motion until proven still
        handler.post(poll)
        Log.i(TAG, "auto-detect started")
    }

    fun stop() {
        running = false
        handler.removeCallbacks(poll)
    }

    /**
     * Pure classifier — unit-testable, no Android deps. Returns the [AppContext] the motion
     * signals point at, or null when they're ambiguous (leave the current context alone).
     */
    fun classify(walking: Boolean, vibration: Float, msSinceWalking: Long): AppContext? = when {
        walking -> AppContext.WALKING
        msSinceWalking < vehicleSettleMs -> null                     // step energy still decaying
        vibration >= vehicleVibration -> AppContext.TRANSIT
        else -> AppContext.STANDING
    }

    private companion object {
        const val TAG = "SecondSense/context"
    }
}
