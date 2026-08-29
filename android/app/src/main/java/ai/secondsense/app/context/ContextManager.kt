package ai.secondsense.app.context

/**
 * Owns the current [AppContext] and how it changes.
 *
 * Priority of sources:
 *  1. [set] — the user said so (voice / menu / swipe). STICKY: sensor suggestions are ignored
 *     for [stickyMs] afterwards, then auto-detection resumes.
 *  2. [suggest] — sensor-inferred (activity recognition, step cadence, vehicle vibration). A
 *     suggestion only takes effect after the SAME context has been suggested continuously for
 *     [graceMs] — no flip-flopping on a 3-second blip.
 *
 * On every real change it fires [onContext] (apply the profile) and [onAnnounce] (a short
 * spoken line). Pure state machine — no Android deps, unit-testable; the caller feeds it wall
 * time so tests can drive it deterministically.
 *
 * Caller: MainActivity; unit test.
 */
class ContextManager(
    private val stickyMs: Long = 5 * 60_000L,
    private val graceMs: Long = 15_000L,
) {
    var context: AppContext = AppContext.WALKING
        private set

    val profile: ContextProfile get() = ContextProfile.profileFor(context)

    var onContext: ((AppContext, ContextProfile) -> Unit)? = null
    var onAnnounce: ((String) -> Unit)? = null

    private var stickyUntilMs = 0L
    private var pendingSuggestion: AppContext? = null
    private var pendingSinceMs = 0L

    /** User-driven change. Always applied immediately; starts the sticky window. */
    fun set(ctx: AppContext, nowMs: Long, announce: Boolean = true) {
        stickyUntilMs = nowMs + stickyMs
        pendingSuggestion = null
        if (ctx == context) {
            if (announce) onAnnounce?.invoke("Already in ${ctx.spoken()} mode.")
            return
        }
        context = ctx
        onContext?.invoke(ctx, profile)
        if (announce) onAnnounce?.invoke("${ctx.spoken().replaceFirstChar { it.uppercase() }} mode.")
    }

    /** Step one notch toward a more / less active context (swipe up / down). Clamps at the ends. */
    fun step(moreActive: Boolean, nowMs: Long) {
        val order = AppContext.values()
        val i = (context.ordinal + if (moreActive) -1 else 1).coerceIn(0, order.size - 1)
        set(order[i], nowMs)
    }

    /** Cycle through all contexts, wrapping (spoken-menu use). */
    fun cycle(nowMs: Long) {
        val order = AppContext.values()
        set(order[(context.ordinal + 1) % order.size], nowMs)
    }

    /** Sensor hint. Ignored while sticky; needs [graceMs] of agreement before it lands. */
    fun suggest(ctx: AppContext, nowMs: Long) {
        if (nowMs < stickyUntilMs) { pendingSuggestion = null; return }
        if (ctx == context) { pendingSuggestion = null; return }
        if (ctx != pendingSuggestion) {
            pendingSuggestion = ctx
            pendingSinceMs = nowMs
            return
        }
        if (nowMs - pendingSinceMs >= graceMs) {
            pendingSuggestion = null
            context = ctx
            onContext?.invoke(ctx, profile)
            onAnnounce?.invoke(autoLine(ctx))
        }
    }

    fun reapply() = onContext?.invoke(context, profile)

    private fun autoLine(ctx: AppContext): String = when (ctx) {
        AppContext.WALKING -> "Looks like you're walking. Full guidance on."
        AppContext.STANDING -> "You've stopped. Going quiet — tap and ask me anything."
        AppContext.SITTING -> "Looks like you're settled. Staying quiet."
        AppContext.TRANSIT -> "Feels like a vehicle. Hazard alerts off, sign reading on."
        AppContext.HOME -> "You're in a mapped room. Using the room memory."
        AppContext.CONVERSATION -> "Conversation mode. Silent, haptics only."
    }
}

private fun AppContext.spoken(): String = when (this) {
    AppContext.WALKING -> "walking"
    AppContext.STANDING -> "standing"
    AppContext.HOME -> "home"
    AppContext.SITTING -> "sitting"
    AppContext.TRANSIT -> "transit"
    AppContext.CONVERSATION -> "conversation"
}
