package ai.secondsense.app.sonification

/**
 * Ticket #25 — the two operating modes. Bible §4: blind users don't walk in one
 * continuous mode — they STOP to orient, then move. The system mirrors that with a
 * deliberate mode split (stated as an HCI decision, not an implementation detail).
 *
 *   FLOW      — walking. Sparse, urgent-only. Center-crop ON (#14), one primary cue,
 *               obstacle/drop-off/head-height hazards only. This is the #14–#24 default.
 *   SCAN_SEEK — stopped, on-demand. Richer, exploratory: "what's around me?" scans the
 *               WHOLE frame (not just center), and this is the home for voice goal-seeking
 *               (Phase 4, #26–#28). Cue density is allowed to rise because the user is
 *               stationary and actively asking, not walking.
 */
enum class OperatingMode {
    FLOW,
    SCAN_SEEK,
}

/**
 * Holds the current mode and exposes the behavior each stage should adopt, so mode logic
 * lives in ONE place instead of being sprinkled through the camera loop, targeting, and
 * cue engine. Toggling the mode changes:
 *   - whether targeting uses the center-crop (#14) or the whole frame,
 *   - how many simultaneous cues are allowed,
 *   - whether voice goal-seeking (Phase 4) is accepting commands.
 */
class ModeController(initial: OperatingMode = OperatingMode.FLOW) {

    @Volatile var mode: OperatingMode = initial
        private set

    /** Listeners (e.g. the analyzer + HUD) notified on change. */
    private val listeners = mutableListOf<(OperatingMode) -> Unit>()

    fun addListener(l: (OperatingMode) -> Unit) { listeners.add(l); l(mode) }

    fun set(newMode: OperatingMode) {
        if (newMode == mode) return
        mode = newMode
        listeners.forEach { it(newMode) }
    }

    fun toggle() = set(if (mode == OperatingMode.FLOW) OperatingMode.SCAN_SEEK else OperatingMode.FLOW)

    // ---- behavior each stage reads off the current mode --------------------

    /** FLOW crops to center (#14, sparse); SCAN_SEEK considers the whole frame. */
    val useCenterCrop: Boolean get() = mode == OperatingMode.FLOW

    /** FLOW caps at one primary cue; SCAN_SEEK allows a small handful for exploration. */
    val maxSimultaneousCues: Int get() = if (mode == OperatingMode.FLOW) 1 else 3

    /** Voice goal-seeking (Phase 4) only listens in SCAN_SEEK — you stop, then ask. */
    val acceptsVoiceCommands: Boolean get() = mode == OperatingMode.SCAN_SEEK
}
