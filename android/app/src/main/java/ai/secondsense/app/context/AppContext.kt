package ai.secondsense.app.context

/**
 * The user's current activity context. Each one reconfigures the whole app — which perception
 * subsystems run, how loud/often it cues, and (later) what a tap does and how the LLM
 * prioritizes. Ordered most-active -> least-active so a swipe can step through them.
 *
 * Callers: ContextManager, MainActivity, unit test.
 */
enum class AppContext { WALKING, STANDING, HOME, SITTING, TRANSIT, CONVERSATION }

/** How chatty spoken scene output is in this context. */
enum class Verbosity { SILENT, MINIMAL, NORMAL }

/**
 * What a context enables. Mirrors [ai.secondsense.app.perf.PerfPolicy] in shape so
 * MainActivity can merge the two (take the more conservative of thermal vs context per knob).
 *
 * SAFETY FLOOR: [hazardEnabled] = false turns off the continuous drop-off / overhead cueing,
 * but MainActivity still keeps a bare "something is very close and closing" haptic in every
 * context — a context tunes verbosity and sensitivity, it never removes the last line of
 * defense.
 *
 * @param sonification  allow the continuous obstacle cue loop at all.
 * @param hazardEnabled run the drop-off / overhead / hazard fusion (off on a moving vehicle,
 *                      where ego-motion + optical flow produce nothing but phantoms).
 * @param depthEveryN   Depth-Anything cadence (>=1; kept on even when quiet so "find my X"
 *                      still has a distance).
 * @param detectEveryN  YOLO frame stride (>=1).
 * @param auxPerception ML Kit OCR + face ("read the sign", "person facing you").
 */
data class ContextProfile(
    val sonification: Boolean,
    val hazardEnabled: Boolean,
    val depthEveryN: Int,
    val detectEveryN: Int,
    val auxPerception: Boolean,
    val verbosity: Verbosity,
    val label: String,
) {
    companion object {
        fun profileFor(ctx: AppContext): ContextProfile = when (ctx) {
            AppContext.WALKING -> ContextProfile(
                sonification = true, hazardEnabled = true, depthEveryN = 2, detectEveryN = 1,
                auxPerception = false, verbosity = Verbosity.NORMAL, label = "walking",
            )
            AppContext.STANDING -> ContextProfile(
                sonification = false, hazardEnabled = true, depthEveryN = 4, detectEveryN = 2,
                auxPerception = false, verbosity = Verbosity.MINIMAL, label = "standing",
            )
            AppContext.HOME -> ContextProfile(
                sonification = false, hazardEnabled = true, depthEveryN = 4, detectEveryN = 3,
                auxPerception = true, verbosity = Verbosity.MINIMAL, label = "home",
            )
            AppContext.SITTING -> ContextProfile(
                sonification = false, hazardEnabled = false, depthEveryN = 6, detectEveryN = 4,
                auxPerception = true, verbosity = Verbosity.SILENT, label = "sitting",
            )
            AppContext.TRANSIT -> ContextProfile(
                sonification = false, hazardEnabled = false, depthEveryN = 6, detectEveryN = 3,
                auxPerception = true, verbosity = Verbosity.SILENT, label = "transit",
            )
            AppContext.CONVERSATION -> ContextProfile(
                sonification = false, hazardEnabled = false, depthEveryN = 8, detectEveryN = 6,
                auxPerception = true, verbosity = Verbosity.SILENT, label = "conversation",
            )
        }
    }
}
