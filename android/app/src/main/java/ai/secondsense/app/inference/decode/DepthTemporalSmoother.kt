package ai.secondsense.app.inference.decode

/**
 * Research-candidate item (secondsense_research_candidates_v1.md, §2) — temporal EMA
 * smoothing of the raw depth map across consecutive depth-inference runs (NOT every camera
 * frame — depth itself only runs every Nth frame already, see TfliteInferenceEngine's
 * depthEveryN). Validated offline before writing this: ~47% frame-to-frame noise reduction
 * on simulated jitter, with negligible lag on the real underlying signal (alpha=0.3 tracks a
 * real trend within 2-3 depth-runs, not dozens).
 *
 * WHY: raw single-frame monocular depth is noisy pixel-to-pixel even when nothing in the
 * scene changed — sensor noise, lighting flicker, model non-determinism. That noise directly
 * feeds DropOffDetector's gradient search, so smoothing it reduces false-positive drop-off
 * triggers from noise alone, without touching the detection LOGIC itself (still the V2
 * adaptive Sobel + sign-check detector — this only cleans its input before it gets there).
 *
 * Deliberately NOT full Kalman/optical-flow registration (tracking corners frame-to-frame to
 * compensate for camera motion between smoothed frames) — that's a bigger, higher-risk build,
 * explicitly flagged as a "future session" item in the research doc. Plain EMA is the cheap,
 * validated first cut: correct when the camera is roughly still or panning slowly (normal
 * walking pace), imperfect during a fast whip-pan — an acceptable tradeoff for now.
 *
 * Runtime-agnostic (no Android dependency), like the rest of decode/ — TFLite and the future
 * QNN engine share this unchanged.
 */
class DepthTemporalSmoother(private val alpha: Float = 0.3f) {

    private var state: FloatArray? = null

    /**
     * @return a NEW smoothed array; [raw] is never mutated. Resets automatically (returns
     *         [raw] unchanged, seeds new state) if the depth map's element count changes
     *         (e.g. a debug-mode switch that alters input size) so a stale, wrong-shaped
     *         state can never silently corrupt output.
     */
    fun smooth(raw: FloatArray): FloatArray {
        val prev = state
        if (prev == null || prev.size != raw.size) {
            val fresh = raw.copyOf()
            state = fresh
            return fresh
        }
        val out = FloatArray(raw.size)
        for (i in raw.indices) {
            out[i] = alpha * raw[i] + (1f - alpha) * prev[i]
        }
        state = out
        return out
    }

    /** Call when depth resumes after a gap (e.g. re-entering FULL debug mode) so stale state doesn't blend into a fresh scene. */
    fun reset() {
        state = null
    }
}
