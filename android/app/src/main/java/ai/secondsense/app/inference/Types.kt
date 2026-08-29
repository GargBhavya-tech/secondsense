package ai.secondsense.app.inference

/**
 * SecondSense — shared pipeline types.
 *
 * These are the contracts every stage speaks in. They are deliberately defined
 * up front (ticket #6) so that #12 (YOLO loop), #13 (depth loop), #15 (targeting),
 * and Phase 3 (sonification) all plug into the SAME shapes without a rewrite.
 *
 * Invariants baked into the types (from the Bible):
 *  - Depth is RELATIVE proximity, never metres (§5.4). Hence `proximity`, not
 *    `distanceMeters`. Range 0f (far) .. 1f (touching), against the #7 baseline.
 *  - Confidence tiering is first-class (§5.3), not a boolean — see [ConfidenceTier].
 *  - Identity and distance are separate fields because they become separate audio
 *    channels (§5.1); nothing here lets pitch do double duty.
 */

/** A normalized bounding box in frame space. All values 0f..1f. */
data class BBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {
    /** Horizontal center, 0f (hard left) .. 1f (hard right). Drives HRTF pan (#18). */
    val centerX: Float get() = (left + right) / 2f
    /** Vertical center, 0f (top) .. 1f (bottom). */
    val centerY: Float get() = (top + bottom) / 2f
    /** Fraction of the frame this box covers — a crude size/closeness prior. */
    val area: Float get() = ((right - left).coerceAtLeast(0f)) * ((bottom - top).coerceAtLeast(0f))
}

/**
 * Confidence tiering (Bible §5.3, ticket #23). The audio texture degrades audibly
 * across tiers — the system sounds unsure, never goes silent, never fakes confidence.
 * Defined here so every detection carries a tier from the very first frame.
 */
enum class ConfidenceTier {
    /** Clean detection + confident depth. Full three-channel cue. */
    WHITE,
    /** Real but low-confidence (poor light / partial occlusion). Grainy texture. */
    BLUE,
    /** Depth present, no reliable identity. Proximity-only, no identity claim. */
    RED,
}

/**
 * One detected thing in the frame. This is what the vision stage emits and the
 * targeting/sonification stages consume.
 *
 * @param label       class name from YOLO (#12), or null when identity is unknown (RED tier).
 * @param score       raw detector confidence 0f..1f.
 * @param box         normalized location in frame.
 * @param proximity   RELATIVE closeness 0f(far)..1f(near) from depth (#13). Never metres.
 * @param approaching rate-of-approach sign/magnitude, +ve = getting closer. From #13.
 * @param moving      coarse frame-to-frame moving/static flag (#15). Not velocity.
 * @param tier        confidence tier (#23).
 */
data class Detection(
    val label: String?,
    val score: Float,
    val box: BBox,
    val proximity: Float,
    val approaching: Float = 0f,
    val moving: Boolean = false,
    val tier: ConfidenceTier = ConfidenceTier.WHITE,
)

/**
 * The full result of running inference on one frame. The engine returns this;
 * the app renders/sonifies it. `depthAvailable` lets the RED tier fire (proximity
 * without identity) even when detection returns nothing.
 */
data class FrameResult(
    val detections: List<Detection>,
    val frameWidth: Int,
    val frameHeight: Int,
    val inferenceMillis: Long,
    val depthAvailable: Boolean = true,
    /**
     * Ticket #17 — a downward drop-off / negative obstacle (curb, step-down, platform edge,
     * pothole) was detected in the lower-center of the depth map this frame. Semantically
     * distinct from an "object in the way": it's the ABSENCE of ground, not the presence of
     * a thing. Defaults false so the mock and every existing call site are unaffected.
     */
    val dropOff: Boolean = false,
    /**
     * Ticket #17 enrichment — WHERE the drop-off is, not just that it's there. 0f (top of
     * frame) .. 1f (bottom/imminent). Null when [dropOff] is false, or when it's true but the
     * Sobel edge locator couldn't find a clean discontinuity above its noise threshold (the
     * fixed-band gate can fire on a softer, less-localizable slope than the edge finder
     * requires — that's fine, the flat hazard cue still applies, just without distance).
     */
    val dropOffRowFraction: Float? = null,
    /**
     * DEBUG/VALIDATION ONLY — the center-frame relative proximity BEFORE temporal EMA
     * smoothing is applied, sampled the same way as [Detection.proximity]'s underlying depth
     * read. Exists so DebugActivity can show raw-vs-smoothed side by side and you can
     * literally watch the smoothed number hold steadier than the raw one, frame to frame,
     * instead of trusting the smoothing works on faith. Null when depth didn't run this
     * frame (reused a cached map) or engine is a mock.
     */
    val debugRawCenterProximity: Float? = null,
    /** Same region, same frame, AFTER temporal EMA smoothing — the direct comparison point. */
    val debugSmoothedCenterProximity: Float? = null,
    /**
     * DEBUG/VALIDATION ONLY — the estimated camera ego-motion this frame (Lucas-Kanade
     * optical flow, normalized 0f..1f-per-frame units), so DebugActivity can show it
     * directly. Null on the first frame (no previous frame to compare against yet).
     */
    val debugEgoMotionX: Float? = null,
    val debugEgoMotionY: Float? = null,
    /**
     * V3 drop-off plan's output contract — SAFE/POSSIBLE_DROP/DROP_CONFIRMED/SENSOR_BLOCKED
     * from [ai.secondsense.app.inference.decode.HazardStateMachine], fusing the RGB edge
     * lattice + depth-as-evidence + IMU corridor stabilization. Null when the engine hasn't
     * wired the V3 fusion pipeline (e.g. MockInferenceEngine). This is IN ADDITION TO
     * [dropOff]/[dropOffRowFraction] (the existing V2 OR-logic detector), not a replacement —
     * both run; see TfliteInferenceEngine's doc comment for how they combine.
     */
    val hazardState: ai.secondsense.app.inference.decode.HazardState? = null,
    val hazardConfidence: Float? = null,
    val hazardUrgency: Float? = null,
    val hazardFirstEdgeY: Float? = null,
)
