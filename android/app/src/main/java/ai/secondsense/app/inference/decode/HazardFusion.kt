package ai.secondsense.app.inference.decode

/**
 * V3 drop-off plan — output contract + rule-baseline fusion (plan §7's "explicit rule
 * baseline for debuggability", implemented before/instead of the learned fusion model, since
 * that needs logged field data this session doesn't have) and the temporal hazard state
 * machine (plan §8) + approach-based urgency (plan §9).
 *
 * DEFERRED, PER SCOPE (see EngineConfig-adjacent conversation): the semantic
 * ascending/descending CNN classifier (plan §3) needs 250-400 real video clips across 20+
 * staircases this project doesn't have yet. [ReasonBits.semanticStairs] and
 * [ReasonBits.footOcclusion] are therefore always false — real flags waiting for that future
 * work, not fake placeholders. [ReasonBits.proximityBlocked] is likewise always false: no
 * proximity-sensor wiring exists in this codebase yet for hazard sensing (HazardSoundDetector's
 * mic listening is unrelated).
 */
enum class HazardState { SAFE, POSSIBLE_DROP, DROP_CONFIRMED, SENSOR_BLOCKED, SCENE_NOT_TRAVERSABLE }

enum class DepthVerdict { SUPPORTS, CONTRADICTS, UNRELIABLE }

data class ReasonBits(
    val semanticStairs: Boolean = false, // always false — needs the deferred CNN classifier
    val edgeLattice: Boolean = false,
    val depthSupport: Boolean = false,
    val lowLight: Boolean = false,
    val footOcclusion: Boolean = false, // always false — needs the deferred CNN classifier
    val highRotation: Boolean = false,
    val proximityBlocked: Boolean = false, // always false — no proximity sensor wired for hazard sensing
)

data class HazardFrameOutput(
    val state: HazardState,
    val confidence: Float,
    val firstEdgeY: Float?,
    val urgency: Float,
    val reasonBits: ReasonBits,
)

/** Per-frame evidence before temporal smoothing — [HazardStateMachine] consumes a sequence of these. */
data class RawEvidence(
    val latticeScore: Float,
    val nearestEdgeY: Float?,
    val depthVerdict: DepthVerdict,
    val highRotation: Boolean,
    val lowLight: Boolean,
    val sensorBlocked: Boolean,
    /**
     * REAL FALSE-POSITIVE FOUND on first live on-device test (user report, desk/laptop/
     * keyboard scene): a keyboard's periodic rows satisfy [EdgeLattice]'s "several parallel
     * near-horizontal evenly-spaced lines" criteria just as well as a real stair lattice does,
     * and a desk's real edge produces a real depth discontinuity — so BOTH evidence channels
     * fired correctly on their own narrow definitions while being wrong about the actual
     * scene. Fraction (0..1) of the lattice's candidate row-band that overlaps YOLO detections
     * of desk-like objects (laptop/table/tv/chair/couch/bed/keyboard/mouse, via
     * CocoLabels.toIconVocab's "furniture"/"chair" buckets plus raw "keyboard"/"mouse") —
     * see [GroundView.suppressionOverlap]. 0f when no such detections exist this frame.
     */
    val objectOverlap: Float = 0f,
    /**
     * Fraction (0..1) of the NEAR-FIELD corridor band (closest to the user, where a real
     * walking surface must be visible for a drop-off to even be meaningful) covered by those
     * same desk-like object detections. High near-field coverage means "there's a desk/table
     * in the way," not "there's floor here" — see [GroundView.groundViewValid].
     */
    val nearFieldObjectCoverage: Float = 0f,
    /**
     * Problem Statement 3 ("Specular Trap") — Veto A. 0..1 confidence that the candidate
     * drop-off edge is a CAST SHADOW (sharp luminance step, no chromaticity step) rather than
     * a physical one — see [ShadowChromaticity]. Only ever downgrades a confirmation; a real
     * geometric depth SUPPORTS still wins, and dark-but-real materials keep a hue edge so they
     * don't score here. 0f when there's no candidate edge to test.
     */
    val shadowLikelihood: Float = 0f,
    /**
     * Problem Statement 3 ("Specular Trap") — Veto B. 0..1 confidence that the candidate
     * drop-off region is moving COPLANAR with the surrounding floor (affine optical-flow
     * residual is low across the edge band) — i.e. a flat puddle / wet marble / polished tile
     * the depth net hallucinated a void in, not a real hole. See [GroundFlowConsistency].
     * 0f when the wearer is stationary or there isn't enough trackable texture.
     */
    val groundCoplanar: Float = 0f,
)

object HazardFusion {
    private const val STRONG_LATTICE = 0.55f
    private const val WEAK_LATTICE = 0.3f
    /** Real desk-scene false positive fix: past this row fraction, an edge is close enough to
     * the frame's bottom border that desk rims/laptop bases/mount artifacts are common — the
     * plan's "bottom-border reliability rule." Confirming a drop there needs depth to actively
     * SUPPORT, not just "strong lattice + anything non-contradicting." */
    private const val BOTTOM_BORDER_Y = 0.94f
    /** Near-field object coverage above this -> the walking surface itself isn't visible, so a
     * drop-off reading is meaningless regardless of what the lattice/depth say. */
    private const val GROUND_VIEW_INVALID_COVERAGE = 0.50f
    /** Lattice score above [STRONG_LATTICE] but overlapping a desk-like object by more than
     * this fraction gets rejected outright (real desk/keyboard false positive fix). */
    private const val OBJECT_REJECT_OVERLAP = 0.60f
    /** Between [OBJECT_REJECT_OVERLAP] and this, the lattice score is scaled down rather than
     * rejected outright — YOLO boxes can legitimately overlap a real stair edge too. */
    private const val OBJECT_DAMPEN_OVERLAP = 0.30f
    /** Specular Trap veto A — chromaticity shadow confidence at/above which a DROP_CONFIRMED
     * is downgraded to POSSIBLE_DROP (see [RawEvidence.shadowLikelihood]). */
    private const val SHADOW_VETO = 0.6f
    /** Specular Trap veto B — flow-coplanarity confidence at/above which the "hole" is really
     * a flat wet/glossy surface: downgrade DROP_CONFIRMED and clear POSSIBLE_DROP. */
    private const val COPLANAR_VETO = 0.6f

    /**
     * Single-frame rule baseline. VALIDATED OFFLINE (debug_v3_fusion.py) on 4 static regression
     * photos (correctly SAFE on a flat floor and ascending stairs, DROP_CONFIRMED on the hard
     * descending case). REVISED after a real on-device false positive on a desk/keyboard scene
     * — that scene independently satisfied both the lattice and depth checks on their own
     * narrow definitions, which the 4 static photos never exercised (none of them contain
     * furniture). Object-overlap suppression and the ground-view gate below are the fix,
     * not a rollback of the original validated logic.
     */
    fun classifySingleFrame(evidence: RawEvidence): HazardState {
        // Ground-view gate FIRST — if the near-field isn't a visible walkable surface (a desk/
        // table/laptop dominates it), no combination of lattice+depth evidence should be able
        // to promote to a drop state. This must outrank everything else.
        if (evidence.nearFieldObjectCoverage >= GROUND_VIEW_INVALID_COVERAGE) {
            return HazardState.SCENE_NOT_TRAVERSABLE
        }

        // Object-overlap suppression on the lattice score itself (a real stair edge can
        // legitimately be near a YOLO box too, so this dampens rather than always rejecting).
        val effectiveLattice = when {
            evidence.objectOverlap >= OBJECT_REJECT_OVERLAP -> 0f
            evidence.objectOverlap >= OBJECT_DAMPEN_OVERLAP ->
                evidence.latticeScore * (1f - evidence.objectOverlap)
            else -> evidence.latticeScore
        }

        val strong = effectiveLattice >= STRONG_LATTICE
        val weak = effectiveLattice >= WEAK_LATTICE
        val nearBottomBorder = (evidence.nearestEdgeY ?: 0f) > BOTTOM_BORDER_Y

        // Bottom-border reliability: strong lattice alone, right at the frame's bottom edge,
        // is exactly the desk-rim/laptop-base/mount-artifact failure mode — require depth to
        // actively support before confirming there, not just "not contradicting."
        if (strong && nearBottomBorder && evidence.depthVerdict != DepthVerdict.SUPPORTS) {
            return if (weak) HazardState.POSSIBLE_DROP else HazardState.SAFE
        }

        val base = when {
            strong && evidence.depthVerdict == DepthVerdict.SUPPORTS -> HazardState.DROP_CONFIRMED
            strong && evidence.depthVerdict == DepthVerdict.UNRELIABLE -> HazardState.POSSIBLE_DROP
            strong && evidence.depthVerdict == DepthVerdict.CONTRADICTS -> HazardState.POSSIBLE_DROP
            weak && evidence.depthVerdict == DepthVerdict.SUPPORTS -> HazardState.POSSIBLE_DROP
            else -> HazardState.SAFE
        }

        // Specular Trap veto B: the edge band is moving coplanar with the floor -> it's a
        // flat wet/glossy surface, not a void. This is the stronger signal (it's physics, not
        // photometry) so it clears POSSIBLE_DROP too. The barometer tie-break in MainActivity
        // still re-escalates if a real descent is measured.
        if (evidence.groundCoplanar >= COPLANAR_VETO) {
            return if (base == HazardState.DROP_CONFIRMED) HazardState.POSSIBLE_DROP else HazardState.SAFE
        }

        // Specular Trap veto A: a sharp luminance edge with NO chromaticity edge is a cast
        // shadow, not a step. Don't slam the emergency brake on it — downgrade a confirmation
        // to POSSIBLE_DROP (still a "careful" cue). A real material/geometry edge keeps a hue
        // step, so it never scores high here — the dark-asphalt-stairs case is safe.
        return if (base == HazardState.DROP_CONFIRMED && evidence.shadowLikelihood >= SHADOW_VETO)
            HazardState.POSSIBLE_DROP
        else base
    }
}

/**
 * V3 plan §8 — temporal hazard state machine, and §9 — approach-based urgency.
 *
 * State transitions (plan's initial values, explicitly "to tune from recorded walking clips,
 * not final constants" — kept as named constants for exactly that reason):
 *   - POSSIBLE_DROP after evidence in 2 of the last 3 frames
 *   - DROP_CONFIRMED after strong evidence in 3 of the last 5 frames
 *   - confirmation decays over [DECAY_MS] rather than clearing on one bad frame
 *   - during high rotation or a blocked/missing reading, HOLD the last state instead of
 *     dropping to SAFE (the plan's explicit "never a silent SAFE" requirement)
 *
 * Runs at whatever rate [update] is called — the plan suggests ~10Hz; this class is agnostic
 * to the caller's actual cadence, using real elapsed time (not frame count) for decay.
 */
class HazardStateMachine {
    private val history = ArrayDeque<RawEvidence>()
    private val edgeYHistory = ArrayDeque<Pair<Long, Float>>() // (timestampMs, firstEdgeY) for urgency

    private var state = HazardState.SAFE
    private var lastNonSafeAtMs = 0L
    private var lastUpdateAtMs = 0L

    fun reset() {
        history.clear()
        edgeYHistory.clear()
        state = HazardState.SAFE
        lastNonSafeAtMs = 0L
        lastUpdateAtMs = 0L
    }

    fun update(evidence: RawEvidence, nowMs: Long): HazardFrameOutput {
        if (evidence.sensorBlocked) {
            // "Invalid sensing becomes SENSOR_BLOCKED, never a silent SAFE" — per the plan,
            // this is a distinct terminal-ish state, not folded into the hold-last-state path.
            state = HazardState.SENSOR_BLOCKED
            lastUpdateAtMs = nowMs
            return HazardFrameOutput(
                state, 0f, null, 0f,
                ReasonBits(highRotation = evidence.highRotation, lowLight = evidence.lowLight,
                    proximityBlocked = false),
            )
        }

        history.addLast(evidence)
        while (history.size > HISTORY_SIZE) history.removeFirst()

        evidence.nearestEdgeY?.let { edgeYHistory.addLast(nowMs to it) }
        while (edgeYHistory.isNotEmpty() && nowMs - edgeYHistory.first().first > URGENCY_WINDOW_MS) {
            edgeYHistory.removeFirst()
        }

        val rawStates = history.map { HazardFusion.classifySingleFrame(it) }
        // SCENE_NOT_TRAVERSABLE is a structural fact about the current frame (a desk/table
        // fills the near field) — no temporal voting needed, and it must not count toward
        // POSSIBLE_DROP's "evidence in N of M frames" (it isn't drop evidence at all).
        val possibleCount = rawStates.count { it == HazardState.POSSIBLE_DROP || it == HazardState.DROP_CONFIRMED }
        val confirmedCount = rawStates.takeLast(5).count { it == HazardState.DROP_CONFIRMED }

        val candidateState = when {
            rawStates.last() == HazardState.SCENE_NOT_TRAVERSABLE -> HazardState.SCENE_NOT_TRAVERSABLE
            confirmedCount >= CONFIRM_FRAMES_OF_5 -> HazardState.DROP_CONFIRMED
            history.size >= 3 && possibleCount >= POSSIBLE_FRAMES_OF_3 -> HazardState.POSSIBLE_DROP
            else -> HazardState.SAFE
        }

        if (candidateState == HazardState.SCENE_NOT_TRAVERSABLE) {
            // Immediate, not gated by rotation-hold/decay — the walking surface being hidden
            // this frame is current-frame fact, not something to smooth over time.
            state = candidateState
        } else if (evidence.highRotation) {
            // Hold the prior state rather than switching to SAFE during fast rotation — the
            // plan's explicit guidance ("retain the prior hazard state briefly instead of
            // switching to safe"). Do NOT update lastNonSafeAtMs here, so decay still counts
            // down normally through the rotation event rather than being extended by it.
        } else if (candidateState != HazardState.SAFE) {
            state = candidateState
            lastNonSafeAtMs = nowMs
        } else if (state != HazardState.SAFE && nowMs - lastNonSafeAtMs < DECAY_MS) {
            // Decay window: keep the last non-safe state briefly rather than clearing on one
            // clean frame (plan: "decay confirmation over 0.5-1.0 seconds").
        } else {
            state = HazardState.SAFE
        }

        lastUpdateAtMs = nowMs

        val confidence = when (state) {
            HazardState.DROP_CONFIRMED -> (confirmedCount / 5f).coerceIn(0.6f, 1f)
            HazardState.POSSIBLE_DROP -> (possibleCount / 3f).coerceIn(0.3f, 0.7f)
            else -> 0f
        }

        val urgency = computeUrgency()
        val nearestEdge = history.lastOrNull { it.nearestEdgeY != null }?.nearestEdgeY

        val latest = history.last()
        return HazardFrameOutput(
            state = state,
            confidence = confidence,
            firstEdgeY = nearestEdge,
            urgency = urgency,
            reasonBits = ReasonBits(
                edgeLattice = latest.latticeScore >= 0.3f,
                depthSupport = latest.depthVerdict == DepthVerdict.SUPPORTS,
                lowLight = latest.lowLight,
                highRotation = latest.highRotation,
            ),
        )
    }

    /**
     * Plan §9 — urgency from the RATE the nearest edge approaches the bottom of frame (a
     * proxy for time-to-edge), not edge row alone. Deliberately does NOT claim a metric
     * distance/time — monocular depth isn't calibrated for that (see the plan's own caution).
     */
    private fun computeUrgency(): Float {
        if (edgeYHistory.size < 2) return 0f
        val (t0, y0) = edgeYHistory.first()
        val (t1, y1) = edgeYHistory.last()
        val dtSec = (t1 - t0) / 1000f
        if (dtSec < 0.05f) return 0f
        // Edge row increases (moves toward the bottom of frame, y=1) as the user approaches.
        val approachRate = (y1 - y0) / dtSec // frame-fractions per second
        // Combine with absolute proximity: an edge already near the bottom of frame (large y1)
        // is more urgent than one far away (small y1), even at the same approach rate.
        val proximityTerm = y1.coerceIn(0f, 1f)
        return (0.5f * proximityTerm + 0.5f * (approachRate / URGENCY_RATE_NORMALIZER).coerceIn(0f, 1f))
            .coerceIn(0f, 1f)
    }

    private companion object {
        const val HISTORY_SIZE = 5
        const val POSSIBLE_FRAMES_OF_3 = 2
        const val CONFIRM_FRAMES_OF_5 = 3
        const val DECAY_MS = 750L // plan's 0.5-1.0s range midpoint
        const val URGENCY_WINDOW_MS = 1500L
        const val URGENCY_RATE_NORMALIZER = 0.3f // frame-fractions/sec considered "fast approach"
    }
}
