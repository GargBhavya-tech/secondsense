package ai.secondsense.app.inference.decode

import android.graphics.Bitmap
import ai.secondsense.app.inference.Detection
import ai.secondsense.app.inference.SettledSighting
import ai.secondsense.app.sensors.ImuTracker

/**
 * The per-frame scene-analysis block that was duplicated in TfliteInferenceEngine and
 * QnnInferenceEngine — extracted so it lives ONCE. Both engines construct one of these and
 * call [analyze]; nothing about the result contract changes.
 *
 * Owns: camera ego-motion (Lucas-Kanade), the V3 drop-off hazard fusion (IMU corridor +
 * RGB edge lattice + depth-as-evidence + object-mask suppression -> HazardStateMachine),
 * and the coarse moving/approaching annotation.
 *
 * THROTTLE ([hazardEveryN]): the expensive parts — the RANSAC ego-motion estimate and the
 * Hough/RANSAC evidence gathering — run every Nth frame and the last result is reused in
 * between. The grayscale downsample and `prevGray` update happen EVERY frame (so the next
 * heavy estimate still compares adjacent frames), and [HazardStateMachine.update] is called
 * EVERY frame with the (possibly reused) evidence, so its time-based decay is never skipped.
 * hazardEveryN = 1 is the exact original behaviour.
 */
class SceneAnalyzer(
    private val imuTracker: ImuTracker? = null,
    private val hazardEveryN: Int = 1,
) {
    private val motion = MotionTracker()
    private val groundPlaneAnalyzer = GroundPlaneAnalyzer()
    private val hazardStateMachine = HazardStateMachine()
    // Object-memory support: rough metric distance + a "has it come to rest" gate. Always on
    // (cheap on the small depth grid); their output only matters once MainActivity wires the
    // memory feature, and is harmless otherwise.
    private val metricScaler = MetricDepthScaler()
    private val restingVerifier = RestingStateVerifier()

    private var prevGray: FloatArray? = null
    private var tick = 0L
    private var lastEvidence: RawEvidence? = null
    private var lastEgo: Pair<Float, Float> = 0f to 0f

    data class Result(
        val detections: List<Detection>,   // ego-motion-annotated (moving / approaching)
        val hazardState: HazardState?,
        val hazardConfidence: Float,
        val hazardUrgency: Float,
        val hazardFirstEdgeY: Float?,
        val egoMotionX: Float,
        val egoMotionY: Float,
        val settledObject: SettledSighting? = null,
    )

    fun analyze(frame: Bitmap, depthFrame: DepthSampler.Frame, detections: List<Detection>): Result {
        val curGray = OpticalFlow.toGrayscale(frame, GRAY_W, GRAY_H)
        val runHeavy = lastEvidence == null || (tick++ % hazardEveryN == 0L)

        val ego: Pair<Float, Float> = if (runHeavy) {
            prevGray?.let { pg ->
                val (dxPx, dyPx) = OpticalFlow.estimateEgoMotion(pg, curGray, GRAY_W, GRAY_H)
                (dxPx / GRAY_W) to (dyPx / GRAY_H)
            } ?: (0f to 0f)
        } else {
            lastEgo
        }
        lastEgo = ego
        prevGray = curGray

        val corridor = TraversableCorridor.from(
            imuTracker?.pitchDeg ?: 0f,
            imuTracker?.rollDeg ?: 0f,
        )

        val evidence: RawEvidence = if (runHeavy) {
            val lattice = EdgeLattice.detect(curGray, GRAY_W, GRAY_H, corridor)
            val (depthVerdict, _) = groundPlaneAnalyzer.depthEvidence(depthFrame, corridor)
            val edgeBand = lattice.nearestRowFraction?.let { (it - 0.05f) to (it + 0.05f) }
            val objectOverlap =
                edgeBand?.let { (lo, hi) -> GroundView.objectCoverage(detections, corridor, lo, hi) } ?: 0f
            val nearFieldY1 = corridor.y1 + 0.6f * (corridor.y2 - corridor.y1)
            val nearFieldObjectCoverage =
                GroundView.objectCoverage(detections, corridor, nearFieldY1, corridor.y2)
            RawEvidence(
                latticeScore = lattice.score,
                nearestEdgeY = lattice.nearestRowFraction,
                depthVerdict = depthVerdict,
                highRotation = imuTracker?.isHighRotation ?: false,
                lowLight = false,
                sensorBlocked = false,
                objectOverlap = objectOverlap,
                nearFieldObjectCoverage = nearFieldObjectCoverage,
            )
        } else {
            lastEvidence!!
        }
        lastEvidence = evidence

        val nowMs = System.currentTimeMillis()
        val hz = hazardStateMachine.update(evidence, nowMs)
        val annotated = motion.annotate(detections, ego)

        // Object-memory: keep the floor->metric fit fresh (throttled), tag each named
        // detection with a rough distance, and ask whether any has come to rest.
        if (runHeavy) {
            metricScaler.updateFloorFit(
                depthFrame, corridor,
                imuTracker?.pitchDeg ?: 0f, imuTracker?.rollDeg ?: 0f,
            )
        }
        val namedWithDist = annotated.mapNotNull { d ->
            if (d.label == null) null else d to metricScaler.metersForBox(depthFrame, d.box)
        }
        val settled = restingVerifier.update(namedWithDist, nowMs).firstOrNull()

        return Result(
            detections = annotated,
            hazardState = hz.state,
            hazardConfidence = hz.confidence,
            hazardUrgency = hz.urgency,
            hazardFirstEdgeY = hz.firstEdgeY,
            egoMotionX = ego.first,
            egoMotionY = ego.second,
            settledObject = settled,
        )
    }

    fun reset() {
        motion.reset()
        hazardStateMachine.reset()
        metricScaler.reset()
        restingVerifier.reset()
        prevGray = null
        tick = 0L
        lastEvidence = null
        lastEgo = 0f to 0f
    }

    private companion object {
        const val GRAY_W = 160
        const val GRAY_H = 120
    }
}
