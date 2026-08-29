package ai.secondsense.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.secondsense.app.inference.decode.CocoLabels
import ai.secondsense.app.inference.decode.DepthSampler
import ai.secondsense.app.inference.decode.DepthTemporalSmoother
import ai.secondsense.app.inference.decode.EdgeLattice
import ai.secondsense.app.inference.decode.GroundPlaneAnalyzer
import ai.secondsense.app.inference.decode.GroundView
import ai.secondsense.app.inference.decode.HazardStateMachine
import ai.secondsense.app.inference.decode.MotionTracker
import ai.secondsense.app.inference.decode.OpticalFlow
import ai.secondsense.app.inference.decode.Preprocess
import ai.secondsense.app.inference.decode.RawEvidence
import ai.secondsense.app.inference.decode.RawTensor
import ai.secondsense.app.inference.decode.TraversableCorridor
import ai.secondsense.app.inference.decode.YoloDecoder
import ai.secondsense.app.inference.qnn.QnnBackend
import ai.secondsense.app.sensors.ImuTracker
import java.nio.ByteBuffer
import kotlin.math.abs

/**
 * The NPU-native engine: runs the Hexagon `qnn_context_binary` builds of YOLO26 +
 * Depth-Anything-V2 through a [QnnBackend], entirely on-device. Behind the SAME
 * [InferenceEngine] seam as the mock and the TFLite engine.
 *
 * A near mirror of [TfliteInferenceEngine] — the ONLY runtime-specific lines are the
 * `backend.run(...)` calls. Everything after (letterbox, YOLO decode + NMS, depth->proximity,
 * RED-tier honesty, ego-motion, V3 hazard fusion, center-crop) is the identical shared
 * decode layer, so the two engines produce the same `FrameResult` shape and MainActivity
 * treats them identically.
 *
 * INPUT LAYOUT: the qai_hub_models QNN exports are **NHWC** `[1,H,W,3]`, value range 0..1
 * (confirmed against each .bin's metadata.json) — same as the TFLite exports. Feeding NCHW
 * scrambles the input and tanks detection confidence, so [channelsFirst] defaults false.
 */
class QnnInferenceEngine(
    private val context: Context,
    private val backend: QnnBackend,
    private val imuTracker: ImuTracker? = null,
    private val yoloAsset: String = "models/yolov11_det.bin",
    private val depthAsset: String = "models/depth_anything_v2.bin",
    private val yoloInputSize: Int = 640,
    private val depthInputSize: Int = 518,
    // Matches TfliteInferenceEngine's yolo26s-tuned value (0.34-0.57 on real photos).
    private val confThreshold: Float = 0.30f,
    private val iouThreshold: Float = 0.50f,
    private val redProximityFloor: Float = 0.80f,
    // qai_hub_models QNN exports are NHWC (see class doc) — do NOT set true for these binaries.
    private val channelsFirst: Boolean = false,
) : InferenceEngine {

    override val name: String = "qnn:yolov11+depth"
    override val isReady: Boolean get() = ready

    private val depthSampler = DepthSampler()
    private val motion = MotionTracker()
    private val groundPlaneAnalyzer = GroundPlaneAnalyzer()
    private val depthSmoother = DepthTemporalSmoother()
    private val hazardStateMachine = HazardStateMachine()
    private var prevGray: FloatArray? = null

    @Volatile private var ready = false

    override fun initialize() {
        if (ready) return
        val yoloOk = backend.load("yolo", readAsset(yoloAsset))
        val depthOk = backend.load("depth", readAsset(depthAsset))
        ready = yoloOk && depthOk
        if (!ready) {
            Log.w(TAG, "QNN backend init/load failed (${backend.name}); engine will return empty frames")
        }
        Log.i(TAG, "initialize: yolo=$yoloOk depth=$depthOk ready=$ready")
    }

    override fun infer(frame: Bitmap, centerCrop: Boolean): FrameResult {
        if (!ready) {
            return FrameResult(emptyList(), frame.width, frame.height, 0L, depthAvailable = false)
        }
        val started = System.nanoTime()

        // --- YOLO (runtime-specific) ---
        val lb = Preprocess.letterbox(frame, yoloInputSize, normalizeTo01 = true, channelsFirst = channelsFirst)
        val yoloTensors = backend.run("yolo", lb.buffer)
        val rawDets = YoloDecoder.decode(yoloTensors, lb, confThreshold, iouThreshold)

        // --- Depth (runtime-specific) ---
        val depthLb = Preprocess.letterbox(frame, depthInputSize, normalizeTo01 = true, channelsFirst = channelsFirst)
        val depthTensor = backend.run("depth", depthLb.buffer).first()
        val smoothedDepth = RawTensor(depthSmoother.smooth(depthTensor.data), depthTensor.shape)
        val depthFrame = depthSampler.parse(smoothedDepth)

        // --- fuse: attach proximity + tier-eligible label to each detection ---
        var detections = rawDets.map { rd ->
            val cocoName = CocoLabels.nameForIndex(rd.cocoIndex)
            val label = cocoName?.let { CocoLabels.toIconVocab(it) }
            Detection(
                label = label,
                score = rd.score,
                box = rd.box,
                proximity = depthSampler.proximityFor(depthFrame, rd.box),
                tier = ConfidenceTier.WHITE, // real tier DERIVED downstream by TierClassifier
            )
        }

        // RED-tier honesty (§5.3): depth sees something close but YOLO named nothing.
        if (detections.isEmpty()) {
            depthSampler.nearestCenterRegion(depthFrame, redProximityFloor)?.let { (box, prox) ->
                detections = listOf(
                    Detection(label = null, score = 0f, box = box, proximity = prox, tier = ConfidenceTier.RED)
                )
            }
        }

        // ego-motion (Lucas-Kanade) — reused by hazard fusion below.
        val curGray = OpticalFlow.toGrayscale(frame, FLOW_GRAY_W, FLOW_GRAY_H)
        val egoMotionNormalized = prevGray?.let { pg ->
            val (dxPx, dyPx) = OpticalFlow.estimateEgoMotion(pg, curGray, FLOW_GRAY_W, FLOW_GRAY_H)
            (dxPx / FLOW_GRAY_W) to (dyPx / FLOW_GRAY_H)
        } ?: (0f to 0f)
        prevGray = curGray

        // --- V3 drop-off / hazard fusion — IDENTICAL to TfliteInferenceEngine so MainActivity
        // (which reads only FrameResult.hazardState) behaves the same on both engines. ---
        val corridor = TraversableCorridor.from(
            imuTracker?.pitchDeg ?: 0f,
            imuTracker?.rollDeg ?: 0f,
        )
        val latticeResult = EdgeLattice.detect(curGray, FLOW_GRAY_W, FLOW_GRAY_H, corridor)
        val (depthVerdict, _) = groundPlaneAnalyzer.depthEvidence(depthFrame, corridor)
        val edgeBand = latticeResult.nearestRowFraction?.let { (it - 0.05f) to (it + 0.05f) }
        val objectOverlap = edgeBand?.let { (lo, hi) -> GroundView.objectCoverage(detections, corridor, lo, hi) } ?: 0f
        val nearFieldY1 = corridor.y1 + 0.6f * (corridor.y2 - corridor.y1)
        val nearFieldObjectCoverage = GroundView.objectCoverage(detections, corridor, nearFieldY1, corridor.y2)
        val hazardOutput = hazardStateMachine.update(
            RawEvidence(
                latticeScore = latticeResult.score,
                nearestEdgeY = latticeResult.nearestRowFraction,
                depthVerdict = depthVerdict,
                highRotation = imuTracker?.isHighRotation ?: false,
                lowLight = false,
                sensorBlocked = false,
                objectOverlap = objectOverlap,
                nearFieldObjectCoverage = nearFieldObjectCoverage,
            ),
            System.currentTimeMillis(),
        )

        detections = motion.annotate(detections, egoMotionNormalized)
        val visible = if (centerCrop) detections.filter { abs(it.box.centerX - 0.5f) <= 0.15f } else detections

        val elapsedMs = (System.nanoTime() - started) / 1_000_000
        return FrameResult(
            detections = visible,
            frameWidth = frame.width,
            frameHeight = frame.height,
            inferenceMillis = elapsedMs,
            depthAvailable = depthFrame.valid,
            hazardState = hazardOutput.state,
            hazardConfidence = hazardOutput.confidence,
            hazardUrgency = hazardOutput.urgency,
            hazardFirstEdgeY = hazardOutput.firstEdgeY,
        )
    }

    override fun close() {
        ready = false
        motion.reset()
        depthSmoother.reset()
        hazardStateMachine.reset()
        prevGray = null
        backend.close()
    }

    fun isOperational(): Boolean = ready

    private fun readAsset(path: String): ByteBuffer {
        context.assets.openFd(path).use { fd ->
            fd.createInputStream().channel.use { ch ->
                return ch.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    private companion object {
        const val TAG = "SecondSense/qnn"
        const val FLOW_GRAY_W = 160
        const val FLOW_GRAY_H = 120
    }
}
