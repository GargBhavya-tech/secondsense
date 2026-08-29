package ai.secondsense.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.secondsense.app.inference.decode.CocoLabels
import ai.secondsense.app.inference.decode.DepthSampler
import ai.secondsense.app.inference.decode.DepthTemporalSmoother
import ai.secondsense.app.inference.decode.DropOffDetector
import ai.secondsense.app.inference.decode.GroundPlaneAnalyzer
import ai.secondsense.app.inference.decode.MotionTracker
import ai.secondsense.app.inference.decode.OpticalFlow
import ai.secondsense.app.inference.decode.Preprocess
import ai.secondsense.app.inference.decode.RawTensor
import ai.secondsense.app.inference.decode.YoloDecoder
import ai.secondsense.app.inference.qnn.QnnBackend
import java.nio.ByteBuffer
import kotlin.math.abs

/**
 * The NPU-native engine: runs the Hexagon `qnn_context_binary` builds of YOLOv11 +
 * Depth-Anything-V2 through a [QnnBackend], entirely on-device. Behind the SAME
 * [InferenceEngine] seam as the mock and the TFLite engine, so MainActivity swaps by one
 * line ([EngineConfig.KIND]) and nothing downstream changes.
 *
 * THE POINT OF THE DESIGN: this class is a near mirror of [TfliteInferenceEngine]. The ONLY
 * difference is the "run the model" lines — TFLite calls an Interpreter; this calls
 * [QnnBackend.run]. Everything else — letterboxing, YOLO decode + NMS, depth→proximity,
 * RED-tier honesty, motion, drop-off, center-crop — is the identical shared decode layer.
 * That is exactly what the whole InferenceEngine/decode split was built to buy: when the
 * iQOO 15's Hexagon binaries land, only [QnnBackend] gets a native implementation.
 *
 * MODELS: place the exports at
 *   app/src/main/assets/models/yolov11_det.bin
 *   app/src/main/assets/models/depth_anything_v2.bin
 * Until the native bridge exists, [QnnBackend.isReady] is false and [EngineConfig] never
 * selects this engine — it falls back to TFLite/MOCK. So this compiles and is wired today,
 * and goes live the moment the bridge + binaries are present.
 */
class QnnInferenceEngine(
    private val context: Context,
    private val backend: QnnBackend,
    private val yoloAsset: String = "models/yolov11_det.bin",
    private val depthAsset: String = "models/depth_anything_v2.bin",
    // Input side lengths come from the model metadata; these are the qai-hub export defaults.
    private val yoloInputSize: Int = 640,
    private val depthInputSize: Int = 518,
    private val confThreshold: Float = 0.35f,
    private val iouThreshold: Float = 0.50f,
    private val redProximityFloor: Float = 0.80f,
    // QNN context binaries are frequently NCHW; flip if the export metadata says NHWC.
    private val channelsFirst: Boolean = true,
) : InferenceEngine {

    override val name: String = "qnn:yolov11+depth"

    private val depthSampler = DepthSampler()
    private val motion = MotionTracker()
    private val dropOffDetector = DropOffDetector()
    private val groundPlaneAnalyzer = GroundPlaneAnalyzer()
    private val depthSmoother = DepthTemporalSmoother()
    private var prevGray: FloatArray? = null

    @Volatile private var ready = false

    override fun initialize() {
        if (ready) return
        // NOTE: do not gate on backend.isReady() here — for NativeQnnBackend, isReady() only
        // becomes true as a SIDE EFFECT of the first load() call (nativeInit is lazy, inside
        // load()). Checking it first would be a permanent deadlock: isReady() stays false
        // forever because load() — the only thing that could flip it — never runs.
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

        // --- YOLO (the ONLY runtime-specific lines vs TFLite) ---
        val lb = Preprocess.letterbox(frame, yoloInputSize, normalizeTo01 = true, channelsFirst = channelsFirst)
        val yoloTensors = backend.run("yolo", lb.buffer)
        val rawDets = YoloDecoder.decode(yoloTensors, lb, confThreshold, iouThreshold)

        // --- Depth ---
        val depthLb = Preprocess.letterbox(frame, depthInputSize, normalizeTo01 = true, channelsFirst = channelsFirst)
        val depthTensor = backend.run("depth", depthLb.buffer).first()
        val smoothedDepth = RawTensor(depthSmoother.smooth(depthTensor.data), depthTensor.shape)
        val depthFrame = depthSampler.parse(smoothedDepth)
        val dropOffEdge = dropOffDetector.detect(depthFrame)
        val groundEdge = groundPlaneAnalyzer.detect(depthFrame)

        // --- fuse: identical to the TFLite path ---
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

        val curGray = OpticalFlow.toGrayscale(frame, FLOW_GRAY_W, FLOW_GRAY_H)
        val egoMotionNormalized = prevGray?.let { pg ->
            val (dxPx, dyPx) = OpticalFlow.estimateEgoMotion(pg, curGray, FLOW_GRAY_W, FLOW_GRAY_H)
            (dxPx / FLOW_GRAY_W) to (dyPx / FLOW_GRAY_H)
        } ?: (0f to 0f)
        prevGray = curGray

        detections = motion.annotate(detections, egoMotionNormalized)
        val visible = if (centerCrop) detections.filter { abs(it.box.centerX - 0.5f) <= 0.15f } else detections

        val elapsedMs = (System.nanoTime() - started) / 1_000_000
        return FrameResult(
            detections = visible,
            frameWidth = frame.width,
            frameHeight = frame.height,
            inferenceMillis = elapsedMs,
            depthAvailable = depthFrame.valid,
            dropOff = dropOffEdge != null || groundEdge != null,
            dropOffRowFraction = dropOffEdge?.rowFraction ?: groundEdge?.rowFraction,
        )
    }

    override fun close() {
        ready = false
        motion.reset()
        depthSmoother.reset()
        prevGray = null
        backend.close()
    }

    /** True if the QNN path can actually run right now (native bridge up + binaries present). */
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
