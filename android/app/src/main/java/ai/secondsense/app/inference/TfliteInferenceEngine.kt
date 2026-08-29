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
import ai.secondsense.app.inference.decode.SceneAnalyzer
import ai.secondsense.app.inference.decode.TraversableCorridor
import ai.secondsense.app.inference.decode.YoloDecoder
import ai.secondsense.app.sensors.ImuTracker
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * The real-model TEST-PATH engine: runs YOLOv11 + Depth-Anything-V2 as .tflite on
 * CPU/XNNPACK (or GPU delegate, opt-in), entirely on-device. Behind the SAME
 * [InferenceEngine] seam as the mock, so MainActivity swaps by one line and the whole
 * sonification stack downstream is untouched.
 *
 * DIVISION OF LABOR (the point of this design): this class does only the runtime-specific
 * part — load the .tflite, run it, extract raw float tensors. Everything after that is the
 * SHARED decode layer (Preprocess / YoloDecoder / DepthSampler / MotionTracker), which the
 * future QnnInferenceEngine will call identically. So when the NPU binaries land, that
 * engine reuses this exact fusion logic; only the "run the model" lines differ.
 *
 * MODELS: place the exports at
 *   app/src/main/assets/models/yolov11_det.tflite
 *   app/src/main/assets/models/depth_anything_v2.tflite
 * and add `androidResources { noCompress += "tflite" }` (see build.gradle) so they can be
 * memory-mapped. If a model is missing, initialize() throws with a clear message; the
 * EngineConfig factory falls back to the mock so the app still runs.
 *
 * SHAPE-GUARD: initialize() logs every input/output tensor's shape + dtype under the tag
 * "SecondSense/tflite". Watch logcat on the first run — those lines tell you the true YOLO
 * output layout and depth map dims, which is how you confirm (or adjust) the decoder
 * assumptions without the model in hand.
 */
class TfliteInferenceEngine(
    private val context: Context,
    /** V3 drop-off plan §1/§2 — optional so every existing construction site (and MockEngine-
     * style tests) keeps working; without it the hazard fusion pipeline uses a fixed default
     * corridor (TraversableCorridor.DEFAULT) instead of an IMU-stabilized one. */
    private val imuTracker: ImuTracker? = null,
    private val yoloAsset: String = "models/yolov11_det.tflite",
    private val depthAsset: String = "models/depth_anything_v2.tflite",
    private val useGpu: Boolean = false,        // opt-in; see build.gradle for the dep + import
    private val numThreads: Int = 4,
    // Lowered from 0.35 after swapping in yolo26s (see debug_yolo.py findings): the new
    // model produces meaningfully higher, better-calibrated scores on real test photos
    // (0.34-0.57 for a correctly-identified bottle vs 0.07-0.34 from the old yolov11n),
    // so 0.30 clears real detections while still rejecting background noise.
    private val confThreshold: Float = 0.30f,
    private val iouThreshold: Float = 0.50f,
    private val redProximityFloor: Float = 0.80f, // depth-only RED trigger (matches PANIC_PROXIMITY)
) : InferenceEngine {

    override val name: String = "tflite:yolov11+depth"
    override val isReady: Boolean get() = ready

    private var yolo: Interpreter? = null
    private var depth: Interpreter? = null

    private var yoloInputSize = 640
    private val depthSampler = DepthSampler()
    // Research candidate (secondsense_research_candidates_v1.md §2) — EMA-smooths the raw
    // depth map across consecutive depth-inference runs before anything else touches it.
    private val depthSmoother = DepthTemporalSmoother()
    // Shared with QnnInferenceEngine: per-frame ego-motion (Lucas-Kanade) + the V3 drop-off
    // hazard fusion (EdgeLattice + GroundPlaneAnalyzer.depthEvidence + object-mask suppression
    // -> HazardStateMachine) + the coarse moving/approaching annotation. hazardEveryN=2 runs
    // the expensive RANSAC/Hough parts every other frame; the state machine still ticks every
    // frame so its time-based decay is never skipped.
    private val sceneAnalyzer = SceneAnalyzer(imuTracker, hazardEveryN = 2)

    @Volatile private var ready = false

    // reusable output holders, sized at init from the real tensor shapes
    private lateinit var yoloOutBuffers: Map<Int, ByteBuffer>
    private lateinit var yoloOutShapes: List<IntArray>
    private lateinit var yoloOutTypes: List<DataType>
    private lateinit var depthOutBuffer: ByteBuffer
    private lateinit var depthOutShape: IntArray
    private var frameLog = 0L

    // PERF: try the NNAPI hardware delegate (offloads to the phone's NPU/GPU/DSP) with a
    // clean fall-back to CPU/XNNPACK. Kept so close() can release native handles.
    // PER-MODEL, not one shared flag: NNAPI on this test phone (MediaTek) saturates every
    // YOLO score at exactly 1.00 on the bigger yolo26s graph (confirmed: CPU/XNNPACK gives
    // correct 0.3-0.7 scores matching the offline debug_yolo.py reference; NNAPI does not —
    // a vendor NNAPI op-compatibility bug, not a decode bug). Depth was NEVER part of that
    // bug (unrelated graph, never touched), so it stays on NNAPI for speed. On the real
    // target (iQOO 15 / Snapdragon 8 Elite), the plan is to bypass generic NNAPI entirely
    // via the native QNN path (QnnInferenceEngine) — Qualcomm's own cloud profiling already
    // showed this exact yolo26s model running in 5.3ms with clean accuracy on that chipset,
    // so this CPU fallback is a TEST-PHONE-ONLY workaround, not the production plan.
    private val useNnapiYolo = false
    private val useNnapiDepth = true
    private val nnapiDelegates = mutableListOf<NnApiDelegate>()

    // PERF: depth is the heavy model; run it every Nth frame and reuse the last map in
    // between so YOLO (detection + pan) stays responsive. Proximity changes slowly enough
    // that reusing a 1-frame-old depth map is imperceptible.
    @Volatile private var depthEveryN = 2
    private var depthTick = 0L
    private var lastDepthFrame: DepthSampler.Frame? = null
    private var lastDebugRawCenterProx: Float? = null
    private var lastDebugSmoothedCenterProx: Float? = null

    // Ego-motion (Lucas-Kanade) state: previous downsampled grayscale frame. Small resolution
    // keeps flow tracking cheap — validated offline that this is plenty for frame-to-frame
    // ego-motion estimation at normal walking pace.
    private var prevGray: FloatArray? = null

    /**
     * DEBUG ONLY (DebugActivity): isolate one model at a time so a bad label or a bad
     * proximity reading can be pinned to YOLO vs depth instead of guessed at from the
     * fused output. Public + volatile so the debug UI can flip it live. Defaults to FULL
     * so nothing about the normal app path changes.
     */
    enum class DebugMode { FULL, YOLO_ONLY, DEPTH_ONLY }
    @Volatile var debugMode: DebugMode = DebugMode.FULL

    override fun initialize() {
        if (ready) return

        yolo = buildInterpreter(loadModel(yoloAsset), "yolo", useNnapiYolo)
        depth = buildInterpreter(loadModel(depthAsset), "depth", useNnapiDepth)

        // YOLO input side length from the real input tensor (square assumed).
        yolo!!.getInputTensor(0).shape().let { s ->
            // NHWC [1,H,W,3] or NCHW [1,3,H,W]; take the spatial dim.
            yoloInputSize = if (s.size == 4 && s[3] == 3) s[1] else if (s.size == 4) s[2] else 640
        }

        // Pre-size YOLO output byte buffers from the REAL per-tensor byte size + dtype. The
        // classes tensor is UINT8 (1 byte/elem), not FLOAT32 — sizing everything as *4 and
        // reading as floats mangles the class indices, so use tensor.numBytes()/dataType().
        val yCount = yolo!!.outputTensorCount
        yoloOutShapes = (0 until yCount).map { yolo!!.getOutputTensor(it).shape() }
        yoloOutTypes = (0 until yCount).map { yolo!!.getOutputTensor(it).dataType() }
        yoloOutBuffers = (0 until yCount).associateWith { i ->
            ByteBuffer.allocateDirect(yolo!!.getOutputTensor(i).numBytes()).order(ByteOrder.nativeOrder())
        }

        depthOutShape = depth!!.getOutputTensor(0).shape()
        depthOutBuffer = ByteBuffer.allocateDirect(depth!!.getOutputTensor(0).numBytes())
            .order(ByteOrder.nativeOrder())

        ready = true
        Log.i(TAG, "ready. yoloInput=$yoloInputSize outShapes=${yoloOutShapes.joinToString { it.joinToString("x") }} " +
            "depthShape=${depthOutShape.joinToString("x")}")
    }

    override fun infer(frame: Bitmap, centerCrop: Boolean): FrameResult {
        // Not ready yet (models still loading on the init thread): return an empty,
        // non-crashing frame so the camera loop and mock-free warmup behave.
        if (!ready) {
            return FrameResult(emptyList(), frame.width, frame.height, 0L, depthAvailable = false)
        }
        val y = yolo!!; val dpt = depth!!
        val started = System.nanoTime()

        // --- YOLO --- (skipped in DEPTH_ONLY debug mode)
        val rawDets = if (debugMode == DebugMode.DEPTH_ONLY) {
            emptyList()
        } else {
            val lb = Preprocess.letterbox(frame, yoloInputSize, normalizeTo01 = true)
            yoloOutBuffers.values.forEach { it.rewind() }
            y.runForMultipleInputsOutputs(arrayOf<Any>(lb.buffer), yoloOutBuffers.mapValues { it.value as Any })
            val yoloTensors = yoloOutShapes.mapIndexed { i, shape -> toRawTensor(yoloOutBuffers[i]!!, shape, yoloOutTypes[i]) }
            YoloDecoder.decode(yoloTensors, lb, confThreshold, iouThreshold)
        }

        // --- Depth (heavy: run every Nth frame, reuse the last map in between) ---
        // Skipped entirely in YOLO_ONLY debug mode — an invalid Frame (hi==lo) makes
        // proximityFor() return a neutral 0.5f everywhere, so downstream code needs no
        // special-casing; it just sees "depth unavailable" honestly.
        val runDepth = debugMode != DebugMode.YOLO_ONLY &&
            (lastDepthFrame == null || (depthTick++ % depthEveryN.coerceAtLeast(1) == 0L))
        val depthFrame: DepthSampler.Frame
        if (debugMode == DebugMode.YOLO_ONLY) {
            depthFrame = DepthSampler.Frame(FloatArray(0), 0, 0, 0f, 0f) // invalid: hi==lo
            lastDebugRawCenterProx = null
            lastDebugSmoothedCenterProx = null
            depthSmoother.reset() // depth is paused in this debug mode; don't blend stale state on resume
        } else if (runDepth) {
            val depthLb = Preprocess.letterbox(frame, depthInputSize(dpt), normalizeTo01 = true)
            depthOutBuffer.rewind()
            dpt.run(depthLb.buffer, depthOutBuffer)
            val rawDepth = toRawTensor(depthOutBuffer, depthOutShape, DataType.FLOAT32)
            // DEBUG/VALIDATION: sample center proximity from the UNSMOOTHED map too, so
            // DebugActivity can show raw-vs-smoothed side by side and the smoothing effect
            // is directly observable, not just claimed.
            val rawFrameForDebug = depthSampler.parse(rawDepth)
            lastDebugRawCenterProx = depthSampler.proximityFor(rawFrameForDebug, BBox(0.42f, 0.42f, 0.58f, 0.58f))
            // Smooth BEFORE parse(), so DepthSampler's percentile normalization and every
            // downstream depth consumer (GroundPlaneAnalyzer.depthEvidence, V3) see the
            // cleaned signal, not raw noise.
            val smoothedDepth = RawTensor(depthSmoother.smooth(rawDepth.data), rawDepth.shape)
            depthFrame = depthSampler.parse(smoothedDepth)
            lastDebugSmoothedCenterProx = depthSampler.proximityFor(depthFrame, BBox(0.42f, 0.42f, 0.58f, 0.58f))
            lastDepthFrame = depthFrame
        } else {
            depthFrame = lastDepthFrame!!
        }

        // --- fuse: attach proximity + tier-eligible label to each detection ---
        var detections = rawDets.map { rd ->
            val cocoName = CocoLabels.nameForIndex(rd.cocoIndex)
            val label = cocoName?.let { CocoLabels.toIconVocab(it) }
            Detection(
                label = label,
                score = rd.score,
                box = rd.box,
                proximity = depthSampler.proximityFor(depthFrame, rd.box),
                // approaching/moving filled by MotionTracker below.
                tier = ConfidenceTier.WHITE, // real tier is DERIVED downstream by TierClassifier
            )
        }

        // RED-tier honesty (§5.3): if YOLO saw nothing but depth shows something close in
        // the center, emit ONE identity-less detection so the system says "something's
        // there, I can't name it" instead of going silent.
        if (detections.isEmpty()) {
            depthSampler.nearestCenterRegion(depthFrame, redProximityFloor)?.let { (box, prox) ->
                detections = listOf(
                    Detection(label = null, score = 0f, box = box, proximity = prox, tier = ConfidenceTier.RED)
                )
            }
        }

        // Per-frame ego-motion (Lucas-Kanade) + V3 drop-off hazard fusion + coarse
        // moving/approaching annotation — shared with QnnInferenceEngine, throttled inside.
        val scene = sceneAnalyzer.analyze(frame, depthFrame, detections)
        detections = scene.detections

        // center-crop filter — mirror the mock/contract: flow mode keeps only centered.
        val visible = if (centerCrop) detections.filter { abs(it.box.centerX - 0.5f) <= 0.15f } else detections

        val elapsedMs = (System.nanoTime() - started) / 1_000_000
        // Throttled on-device sanity log: confirms the full pipeline runs and class indices
        // decode sanely (a real label, not garbage) now that dtype is handled.
        if (++frameLog % 30 == 0L) {
            val detSummary = visible.take(5).joinToString(" | ") {
                "${it.label ?: "unk"} s=${"%.2f".format(it.score)} prox=${"%.2f".format(it.proximity)}"
            }.ifEmpty { "(none)" }
            Log.i(TAG, "frame#$frameLog dets=${visible.size} hazard=${scene.hazardState} ${elapsedMs}ms :: $detSummary")
        }
        return FrameResult(
            detections = visible,
            frameWidth = frame.width,
            frameHeight = frame.height,
            inferenceMillis = elapsedMs,
            depthAvailable = depthFrame.valid,
            // V2's dropOff/dropOffRowFraction (defaults false/null) are no longer computed —
            // removed per explicit user request; V3's hazardState/etc below are now the sole
            // drop-off signal. See the class-level doc comment on groundPlaneAnalyzer.
            debugRawCenterProximity = lastDebugRawCenterProx,
            debugSmoothedCenterProximity = lastDebugSmoothedCenterProx,
            debugEgoMotionX = scene.egoMotionX,
            debugEgoMotionY = scene.egoMotionY,
            hazardState = scene.hazardState,
            hazardConfidence = scene.hazardConfidence,
            hazardUrgency = scene.hazardUrgency,
            hazardFirstEdgeY = scene.hazardFirstEdgeY,
            settledObject = scene.settledObject,
            cameraHealth = scene.cameraHealth,
        )
    }

    override fun close() {
        ready = false
        yolo?.close(); yolo = null
        depth?.close(); depth = null
        nnapiDelegates.forEach { runCatching { it.close() } }
        nnapiDelegates.clear()
        lastDepthFrame = null
        depthSmoother.reset()
        sceneAnalyzer.reset()
    }

    override fun setDepthEveryN(n: Int) { depthEveryN = n.coerceIn(1, 12) }
    override fun setHazardEveryN(n: Int) { sceneAnalyzer.hazardEveryN = n.coerceIn(1, 12) }

    // ---- helpers -----------------------------------------------------------

    /**
     * Build an Interpreter, preferring the NNAPI hardware delegate (NPU/GPU/DSP) and falling
     * back to CPU/XNNPACK if the device/model rejects it — so a flaky vendor NNAPI never
     * breaks the demo, it just runs slower. Logs which path each model took.
     */
    private fun buildInterpreter(model: ByteBuffer, tag: String, useNnapi: Boolean): Interpreter {
        if (useNnapi) {
            try {
                val delegate = NnApiDelegate()
                val opts = Interpreter.Options().apply { setNumThreads(numThreads); addDelegate(delegate) }
                val itp = Interpreter(model, opts)
                nnapiDelegates.add(delegate)
                Log.i(TAG, "$tag: using NNAPI hardware delegate")
                logIo(tag, itp)
                return itp
            } catch (t: Throwable) {
                Log.w(TAG, "$tag: NNAPI delegate failed (${t.message}); falling back to CPU/XNNPACK")
            }
        }
        val opts = Interpreter.Options().apply { setNumThreads(numThreads) }
        val itp = Interpreter(model, opts)
        Log.i(TAG, "$tag: using CPU/XNNPACK ($numThreads threads)")
        logIo(tag, itp)
        return itp
    }

    private fun depthInputSize(dpt: Interpreter): Int {
        val s = dpt.getInputTensor(0).shape()
        return if (s.size == 4 && s[3] == 3) s[1] else if (s.size == 4) s[2] else 518
    }

    /**
     * Read a directly-allocated output buffer into a flat float RawTensor, honoring the real
     * tensor dtype. YOLO's class tensor is UINT8; scores/boxes/depth are FLOAT32 — reading a
     * UINT8 tensor as floats (the old behavior) produced garbage class indices.
     */
    private fun toRawTensor(buf: ByteBuffer, shape: IntArray, dtype: DataType): RawTensor {
        buf.rewind()
        val n = numElements(shape)
        val out = FloatArray(n)
        when (dtype) {
            DataType.FLOAT32 -> buf.asFloatBuffer().get(out)
            DataType.UINT8 -> for (i in 0 until n) out[i] = (buf.get().toInt() and 0xFF).toFloat()
            DataType.INT8 -> for (i in 0 until n) out[i] = buf.get().toFloat()
            DataType.INT32 -> { val ib = buf.asIntBuffer(); for (i in 0 until n) out[i] = ib.get().toFloat() }
            DataType.INT64 -> { val lb = buf.asLongBuffer(); for (i in 0 until n) out[i] = lb.get().toFloat() }
            else -> buf.asFloatBuffer().get(out) // best-effort for any other float-like type
        }
        return RawTensor(out, shape)
    }

    private fun numElements(shape: IntArray): Int = shape.fold(1) { a, b -> a * b }

    private fun loadModel(asset: String): ByteBuffer {
        context.assets.openFd(asset).use { fd ->
            fd.createInputStream().channel.use { ch ->
                return ch.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    private fun logIo(tag: String, itp: Interpreter) {
        for (i in 0 until itp.inputTensorCount) {
            val t = itp.getInputTensor(i)
            Log.i(TAG, "$tag input[$i] ${t.shape().joinToString("x")} ${t.dataType()}")
        }
        for (i in 0 until itp.outputTensorCount) {
            val t = itp.getOutputTensor(i)
            Log.i(TAG, "$tag output[$i] ${t.shape().joinToString("x")} ${t.dataType()}")
        }
    }

    private companion object {
        const val TAG = "SecondSense/tflite"
        const val FLOW_GRAY_W = 160
        const val FLOW_GRAY_H = 120
    }
}
