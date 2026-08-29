package ai.secondsense.app.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import android.widget.RadioGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import ai.secondsense.app.R
import ai.secondsense.app.camera.FrameAnalyzer
import ai.secondsense.app.inference.FrameResult
import ai.secondsense.app.inference.TfliteInferenceEngine
import ai.secondsense.app.output.AudioOutput
import ai.secondsense.app.output.HapticOutput
import ai.secondsense.app.sonification.Spearcon
import java.util.concurrent.Executors
import kotlin.concurrent.thread

/**
 * DEBUG-ONLY screen: isolate and test each pipeline stage one at a time on-device, instead
 * of reading everything fused through MainActivity's production HUD. Requested after a
 * wrong-label report on-phone made it clear we needed to see YOLO alone, depth alone, and
 * each output channel alone to pin down where a bug actually is.
 *
 * Deliberately talks to [TfliteInferenceEngine] directly (not EngineConfig) — this screen
 * IS the tool for finding out if the tflite path itself is broken, so it must not silently
 * fall back to the mock the way the production engine selector does.
 *
 * Not reachable from outside the app; launched only from MainActivity's "Debug Panel" button.
 */
class DebugActivity : AppCompatActivity() {

    private lateinit var engine: TfliteInferenceEngine
    private lateinit var audio: AudioOutput
    private lateinit var haptics: HapticOutput
    private lateinit var spearcon: Spearcon
    private lateinit var analyzer: FrameAnalyzer
    private lateinit var boxOverlay: BoxOverlayView
    private lateinit var hud: android.widget.TextView

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    @Volatile private var frameCount = 0L

    private val requestCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_debug)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        boxOverlay = findViewById(R.id.boxOverlay)
        hud = findViewById(R.id.hud)

        audio = AudioOutput().also { it.initialize() }
        haptics = HapticOutput(this)
        spearcon = Spearcon(this).also { it.initialize() }

        engine = TfliteInferenceEngine(applicationContext)
        thread(name = "debug-engine-init") {
            try {
                engine.initialize()
            } catch (t: Throwable) {
                runOnUiThread {
                    Toast.makeText(this, "Engine init failed: ${t.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        analyzer = FrameAnalyzer(engine) { result -> onFrameResult(result) }

        findViewById<RadioGroup>(R.id.radioDebugMode).setOnCheckedChangeListener { _, checkedId ->
            engine.debugMode = when (checkedId) {
                R.id.radioYoloOnly -> TfliteInferenceEngine.DebugMode.YOLO_ONLY
                R.id.radioDepthOnly -> TfliteInferenceEngine.DebugMode.DEPTH_ONLY
                else -> TfliteInferenceEngine.DebugMode.FULL
            }
        }

        findViewById<com.google.android.material.materialswitch.MaterialSwitch>(R.id.switchCenterCropDbg)
            .setOnCheckedChangeListener { _, checked -> analyzer.centerCrop = checked }

        findViewById<android.view.View>(R.id.btnDbgTone).setOnClickListener { audio.testTone() }
        findViewById<android.view.View>(R.id.btnDbgBuzz).setOnClickListener { haptics.testBuzz() }
        findViewById<android.view.View>(R.id.btnDbgPanic).setOnClickListener { haptics.panic() }
        findViewById<android.view.View>(R.id.btnDbgDropOff).setOnClickListener { haptics.dropOff() }

        listOf(
            R.id.btnWordPerson to "person",
            R.id.btnWordDog to "dog",
            R.id.btnWordVehicle to "vehicle",
            R.id.btnWordChair to "chair",
        ).forEach { (id, word) ->
            findViewById<android.view.View>(id).setOnClickListener { playSpearconWord(word) }
        }

        findViewById<android.view.View>(R.id.btnBackToMain).setOnClickListener { finish() }

        if (hasCameraPermission()) startCamera()
        else requestCamera.launch(Manifest.permission.CAMERA)
    }

    private fun playSpearconWord(word: String) {
        val buf = spearcon.get(word)
        if (buf == null) {
            Toast.makeText(this, "\"$word\" still baking (on-device TTS) — tap again in a sec", Toast.LENGTH_SHORT).show()
        } else {
            audio.playMono(buf, pan = 0.5f)
        }
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val previewView = findViewById<androidx.camera.view.PreviewView>(R.id.preview)
            // BUG FIX: Preview and ImageAnalysis must share the same aspect ratio/FOV, or the
            // camera crops each stream differently and a box computed in analysis-frame
            // normalized coords lands in the wrong place over the preview. Same selector,
            // same 640x480 (4:3) target, for both streams.
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()
            val previewUseCase = Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(analysisExecutor, analyzer) }
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, previewUseCase, analysis)
            } catch (e: Exception) {
                Toast.makeText(this, "Camera bind failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFrameResult(result: FrameResult) {
        frameCount++
        boxOverlay.setDetections(result.detections)
        runOnUiThread {
            // The camera streams frames the instant the preview binds, but the model isn't
            // ready until NNAPI finishes compiling the graph (15-30s, one-time). Frames keep
            // arriving and incrementing frameCount the whole time (infer() just returns empty),
            // which is exactly what looked like "1000+ frames before it starts detecting" —
            // it wasn't broken, it just hadn't finished warming up yet. Say so explicitly.
            if (!engine.isReady) {
                hud.text = "⏳ warming up NNAPI (compiling model graph, ~15-30s one-time)…\n" +
                    "camera frames received: $frameCount   (not yet run through the model)"
                return@runOnUiThread
            }
            hud.text = buildString {
                append("mode: ${engine.debugMode}   frames: $frameCount   infer: ${result.inferenceMillis}ms\n")
                // V2's result.dropOff was removed (see MainActivity/TfliteInferenceEngine) —
                // V3's hazardState (below) is the only drop-off signal now.
                append("dets: ${result.detections.size}   depthAvail: ${result.depthAvailable}\n")
                if (result.debugRawCenterProximity != null) {
                    append(
                        "depth center: raw=%.3f  smoothed=%.3f  (watch smoothed hold steadier)\n"
                            .format(result.debugRawCenterProximity, result.debugSmoothedCenterProximity ?: 0f)
                    )
                }
                if (result.hazardState != null) {
                    append(
                        "hazard: %s  conf=%.2f  urgency=%.2f  edgeY=%s\n".format(
                            result.hazardState, result.hazardConfidence ?: 0f, result.hazardUrgency ?: 0f,
                            result.hazardFirstEdgeY?.let { "%.2f".format(it) } ?: "-",
                        )
                    )
                }
                if (result.debugEgoMotionX != null) {
                    append(
                        "ego-motion (camera pan): dx=%+.4f  dy=%+.4f  (should track YOUR panning, not object motion)\n"
                            .format(result.debugEgoMotionX, result.debugEgoMotionY ?: 0f)
                    )
                }
                if (result.detections.isEmpty()) {
                    append("(no detections this frame)")
                } else {
                    result.detections.take(6).forEachIndexed { i, d ->
                        append(
                            "${i + 1}. ${d.label ?: "(unknown)"}  s=${"%.2f".format(d.score)}  " +
                                "p=${"%.2f".format(d.proximity)}  " +
                                "moving=${if (d.moving) "YES" else "no"}  appr=${"%+.2f".format(d.approaching)}  " +
                                "box=[${"%.2f".format(d.box.left)}," +
                                "${"%.2f".format(d.box.top)},${"%.2f".format(d.box.right)}," +
                                "${"%.2f".format(d.box.bottom)}]\n"
                        )
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        audio.release()
        haptics.release()
        spearcon.release()
        engine.close()
    }
}
