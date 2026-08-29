package ai.secondsense.app.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Size
import android.view.GestureDetector
import android.view.MotionEvent
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
import ai.secondsense.app.audio.HazardSoundDetector
import ai.secondsense.app.camera.FrameAnalyzer
import ai.secondsense.app.inference.decode.DetectionStabilizer
import ai.secondsense.app.perception.LanguagePrefs
import ai.secondsense.app.perception.MlKitPerception
import ai.secondsense.app.perception.OcrTranslator
import ai.secondsense.app.memory.DeadReckoner
import ai.secondsense.app.memory.MemoryPhrase
import ai.secondsense.app.memory.ObjectMemory
import ai.secondsense.app.sensors.PedometerTracker
import ai.secondsense.app.perf.PerfPolicy
import ai.secondsense.app.perf.ThermalGovernor
import ai.secondsense.app.sonification.CueTarget
import ai.secondsense.app.sonification.ObstacleHabituation
import java.util.Locale
import ai.secondsense.app.voice.SceneNarrator
import ai.secondsense.app.dashboard.DashboardServer
import ai.secondsense.app.dashboard.QrCodeGenerator
import ai.secondsense.app.databinding.ActivityMainBinding
import ai.secondsense.app.sensors.BarometerMonitor
import org.json.JSONArray
import org.json.JSONObject
import ai.secondsense.app.inference.CameraHealth
import ai.secondsense.app.inference.EngineConfig
import ai.secondsense.app.inference.FrameResult
import ai.secondsense.app.inference.InferenceEngine
import ai.secondsense.app.output.AudioOutput
import ai.secondsense.app.output.HapticOutput
import ai.secondsense.app.sonification.CueEngine
import ai.secondsense.app.sonification.ModeController
import ai.secondsense.app.sonification.OperatingMode
import ai.secondsense.app.sonification.Spearcon
import ai.secondsense.app.sonification.TargetSelector
import ai.secondsense.app.sonification.TierClassifier
import ai.secondsense.app.sonification.TemporalSmoother
import ai.secondsense.app.sonification.Calibration
import ai.secondsense.app.inference.qnn.StubQnnBackend
import ai.secondsense.app.voice.GoalGrounding
import ai.secondsense.app.voice.OwlVitQnnGrounder
import ai.secondsense.app.voice.SpeechRecognizer
import ai.secondsense.app.voice.VectorToGoalController
import ai.secondsense.app.voice.VoiceCommandCapture
import ai.secondsense.app.voice.VoiceRecognizers
import ai.secondsense.app.voice.WhisperQnnRecognizer
import java.util.concurrent.Executors
import kotlin.concurrent.thread

/**
 * SecondSense — build-map ticket #6: on-device app skeleton + camera capture.
 *
 * Done-condition (from the build map):
 *   camera-in, audio-out, and haptic-out all work on-device with no laptop attached;
 *   a test tone + a test vibration both fire on a button tap.
 *
 * The inference engine is swappable ([InferenceEngine]); here we use the MOCK so the
 * whole app builds and runs before any model conversion (#8–#11) is done. When the
 * first QNN binary exists, replace `MockInferenceEngine()` with `QnnInferenceEngine(...)`
 * and nothing else in this file changes.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    // Engine is selected by EngineConfig.KIND (MOCK / TFLITE / QNN). Constructing it is
    // cheap; the heavy model load happens in engine.initialize() off the main thread below.
    private lateinit var engine: InferenceEngine
    private lateinit var audio: AudioOutput
    private lateinit var haptics: HapticOutput

    // Phase 3 — sonification spine (#18–#22)
    private lateinit var spearcon: Spearcon
    private lateinit var cueEngine: CueEngine
    private val targetSelector = TargetSelector()
    private val tierClassifier = TierClassifier()
    private val temporalSmoother = TemporalSmoother()   // #16
    private val calibration = Calibration()              // #7
    private val modeController = ModeController(OperatingMode.FLOW)
    @Volatile private var sonifying = false
    // latest raw center proximity, snapshotted by the #7 calibrate tap.
    @Volatile private var lastCenterProximity = 0.5f

    // Phase 4 — voice goal-seeking (#26–#28). QNN-backed: the recognizer/grounder go live
    // when the native bridge lands; the capture + noun-extraction + vector-to-goal logic is
    // wired now. Shared stub backend until then.
    private val voiceBackend = StubQnnBackend()
    // #26 ASR: sherpa-onnx offline keyword spotting when the optional module is compiled in
    // (-PenableSherpa), otherwise the QNN Whisper stub (honest "not loaded"). The real QNN
    // Whisper drops back in behind the same SpeechRecognizer interface later, no other change.
    private val speechRecognizer: SpeechRecognizer by lazy {
        VoiceRecognizers.create(this, WhisperQnnRecognizer(voiceBackend))
    }
    // #27 grounding: OWL-ViT is roadmap (QNN-only, blocked). The live path grounds the spoken
    // noun against the yolo26s COCO detections instead — see GoalGrounding + onFrameResult.
    private val grounder = OwlVitQnnGrounder(voiceBackend)
    private val vectorToGoal = VectorToGoalController()                   // #28
    private val voiceCapture by lazy { VoiceCommandCapture(speechRecognizer) }

    // Dedicated single thread for CameraX analysis callbacks.
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private lateinit var analyzer: FrameAnalyzer

    // simple running counters for the HUD
    @Volatile private var frameCount = 0L
    @Volatile private var lastInferenceMs = 0L
    // #17: edge-trigger the drop-off hazard cue so it fires once on entry, not every frame.
    // V3 drop-off plan — V2 (Sobel + ground-plane OR-logic) was REMOVED entirely after a real
    // on-device false positive (desk+keyboard scene fired BOTH V2 and V3 independently — see
    // HazardFusion.kt's revision notes). V3's hazardState is now the only drop-off signal.
    @Volatile private var lastHazardState: ai.secondsense.app.inference.decode.HazardState? = null

    // Accuracy: multi-frame detection-confidence stabilization (no model change).
    private val stabilizer = DetectionStabilizer()
    // Stop nagging about a static obstacle the user is standing near but not approaching.
    private val habituation = ObstacleHabituation()
    // Camera tamper / occlusion / knocked-off-mount warning state.
    @Volatile private var lastCamHealth = CameraHealth.OK
    private var lastCamNagMs = 0L
    private val CAM_NAG_MS = 7_000L
    // Overhead / head-height hazard channel (Bible §3) — edge-triggered.
    @Volatile private var lastOverhead = false
    // Double-tap "what's around me" scene description.
    @Volatile private var lastResult: FrameResult? = null
    private var tts: TextToSpeech? = null
    @Volatile private var ttsReady = false
    private var sceneGestures: GestureDetector? = null

    // Spoken-language preference (English / Hindi) + on-device Hindi<->English translation.
    private val langPrefs by lazy { LanguagePrefs(this) }
    private val translator by lazy { OcrTranslator() }

    // Episodic object memory ("where are my keys") — no SLAM. Pedometer + IMU heading feed a
    // drifting local-frame dead-reckoner; settled sightings from SceneAnalyzer land in
    // ObjectMemory; a voice "where is my X" reads back a coarse bearing and steers to it until
    // the object re-enters the camera (then the live goal-seek takes over).
    private val pedometer by lazy { PedometerTracker(this) }
    private val deadReckoner = DeadReckoner()
    private val objectMemory = ObjectMemory()
    @Volatile private var memoryNavActive = false

    // Problem Statement 6 — thermal throttling / deterministic latency in the closed harness.
    private val thermalGovernor by lazy { ThermalGovernor(walkingSupplier = { pedometer.isWalking }) }
    @Volatile private var perceptionEnabled = true
    @Volatile private var yamnetWanted = true
    @Volatile private var lowResActive = false

    private fun currentPose(): DeadReckoner.Pose =
        deadReckoner.pose(EngineConfig.imuTracker?.headingDeg ?: 0f)

    // ML Kit (offline): OCR sign-reading (Latin + Devanagari) + "person facing you".
    private val perception by lazy {
        MlKitPerception(
            onSign = { text, isDeva -> onSignRead(text, isDeva) },
            onFacingPerson = {
                val hi = langPrefs.speakHindi
                speakLocalized(
                    if (hi) "सामने एक व्यक्ति आपकी ओर देख रहा है" else "Person facing you",
                    hi,
                    TextToSpeech.QUEUE_ADD,
                    "aux",
                )
                haptics.testBuzz()
            },
        )
    }

    /** A sign was OCR'd — translate to the listener's language if needed, then speak it. */
    private fun onSignRead(text: String, isDevanagari: Boolean) {
        translator.localize(
            text = text,
            sourceIsDevanagari = isDevanagari,
            wantHindi = langPrefs.speakHindi,
            translateEnabled = langPrefs.translateSigns,
        ) { spoken, isHindi ->
            speakLocalized((if (isHindi) "साइन: " else "Sign: ") + spoken, isHindi, TextToSpeech.QUEUE_ADD, "aux")
        }
    }

    /**
     * Speak [text] in Hindi (hi-IN) or English, switching the TTS voice per call. Falls back to
     * the default voice if no Hindi data is installed (glyphs still spoken, just accented).
     */
    private fun speakLocalized(text: String, hindi: Boolean, queueMode: Int, utteranceId: String) {
        runOnUiThread {
            val t = tts
            val spoke = if (t != null && ttsReady) {
                val r = t.setLanguage(if (hindi) Locale("hi", "IN") else Locale.US)
                if (hindi && (r == TextToSpeech.LANG_MISSING_DATA || r == TextToSpeech.LANG_NOT_SUPPORTED)) {
                    t.setLanguage(Locale.US)
                }
                t.speak(text, queueMode, null, utteranceId)
            } else {
                TextToSpeech.ERROR
            }
            Toast.makeText(this, "🔊 $text", Toast.LENGTH_SHORT).show()
            if (spoke != TextToSpeech.SUCCESS) {
                Toast.makeText(this, "(no TTS voice — install one in Settings › Text-to-speech)", Toast.LENGTH_SHORT).show()
            }
        }
    }
    @Volatile private var lastPossibleDropAtMs = 0L
    private val POSSIBLE_DROP_COOLDOWN_MS = 1500L
    private val OVERHEAD_PROXIMITY = 0.45f

    // #30 laptop dashboard — a spectator view for judges/demo-partners, not the user (who
    // never sees the screen). Null if the port failed to bind; everything else degrades
    // gracefully around that (publish() calls just become no-ops via ?.).
    private var dashboardServer: DashboardServer? = null

    // Research-candidate #7 (secondsense_research_candidates_v1.md) — independent barometer
    // cross-check for a drop-off, targeting the CONFIRMED bug where the depth model reads
    // the wrong sign on a real descending staircase. A barometer can't be fooled by texture
    // or lighting the way monocular depth can, so it's a genuinely independent second opinion.
    private val barometer by lazy { BarometerMonitor(this) }

    // Ticket #33 — continuous hazard-sound detection (car horns, sirens, alarms), a sensing
    // modality entirely independent of the camera: catches hazards outside the vision
    // pipeline's field of view (e.g. a car approaching from behind).
    private val hazardDetector by lazy { HazardSoundDetector(this) }
    @Volatile private var lastHazardLabel: String? = null
    @Volatile private var lastHazardScore: Float = 0f
    @Volatile private var lastHazardSampleLabel: String? = null
    @Volatile private var lastHazardSampleScore: Float = 0f
    // Ticket #34 — voice auto-ducking state, for HUD visibility.
    @Volatile private var isDucked: Boolean = false

    private val requestCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
        }

    private val requestMic =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                // startVoiceCapture() pauses hazard listening for its capture window and
                // restarts it in its callback, so DON'T also start it here — two AudioRecords
                // on the MIC at once is exactly the contention we're avoiding.
                startVoiceCapture()
            } else Toast.makeText(this, "Mic permission is required for voice search", Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        // Hands-free, always-on aid: never let the screen sleep while the app is foreground
        // (sleep pauses the Activity, which stops the camera + inference mid-use).
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        audio = AudioOutput().also { it.initialize() }
        haptics = HapticOutput(this)
        spearcon = Spearcon(this).also { it.initialize() }
        cueEngine = CueEngine(audio, haptics, spearcon)

        // Pick the engine (MOCK / TFLITE / QNN) — one switch in EngineConfig.
        engine = EngineConfig.create(this)
        // V3 drop-off plan §1 — the IMU (gyro+accel) fusion feeding the traversable corridor.
        // EngineConfig.create() always populates imuTracker (even for MOCK). It's started in
        // onResume and stopped in onPause so it doesn't keep listening while backgrounded.

        // Engine init off the main thread (real model load is heavier than the mock). Caught
        // broadly: a real crash here (native JNI in particular) would otherwise kill the whole
        // app before it ever shows a frame — degrade to empty/mock-like frames instead, same
        // failure mode as a normal engine.initialize() logging-and-returning failure.
        thread(name = "engine-init") {
            try {
                engine.initialize()
            } catch (t: Throwable) {
                android.util.Log.e("SecondSense/engine", "engine.initialize() crashed; app continues with an uninitialized engine", t)
            }
        }

        // #26 — load the offline ASR model (sherpa KWS) off the main thread; harmless no-op
        // for the QNN stub. Touching `speechRecognizer` here also triggers its lazy build.
        thread(name = "asr-init") {
            try {
                speechRecognizer.initialize()
            } catch (t: Throwable) {
                android.util.Log.w("SecondSense/voice", "speechRecognizer.initialize() failed", t)
            }
        }

        analyzer = FrameAnalyzer(
            engine,
            onResult = { result -> onFrameResult(result) },
            frameSink = { bmp -> if (perceptionEnabled) perception.offer(bmp) },
        )

        // --- #6 done-condition: test tone + test buzz on tap ---
        binding.btnTestTone.setOnClickListener { audio.testTone() }
        binding.btnTestBuzz.setOnClickListener { haptics.testBuzz() }

        // Mode (#25): FLOW drives center-crop ON (sparse); SCAN_SEEK widens the field.
        // The controller is the single source of truth; it drives the analyzer.
        modeController.addListener { mode ->
            analyzer.centerCrop = (mode == OperatingMode.FLOW)
            habituation.reset()   // a mode switch is a fresh intent — re-alert obstacles
            runOnUiThread { binding.switchCenterCrop.isChecked = analyzer.centerCrop }
        }
        binding.switchMode.setOnCheckedChangeListener { _, checked ->
            // checked = SCAN_SEEK (stopped, exploring); unchecked = FLOW (walking).
            modeController.set(if (checked) OperatingMode.SCAN_SEEK else OperatingMode.FLOW)
        }

        // flow-mode center-crop toggle (#14): a low-level dev override under the mode.
        binding.switchCenterCrop.isChecked = analyzer.centerCrop
        binding.switchCenterCrop.setOnCheckedChangeListener { _, checked ->
            analyzer.centerCrop = checked
        }

        // #7 one-tap calibration: snapshot the current forward clearance as the baseline;
        // tap again to clear. Off by default (identity passthrough) until first tap.
        binding.btnCalibrate.setOnClickListener {
            if (calibration.isCalibrated) {
                calibration.clear()
                Toast.makeText(this, "Calibration cleared", Toast.LENGTH_SHORT).show()
            } else {
                calibration.capture(lastCenterProximity)
                // Also define "level" for the mount: capture the CURRENT pose as vertical so
                // the camera-health monitor judges tilt relative to how the wearer set it up,
                // not an auto-guess. Hold the phone perfectly vertical when tapping this.
                EngineConfig.imuTracker?.calibrateMountingOffset()
                habituation.reset()
                speakLocalized(
                    if (langPrefs.speakHindi) "कैमरा कोण सीधा सेट किया गया" else "Camera angle set to vertical",
                    langPrefs.speakHindi, TextToSpeech.QUEUE_FLUSH, "camcal",
                )
                Toast.makeText(
                    this,
                    "Calibrated: forward @ %.2f + mount angle = vertical".format(lastCenterProximity),
                    Toast.LENGTH_SHORT,
                ).show()
            }
        }

        // Sonification on/off — start/stop the continuous cue loop (#22).
        binding.switchSonify.setOnCheckedChangeListener { _, checked ->
            sonifying = checked
            if (checked) cueEngine.start() else { cueEngine.stop(); cueEngine.update(null) }
        }
        // Default ON: the cue loop is what actually renders every audio/haptic cue (obstacle
        // AND voice steering). Setting it here fires the listener above -> sonifying=true +
        // cueEngine.start(). Toggle off in-app if you need silence during setup.
        binding.switchSonify.isChecked = true

        // Spoken-language preference — English (default) or Hindi. Controls the TTS voice for
        // every cue the app speaks (sign read-outs, scene description, "person facing you").
        binding.switchHindi.isChecked = langPrefs.speakHindi
        binding.switchHindi.setOnCheckedChangeListener { _, checked ->
            langPrefs.speakHindi = checked
            speakLocalized(if (checked) "अब हिंदी में" else "English now", checked, TextToSpeech.QUEUE_FLUSH, "lang")
        }
        // When a sign's script differs from that preference, translate it before speaking
        // (Hindi<->English, on-device). Off = always read the sign in its printed script.
        binding.switchTranslate.isChecked = langPrefs.translateSigns
        binding.switchTranslate.setOnCheckedChangeListener { _, checked ->
            langPrefs.translateSigns = checked
        }

        // Double-tap the camera preview -> speak "what's around me" (offline, no LLM).
        tts = TextToSpeech(this) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) {
                tts?.setLanguage(if (langPrefs.speakHindi) Locale("hi", "IN") else Locale.US)
            }
        }
        // One-time Hindi<->English translation model fetch (Wi-Fi only); no-op once cached.
        translator.prewarm()

        // Object-memory dead-reckoning: advance the local-frame position one stride per step,
        // along the heading at that instant.
        pedometer.onStep = {
            deadReckoner.onStep(EngineConfig.imuTracker?.headingDeg ?: 0f, pedometer.strideMeters)
        }

        // Thermal governor — turns cadences/resolution/aux load down as the harness heats up.
        thermalGovernor.onPolicy = { p -> runOnUiThread { applyPerfPolicy(p) } }
        thermalGovernor.onNotice = { msg ->
            speakLocalized(msg, langPrefs.speakHindi, TextToSpeech.QUEUE_ADD, "thermal")
        }
        sceneGestures = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent): Boolean = true
            override fun onDoubleTap(e: MotionEvent): Boolean { narrateScene(); return true }
        })
        // dispatchTouchEvent (below) feeds it every touch on the window, before any view can
        // consume it — the PreviewView surface + child buttons otherwise eat the events.

        // Phase 4 voice search (#26–#28): gated to SCAN_SEEK (#25 — you stop, then ask).
        binding.btnFind.setOnClickListener {
            if (!modeController.acceptsVoiceCommands) {
                Toast.makeText(this, "Switch to Scan/Seek mode for voice search (#25)", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (hasMicPermission()) startVoiceCapture()
            else requestMic.launch(Manifest.permission.RECORD_AUDIO)
        }

        binding.btnDebugPanel.setOnClickListener {
            startActivity(android.content.Intent(this, DebugActivity::class.java))
        }

        // Path B — AR room scan (experimental). Isolated activity; owns the camera via ARCore.
        binding.btnRoomScan.setOnClickListener {
            when (ai.secondsense.app.ar.ArSupport.availability(this)) {
                ai.secondsense.app.ar.ArSupport.State.UNSUPPORTED ->
                    Toast.makeText(this, "This device isn't AR-capable — room scan unavailable", Toast.LENGTH_LONG).show()
                else ->
                    startActivity(android.content.Intent(this, ai.secondsense.app.ar.RoomScanActivity::class.java))
            }
        }

        startDashboardServer()
        // barometer + hazard-sound listening are started in onResume (and stopped in onPause)
        // so they don't keep the mic/sensor open while the app is backgrounded.

        if (hasCameraPermission()) startCamera()
        else requestCamera.launch(Manifest.permission.CAMERA)
    }

    /**
     * #30 — start the embedded HTTP server and, if a local IP is available, show a QR code
     * for a laptop on the same Wi-Fi/hotspot to scan. Never crashes the app if the port is
     * busy or no network interface is up (e.g. true airplane mode with Wi-Fi also off) —
     * the assistive pipeline doesn't depend on this succeeding.
     */
    private fun startDashboardServer() {
        try {
            val server = DashboardServer().also { it.start() }
            dashboardServer = server
            val ip = DashboardServer.localIpAddress(this)
            if (ip != null) {
                val url = "http://$ip:8085/"
                binding.dashboardQr.setImageBitmap(QrCodeGenerator.generate(url))
                binding.dashboardUrl.text = url
                binding.dashboardQr.visibility = android.view.View.VISIBLE
                binding.dashboardUrl.visibility = android.view.View.VISIBLE
            }
        } catch (t: Throwable) {
            // Port busy, no network, etc. — dashboard is a spectator aid, never block the app.
            android.util.Log.w("SecondSense/dashboard", "server start failed: ${t.message}")
        }
    }

    /** #30 — one JSON snapshot per frame for the dashboard; mirrors the HUD text above. */
    private fun publishDashboardState(
        result: FrameResult,
        target: ai.secondsense.app.sonification.CueTarget?,
    ) {
        val server = dashboardServer ?: return
        try {
            val json = JSONObject().apply {
                put("engine", engine.name)
                put("mode", modeController.mode.toString())
                put("inferMs", lastInferenceMs)
                put("frames", frameCount)
                put("thermalTier", thermalGovernor.tier.name)
                put("perfPolicy", thermalGovernor.policy.label)
                put("p90InferMs", thermalGovernor.p90Ms)
                put("battTempC", thermalGovernor.batteryTempC.let { if (it.isNaN()) null else it })
                put("thermalHeadroom", thermalGovernor.headroom.let { if (it.isNaN()) null else it })
                put("tier", target?.tier?.toString())
                put("cueLabel", target?.label)
                put("cueDir", target?.let { if (it.azimuth < 0.4f) "L" else if (it.azimuth > 0.6f) "R" else "C" })
                put("cueProx", target?.proximity)
                // V3 is the sole drop-off source now (V2 removed). Key names kept as-is so
                // the existing #30 laptop dashboard page doesn't need a matching edit.
                put("dropOff", result.hazardState == ai.secondsense.app.inference.decode.HazardState.DROP_CONFIRMED)
                put("dropOffPct", result.hazardFirstEdgeY?.let { (it * 100).toInt() })
                put("dropOffBaroConfirmed",
                    result.hazardState == ai.secondsense.app.inference.decode.HazardState.DROP_CONFIRMED &&
                        barometer.descendingConfirmed())
                put("hazardState", result.hazardState?.name)
                put("hazardConfidence", result.hazardConfidence)
                put("hazardUrgency", result.hazardUrgency)
                put("hazardListening", lastHazardSampleLabel)
                put("hazardLast", lastHazardLabel)
                put("ducked", isDucked)
                put("overhead", lastOverhead)
                put("goal", if (vectorToGoal.isActive) vectorToGoal.activeGoal else null)
                put("detections", JSONArray().apply {
                    result.detections.take(6).forEach { d ->
                        put(JSONObject().apply {
                            put("label", d.label)
                            put("score", "%.2f".format(d.score))
                            put("prox", "%.2f".format(d.proximity))
                        })
                    }
                })
            }
            server.publish(json.toString())
        } catch (t: Throwable) {
            android.util.Log.w("SecondSense/dashboard", "publish failed: ${t.message}")
        }
    }

    /**
     * #33 — starts continuous audio hazard detection (car horns, sirens, alarms). Safe to
     * call more than once (HazardSoundDetector.start() no-ops if already running). Never
     * blocks or crashes the app if the model fails to load — a different sensing modality
     * being unavailable shouldn't take down the vision pipeline.
     */
    private fun startHazardDetection() {
        thread(name = "hazard-init") {
            if (!hazardDetector.isReady) hazardDetector.initialize()
            hazardDetector.start(
                onHazard = { hazard ->
                    lastHazardLabel = hazard.label
                    lastHazardScore = hazard.score
                    haptics.hazardSound()
                    runOnUiThread {
                        Toast.makeText(this, "⚠ Hazard sound: ${hazard.label}", Toast.LENGTH_SHORT).show()
                    }
                },
                onSample = { label, score ->
                    lastHazardSampleLabel = label
                    lastHazardSampleScore = score
                },
                onSpeechChanged = { speaking ->
                    isDucked = speaking
                    cueEngine.setDucked(speaking)
                },
            )
        }
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun hasMicPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    /**
     * #26 — capture a short spoken command, extract the target noun, and set it as the
     * vector-to-goal target (#28). Transcription is the QNN Whisper model: until the native
     * bridge lands the recognizer isn't ready, and we say so honestly instead of faking a goal.
     */
    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        sceneGestures?.onTouchEvent(ev)   // observe every touch; don't consume
        return super.dispatchTouchEvent(ev)
    }

    /**
     * Tell the wearer (they can't see the screen) when the camera is covered or knocked off
     * its mount, and again every [CAM_NAG_MS] while it stays bad. Announce recovery once.
     */
    private fun handleCameraHealth(h: CameraHealth) {
        val bad = h == CameraHealth.BLOCKED || h == CameraHealth.MISALIGNED
        val wasBad = lastCamHealth == CameraHealth.BLOCKED || lastCamHealth == CameraHealth.MISALIGNED
        val now = System.currentTimeMillis()
        val hi = langPrefs.speakHindi
        if (bad && (!wasBad || h != lastCamHealth || now - lastCamNagMs > CAM_NAG_MS)) {
            lastCamNagMs = now
            val msg = when (h) {
                CameraHealth.BLOCKED ->
                    if (hi) "कैमरा ढका हुआ है, कृपया इसे साफ़ करें" else "Camera is blocked. Please clear it."
                CameraHealth.MISALIGNED ->
                    if (hi) "कैमरा हिल गया है, कृपया इसे सीधा करें" else "Camera has moved. Please straighten it."
                else -> ""
            }
            speakLocalized(msg, hi, TextToSpeech.QUEUE_FLUSH, "camhealth")
        } else if (!bad && wasBad) {
            speakLocalized(if (hi) "कैमरा ठीक है" else "Camera is okay now", hi, TextToSpeech.QUEUE_ADD, "camhealth")
            haptics.testBuzz()
        }
        lastCamHealth = h
    }

    /** Apply a [PerfPolicy] from the thermal governor — cadences, aux load, resolution. */
    private fun applyPerfPolicy(p: PerfPolicy) {
        analyzer.processEveryN = p.frameEveryN
        engine.setDepthEveryN(p.depthEveryN)
        engine.setHazardEveryN(p.hazardEveryN)
        perceptionEnabled = p.auxEnabled

        if (p.yamnetEnabled != yamnetWanted) {
            yamnetWanted = p.yamnetEnabled
            if (yamnetWanted) { if (hasMicPermission()) startHazardDetection() } else hazardDetector.stop()
        }
        if (p.lowRes != lowResActive) {
            lowResActive = p.lowRes
            if (hasCameraPermission()) startCamera()   // rebinds ImageAnalysis at the new size
        }
        android.util.Log.i(
            "SecondSense/thermal",
            "applied ${p.label}: frame/${p.frameEveryN} depth/${p.depthEveryN} hazard/${p.hazardEveryN} " +
                "aux=${p.auxEnabled} yamnet=${p.yamnetEnabled} lowRes=${p.lowRes}",
        )
    }

    /** Double-tap handler: speak a one-sentence description of the current frame (offline). */
    private fun narrateScene() {
        val text = SceneNarrator.describe(lastResult)
        android.util.Log.i("SecondSense/scene", "narrateScene: \"$text\" (tts=${tts != null})")
        if (langPrefs.speakHindi && langPrefs.translateSigns) {
            // SceneNarrator emits English — route it through en->hi so it's spoken in Hindi
            // (falls back to the English sentence if the model pair isn't downloaded).
            translator.localize(text, sourceIsDevanagari = false, wantHindi = true, translateEnabled = true) { spoken, isHindi ->
                speakLocalized(spoken, isHindi, TextToSpeech.QUEUE_FLUSH, "scene")
            }
        } else {
            speakLocalized(text, langPrefs.speakHindi, TextToSpeech.QUEUE_FLUSH, "scene")
        }
    }

    private fun startVoiceCapture() {
        Toast.makeText(this, "Listening… say \"find the …\"", Toast.LENGTH_SHORT).show()
        // The hazard detector holds a continuous AudioRecord on the MIC; a second AudioRecord
        // on the same source (this capture) fails on many devices. Pause hazard listening for
        // the short capture window, then restart it.
        hazardDetector.stop()
        voiceCapture.capture { noun, transcript, recognizerReady ->
            if (hasMicPermission()) startHazardDetection()
            // "where is my X" / "where's my X" -> recall from memory; anything else -> live seek.
            val isRecall = transcript?.lowercase()?.let {
                it.contains("where") || it.contains("last seen") || it.contains("did i leave")
            } == true
            runOnUiThread {
                when {
                    !recognizerReady -> Toast.makeText(
                        this,
                        "Voice model not loaded — build with -PenableSherpa + add the KWS model (see android/app/src/sherpa/README.md)",
                        Toast.LENGTH_LONG,
                    ).show()
                    noun == null -> Toast.makeText(this, "Didn't catch a target", Toast.LENGTH_SHORT).show()
                    isRecall -> recallObject(noun)
                    else -> {
                        vectorToGoal.setGoal(noun)
                        memoryNavActive = false
                        binding.switchSonify.isChecked = true   // ensure the cue loop is running to steer
                        Toast.makeText(this, "Goal set: $noun", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    /** Answer "where is my <noun>" from short-horizon memory, then steer toward the bearing. */
    private fun recallObject(noun: String) {
        val hit = objectMemory.recall(noun, currentPose(), System.currentTimeMillis())
        if (hit == null) {
            speakMemory("I don't have a memory of your $noun yet. I only remember things I've seen you set down.")
            return
        }
        speakMemory(MemoryPhrase.build(noun, hit))
        vectorToGoal.setGoal(noun)
        memoryNavActive = true
        binding.switchSonify.isChecked = true
        Toast.makeText(this, "🧠 ${MemoryPhrase.build(noun, hit)}", Toast.LENGTH_LONG).show()
    }

    /** Speak a memory sentence, translating to Hindi first when that's the preference. */
    private fun speakMemory(english: String) {
        if (langPrefs.speakHindi && langPrefs.translateSigns) {
            translator.localize(english, sourceIsDevanagari = false, wantHindi = true, translateEnabled = true) { spoken, isHi ->
                speakLocalized(spoken, isHi, TextToSpeech.QUEUE_FLUSH, "mem")
            }
        } else {
            speakLocalized(english, langPrefs.speakHindi, TextToSpeech.QUEUE_FLUSH, "mem")
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.preview.surfaceProvider)
            }

            // Request a modest analysis resolution — the NPU models want small
            // inputs anyway, and this keeps the loop fast. RGBA_8888 so the
            // analyzer's toBitmap() path is direct.
            // Thermal-aware: drop to 320x240 when the governor says the harness is HOT — halves
            // the ISP / YUV-convert / optical-flow load. Rebind (via startCamera()) applies it.
            val analysisSize = if (lowResActive) Size(320, 240) else Size(640, 480)
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        analysisSize,
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(analysisExecutor, analyzer) }

            // Rear camera — chest-mounted, forward-facing (Bible §17).
            val selector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview, analysis)
            } catch (e: Exception) {
                Toast.makeText(this, "Camera bind failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFrameResult(raw: FrameResult) {
        frameCount++
        lastInferenceMs = raw.inferenceMillis
        thermalGovernor.onInferenceMs(raw.inferenceMillis)

        // ACCURACY: multi-frame confidence stabilization — boost consistently re-seen
        // detections, hold down one-frame flickers. Engine-agnostic post-processing.
        val result = raw.copy(detections = stabilizer.update(raw.detections))
        lastResult = result

        handleCameraHealth(result.cameraHealth)

        // OBJECT MEMORY: a named object just came to rest in view -> log where it is, in the
        // dead-reckoned local frame. Cheap; only consulted by a later "where is my X" query.
        result.settledObject?.let { s ->
            objectMemory.remember(s, currentPose(), System.currentTimeMillis())
            android.util.Log.i(
                "SecondSense/memory",
                "logged %s ~%.1fm @%.0f° (mem=%d)".format(s.label, s.distanceM, s.bearingDeg, objectMemory.size),
            )
        }

        // OVERHEAD / head-height hazard (Bible §3, the #1 differentiator): a close object whose
        // WHOLE box sits high in the frame is exactly the cane's blind spot. Distinct cue, on
        // the rising edge only, independent of the sonification toggle.
        // Box CENTER in the upper 40% of the frame (a real head-height hazard spans the
        // upper-middle; requiring the whole box up top was too strict to ever fire).
        val overhead = result.detections.any {
            it.box.centerY < 0.40f && it.proximity >= OVERHEAD_PROXIMITY
        }
        if (overhead && !lastOverhead) haptics.overhead()
        lastOverhead = overhead

        // #17 DROP-OFF: a downward negative obstacle is life-safety-critical, so it fires its
        // own distinct haptic REGARDLESS of the sonification toggle — but only on the rising
        // edge (entry), so a taped curb line doesn't machine-gun the motor every frame.
        //
        // V3 is now the SOLE drop-off detector — the old V2 (Sobel + ground-plane OR-logic)
        // path was removed entirely per explicit user request after it independently
        // false-positived on a desk/keyboard scene alongside V3.
        //
        // Research-candidate #7: an independent barometer cross-check. Doesn't gate whether
        // the haptic fires (vision stays the safety-critical trigger — a barometer alone is
        // too slow/noisy to be a sole source), but a confirmed real descent bumps urgency to
        // max regardless of how well-localized the vision edge was.
        //
        // DROP_CONFIRMED gets the escalating pattern, on the rising edge only; POSSIBLE_DROP
        // gets a single subdued pulse, cooldown-gated so it can't machine-gun on flicker
        // between POSSIBLE_DROP and SAFE frame to frame; SCENE_NOT_TRAVERSABLE and
        // SENSOR_BLOCKED get no drop cue at all (the plan asks for a distinct "path blocked"
        // cue — deferred to keep scope tight; silence is at least not a FALSE drop cue).
        val hazardState = result.hazardState
        val baroConfirmed = hazardState == ai.secondsense.app.inference.decode.HazardState.DROP_CONFIRMED &&
            barometer.descendingConfirmed()
        if (hazardState == ai.secondsense.app.inference.decode.HazardState.DROP_CONFIRMED &&
            lastHazardState != ai.secondsense.app.inference.decode.HazardState.DROP_CONFIRMED
        ) {
            val visionUrgency = result.hazardFirstEdgeY?.let { ((it - 0.5f) / 0.5f).coerceIn(0f, 1f) } ?: 1f
            haptics.dropOff(if (baroConfirmed) 1f else visionUrgency)
        } else if (hazardState == ai.secondsense.app.inference.decode.HazardState.POSSIBLE_DROP) {
            val now = System.currentTimeMillis()
            if (now - lastPossibleDropAtMs > POSSIBLE_DROP_COOLDOWN_MS) {
                haptics.possibleDrop()
                lastPossibleDropAtMs = now
            }
        } else if (
            (hazardState == ai.secondsense.app.inference.decode.HazardState.SCENE_NOT_TRAVERSABLE ||
                hazardState == ai.secondsense.app.inference.decode.HazardState.SENSOR_BLOCKED) &&
            lastHazardState != hazardState
        ) {
            // Gap fix: "path blocked / can't tell" — was silent before. Rising edge only.
            haptics.pathBlocked()
        }
        lastHazardState = hazardState

        // PHASE 4 (#27/#28) — voice goal-seeking, closed-vocab TFLite path. When a spoken goal
        // is active AND we're stopped (SCAN_SEEK, per #25), steer toward the matching COCO
        // detection instead of cueing the nearest obstacle. On arrival: a distinct haptic,
        // clear the goal, fall back to the obstacle spine.
        val goalMatch = if (modeController.acceptsVoiceCommands)
            GoalGrounding.match(result.detections, vectorToGoal.activeGoal) else null
        val goalCue = goalMatch?.let { m ->
            val prox = calibration.apply(m.proximity)
            if (vectorToGoal.hasArrived(m.box, prox)) {
                haptics.arrived()
                val reached = vectorToGoal.activeGoal
                vectorToGoal.setGoal(null)
                runOnUiThread { Toast.makeText(this, "Arrived: $reached", Toast.LENGTH_SHORT).show() }
                null
            } else {
                vectorToGoal.cueFor(m.box, prox)
            }
        }

        // MEMORY NAV: when steering to a REMEMBERED object, cue its dead-reckoned bearing
        // until it re-enters the camera — at which point goalMatch above fires and the live
        // visual cue takes over (the "visual handoff"). Lower priority than a live match.
        if (goalMatch != null && memoryNavActive) memoryNavActive = false
        val memoryCue: CueTarget? = if (memoryNavActive && goalCue == null) {
            val hit = objectMemory.recall(vectorToGoal.activeGoal, currentPose(), System.currentTimeMillis())
            when {
                hit == null -> { memoryNavActive = false; null }
                hit.distanceM < 0.8f -> {
                    memoryNavActive = false
                    val g = vectorToGoal.activeGoal ?: "it"
                    vectorToGoal.setGoal(null)
                    speakMemory("You should be right next to your $g. I can't see it yet.")
                    null
                }
                else -> CueTarget(
                    azimuth = (hit.bearingDeg.coerceIn(-90f, 90f) / 90f) * 0.5f + 0.5f,
                    proximity = (1f - hit.distanceM / 6f).coerceIn(0.12f, 0.60f),
                    label = hit.label,
                    tier = ai.secondsense.app.inference.ConfidenceTier.BLUE,
                )
            }
        } else null

        // TARGETING (#14/#15) + TIER DERIVATION (#23): resolve the frame to the single
        // thing to cue, and stamp a smoothed, signal-derived confidence tier onto it.
        val rawTarget = targetSelector.selectWithTier(result, tierClassifier)
        // #7: remember the raw center proximity so a calibrate tap can snapshot it.
        rawTarget?.let { lastCenterProximity = it.proximity }
        // #7: re-reference proximity against the one-tap baseline (identity passthrough if
        // uncalibrated). #16: gate on ~3-frame persistence so flicker never fires a cue.
        val calibrated = rawTarget?.copy(proximity = calibration.apply(rawTarget.proximity))
        val obstacleTarget = temporalSmoother.update(calibrated)
        // Habituation: once a static obstacle has been announced and the user isn't closing on
        // it, stop cueing it until the situation changes. Safety hazards (drop-off / overhead)
        // are edge-triggered haptics and bypass this entirely.
        val gatedObstacle = habituation.filter(obstacleTarget, pedometer.isWalking, System.currentTimeMillis())
        // A visible, not-yet-reached voice goal wins; then a remembered object's bearing;
        // otherwise the (habituation-gated) obstacle spine (#22). When the camera is blocked
        // or off its mount, EVERY vision-derived cue is suspect — go silent on the spine and
        // let handleCameraHealth() do the talking (the SENSOR_BLOCKED hazard path still fires
        // its own "path blocked" haptic).
        val camUsable = result.cameraHealth == CameraHealth.OK || result.cameraHealth == CameraHealth.DIM
        val target = if (!camUsable) null else goalCue ?: memoryCue ?: gatedObstacle
        if (sonifying) cueEngine.update(target)
        publishDashboardState(result, target)

        runOnUiThread {
            // #23: tier is visible per-cue — colored badge + text. (Full dashboard is #30.)
            val tierColor = when (target?.tier) {
                ai.secondsense.app.inference.ConfidenceTier.WHITE -> 0xFFEDEDED.toInt()
                ai.secondsense.app.inference.ConfidenceTier.BLUE -> 0xFF3B82F6.toInt()
                ai.secondsense.app.inference.ConfidenceTier.RED -> 0xFFEF4444.toInt()
                null -> 0x33FFFFFF
            }
            binding.tierBadge.setBackgroundColor(tierColor)

            binding.hud.text = buildString {
                append("engine: ${engine.name}   mode: ${modeController.mode}\n")
                append("frames: $frameCount   infer: ${lastInferenceMs}ms  p90: ${thermalGovernor.p90Ms}ms\n")
                append("thermal: ${thermalGovernor.tier} [${thermalGovernor.policy.label}]  batt %.1f°C  headroom %.2f\n"
                    .format(thermalGovernor.batteryTempC, thermalGovernor.headroom))
                append("dets: ${result.detections.size}   crop: ${analyzer.centerCrop}   son: $sonifying\n")
                append("cal: ${if (calibration.isCalibrated) "on@%.2f".format(calibration.baselineValue ?: 0f) else "off"}\n")
                run {
                    val p = currentPose()
                    append("mem: ${objectMemory.size} obj   steps: ${pedometer.stepCount}   dr: x=%.1f z=%.1f hdg=%.0f°".format(p.x, p.z, p.headingDeg))
                    append(if (memoryNavActive) "   → ${vectorToGoal.activeGoal}\n" else "\n")
                }
                if (overhead) append("⚠ OVERHEAD / HEAD-HEIGHT HAZARD\n")
                if (result.cameraHealth != CameraHealth.OK) append("⚠ CAMERA: ${result.cameraHealth}\n")
                if (EngineConfig.imuTracker?.hasMountCalibration != true)
                    append("cam angle: not calibrated (hold vertical, tap Calibrate)\n")
                if (habituation.muted) append("obstacle cue: muted (static, not approaching)\n")
                append("baro: ${if (!barometer.isAvailable) "no sensor on this device" else "trend=%.3f hPa".format(barometer.pressureTrendHpa() ?: 0f)}\n")
                if (hazardDetector.isReady) {
                    append("hazard listen: ${lastHazardSampleLabel ?: "—"} (%.2f)\n".format(lastHazardSampleScore))
                    if (lastHazardLabel != null) append("⚠ LAST HAZARD: $lastHazardLabel (%.2f)\n".format(lastHazardScore))
                    append("ducking: ${if (isDucked) "ON — speech detected, cues quiet" else "off"}\n")
                }
                // DETECTIONS: all objects this frame (before temporal gate) — label, score, prox.
                // Lets you verify if YOLO labels match what's actually in front of the camera.
                val topDets = result.detections.take(3)
                if (topDets.isEmpty()) {
                    append("detects: (none)\n")
                } else {
                    topDets.forEachIndexed { i, d ->
                        append("det${i+1}: ${d.label ?: "(unknown)"} s=${"%.2f".format(d.score)} prox=${"%.2f".format(d.proximity)} cy=${"%.2f".format(d.box.centerY)}\n")
                    }
                }
                val rawSide = rawTarget?.let {
                    when { it.azimuth < 0.4f -> "L"; it.azimuth > 0.6f -> "R"; else -> "C" }
                }
                append("raw: ${rawTarget?.label ?: "—"}  ${rawSide ?: ""}  prox=${"%.2f".format(rawTarget?.proximity ?: 0f)}\n")
                if (vectorToGoal.isActive) {
                    append("GOAL: ${vectorToGoal.activeGoal}  ${if (goalMatch != null) "→ steering" else "(not in view / not a COCO class)"}\n")
                }
                // The ONLY drop-off signal now (V2's Sobel/ground-plane OR-logic was removed
                // entirely) — IMU-stabilized corridor + RGB edge lattice + depth-as-evidence,
                // fused through the temporal hazard state machine.
                if (hazardState != null) {
                    val edgeStr = result.hazardFirstEdgeY?.let { " edgeY=%.2f".format(it) } ?: ""
                    val baro = if (baroConfirmed) " [baro CONFIRMED]" else ""
                    if (hazardState == ai.secondsense.app.inference.decode.HazardState.DROP_CONFIRMED) {
                        append("⚠ DROP-OFF AHEAD$edgeStr$baro (urgency %.2f)\n".format(result.hazardUrgency ?: 0f))
                    } else {
                        append("hazard: $hazardState conf=%.2f urgency=%.2f$edgeStr\n"
                            .format(result.hazardConfidence ?: 0f, result.hazardUrgency ?: 0f))
                    }
                }
                if (target != null) {
                    val id = target.label ?: "(unknown)"
                    val side = when {
                        target.azimuth < 0.4f -> "L"
                        target.azimuth > 0.6f -> "R"
                        else -> "C"
                    }
                    append(
                        "CUE: %s  dir=%s(%.2f)  prox=%.2f  tier=%s"
                            .format(id, side, target.azimuth, target.proximity, target.tier)
                    )
                } else {
                    append("CUE: — (smoother waiting for stable target)")
                }
            }
        }
    }

    override fun onPause() {
        super.onPause()
        // Nothing that makes sound or holds a mic/sensor should keep running while the app is
        // backgrounded. CameraX is lifecycle-bound and stops itself; these are not:
        //  - the cue loop would otherwise keep firing the LAST posted target forever, because
        //    no new frames arrive to update or clear it;
        //  - the IMU + barometer would keep listening;
        //  - the hazard detector would keep the mic recording.
        cueEngine.stop()
        cueEngine.update(null)
        EngineConfig.imuTracker?.stop()
        pedometer.stop()
        barometer.stop()
        thermalGovernor.stop()
        hazardDetector.stop()
        stabilizer.reset()   // camera pauses -> tracks would go stale
        habituation.reset()
    }

    override fun onResume() {
        super.onResume()
        EngineConfig.imuTracker?.start()
        pedometer.start()
        barometer.start()
        thermalGovernor.start(this)
        if (sonifying) cueEngine.start()
        if (yamnetWanted && hasMicPermission()) startHazardDetection()
    }

    override fun onDestroy() {
        super.onDestroy()
        EngineConfig.imuTracker?.stop()
        cueEngine.stop()
        // Drain any in-flight analyze() BEFORE closing the engine, or engine.close() nulls the
        // interpreters under a frame still being processed on the analysis thread (NPE on
        // yolo!!/depth!!).
        analysisExecutor.shutdown()
        try {
            analysisExecutor.awaitTermination(500, java.util.concurrent.TimeUnit.MILLISECONDS)
        } catch (_: InterruptedException) {
        }
        spearcon.release()
        audio.release()
        haptics.release()
        voiceCapture.cancel()
        runCatching { speechRecognizer.release() }
        voiceBackend.close()
        engine.close()
        dashboardServer?.stop()
        barometer.stop()
        hazardDetector.close()
        tts?.run { stop(); shutdown() }
        runCatching { perception.close() }
        runCatching { translator.close() }
    }
}
