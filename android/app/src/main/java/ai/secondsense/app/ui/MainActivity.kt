package ai.secondsense.app.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Size
import android.view.GestureDetector
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.view.accessibility.AccessibilityManager
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
import ai.secondsense.app.context.AppContext
import ai.secondsense.app.context.ContextAutoDetector
import ai.secondsense.app.context.ContextManager
import ai.secondsense.app.sonification.CueTarget
import ai.secondsense.app.sonification.ObstacleHabituation
import java.util.Locale
import ai.secondsense.app.voice.SceneNarrator
import ai.secondsense.app.dashboard.DashboardServer
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
import ai.secondsense.app.voice.IntentInterpreter
import ai.secondsense.app.voice.LlmAssistant
import ai.secondsense.app.voice.LlmAssistants
import ai.secondsense.app.voice.LlmResolution
import ai.secondsense.app.voice.PhoneActions
import ai.secondsense.app.voice.SafetyGate
import ai.secondsense.app.voice.SafetyVectorGate
import ai.secondsense.app.voice.SafetyVectorGates
import ai.secondsense.app.voice.SceneBrief
import ai.secondsense.app.voice.VoiceIntent
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
    // Phase 4 — on-device LLM fallback for anything the IntentInterpreter grammar can't place.
    // Stub unless built with -PenableLlm=true AND a model file is side-loaded (see
    // MediaPipeLlmAssistant); every call is a safe no-op otherwise.
    private val llm: LlmAssistant by lazy { LlmAssistants.create(this) }
    @Volatile private var pendingCallName: String? = null
    // Tier-2 safety gate: multilingual-e5 embedding classifier. Noop until the model ships;
    // when ready it intercepts safety/movement questions in ANY phrasing/language before the
    // grammar or the LLM see them.
    private val safetyVectorGate: SafetyVectorGate by lazy { SafetyVectorGates.create(this) }

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
    // Rolling ~1.5 s window of fused hazard states — the safety gate answers "is it clear" from
    // the WORST state in this window, so a drop-off that flickered mid-question still counts
    // (temporal hysteresis, deep-research layer 1).
    private val hazardWindow = ArrayDeque<Pair<Long, ai.secondsense.app.inference.decode.HazardState?>>()
    private val HAZARD_WINDOW_MS = 1_500L

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
    // Most recent upright frame, stashed unconditionally for the "read this" voice intent.
    @Volatile private var lastFrameForOcr: android.graphics.Bitmap? = null
    private var tts: TextToSpeech? = null
    @Volatile private var ttsReady = false
    private var sceneGestures: GestureDetector? = null

    // --- Blind-first interaction layer ---
    private val audioManager by lazy { getSystemService(Context.AUDIO_SERVICE) as AudioManager }
    private val uiPrefs by lazy { getSharedPreferences("secondsense_ui", Context.MODE_PRIVATE) }
    @Volatile private var talkbackOn = false
    @Volatile private var paused = false
    private var lastAnnouncement: String? = null
    private var lastBigStatusMs = 0L
    // multi-finger tap tracking (GestureDetector only reports single-pointer gestures)
    private var gStartMs = 0L
    private var gStartX = 0f
    private var gStartY = 0f
    private var gMaxPointers = 1
    private var gMoved = false
    // spoken settings menu
    private var menuOpen = false
    private var menuIndex = 0
    private var cueVolLevel = 2   // 0 low, 1 med, 2 high
    private var voiceSpeedLevel = 1 // 0 slow, 1 normal, 2 fast

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
    // "find X" demo safety-net: when the goal was set, and whether it's ever been seen.
    @Volatile private var goalSetAtMs = 0L
    @Volatile private var goalEverSeen = false

    // Problem Statement 6 — thermal throttling / deterministic latency in the closed harness.
    private val thermalGovernor by lazy { ThermalGovernor(walkingSupplier = { pedometer.isWalking }) }
    @Volatile private var perceptionEnabled = true
    @Volatile private var yamnetWanted = true
    @Volatile private var lowResActive = false

    // Activity context (Walking/Standing/Sitting/Home/Transit/Conversation) — reconfigures the
    // whole pipeline; merged with the thermal policy in applyEffectivePolicy().
    private val contextManager = ContextManager()
    // Phase 2: infers Walking/Standing/Transit from step cadence + accel vibration and feeds
    // contextManager.suggest() (15 s grace, yields to anything the user set). Permission-free.
    private val contextAutoDetector by lazy {
        ContextAutoDetector(
            manager = contextManager,
            walkingSupplier = { pedometer.isWalking },
            vibrationSupplier = { pedometer.vibrationLevel },
        )
    }
    private var lastFloorMs = 0L

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

    // Phase 4 "call <contact>" — CALL_PHONE + READ_CONTACTS, requested only the first time the
    // user actually asks to call someone. On grant, retry the call we stashed.
    private val requestCallPerms =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            val name = pendingCallName ?: return@registerForActivityResult
            pendingCallName = null
            if (result.values.all { it }) announce(PhoneActions.call(this, name))
            else announce("I don't have permission to make calls.")
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
            // Phase 4 LLM: heavy (~seconds) model load; no-op stub unless -PenableLlm + model file.
            try {
                llm.initialize()
            } catch (t: Throwable) {
                android.util.Log.w("SecondSense/llm", "llm.initialize() failed", t)
            }
            try {
                safetyVectorGate.initialize()
            } catch (t: Throwable) {
                android.util.Log.w("SecondSense/llm", "safetyVectorGate.initialize() failed", t)
            }
        }

        analyzer = FrameAnalyzer(
            engine,
            onResult = { result -> onFrameResult(result) },
            frameSink = { bmp ->
                lastFrameForOcr = bmp   // kept for a one-shot "read this" even when aux OCR is off
                if (perceptionEnabled) perception.offer(bmp)
            },
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

        // Thermal governor + activity context both feed one merge point (most-conservative wins).
        thermalGovernor.onPolicy = { runOnUiThread { applyEffectivePolicy() } }
        thermalGovernor.onNotice = { msg ->
            speakLocalized(msg, langPrefs.speakHindi, TextToSpeech.QUEUE_ADD, "thermal")
        }
        contextManager.onContext = { _, _ -> runOnUiThread { applyEffectivePolicy() } }
        contextManager.onAnnounce = { msg -> runOnUiThread { announce(msg); haptics.testBuzz() } }
        applyEffectivePolicy()   // apply the default (WALKING) profile from the first frame
        // --- Blind-first gesture surface. dispatchTouchEvent (below) feeds this every touch
        // before any child view can eat it. Suppressed entirely when TalkBack is on (it drives
        // the visible fallback buttons instead) and while the spoken menu owns input. ---
        sceneGestures = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            private fun surfaceInactive() =
                talkbackOn || binding.adminPanel.visibility == View.VISIBLE

            override fun onDown(e: MotionEvent): Boolean = true

            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                if (surfaceInactive() || gMaxPointers >= 2) return false
                if (menuOpen) return true
                tapToTalk()   // one tap anywhere = wake + listen (Phase 3)
                return true
            }

            override fun onDoubleTap(e: MotionEvent): Boolean {
                if (surfaceInactive() || gMaxPointers >= 2) return false
                if (menuOpen) { menuActivate(); return true }
                narrateScene()   // offline "what's ahead" — no mic, for noisy places
                return true
            }

            override fun onLongPress(e: MotionEvent) {
                if (talkbackOn || menuOpen) return
                if (binding.adminPanel.visibility == View.VISIBLE) return
                if (e.x > rootW() * 0.66f && e.y > rootH() * 0.66f) toggleAdmin() else openSpokenMenu()
            }

            override fun onFling(e1: MotionEvent?, e2: MotionEvent, vx: Float, vy: Float): Boolean {
                if (surfaceInactive()) return false
                if (kotlin.math.abs(vy) < kotlin.math.abs(vx) || kotlin.math.abs(vy) < 900f) return false
                if (menuOpen) { if (vy < 0) menuMove(-1) else menuMove(1) }
                else contextManager.step(moreActive = vy < 0f, System.currentTimeMillis()) // up = more active
                return true
            }
        })

        // TalkBack / explore-by-touch: switch to visible focusable buttons, don't fight it.
        talkbackOn = (getSystemService(Context.ACCESSIBILITY_SERVICE) as? AccessibilityManager)
            ?.isTouchExplorationEnabled == true
        binding.tbButtons.visibility = if (talkbackOn) View.VISIBLE else View.GONE
        binding.talkbackHint.visibility = if (talkbackOn) View.GONE else View.VISIBLE
        binding.btnTbAround.setOnClickListener { narrateScene() }
        binding.btnTbStatus.setOnClickListener { announceStatus() }
        binding.btnTbFind.setOnClickListener { tapToTalk() }
        binding.btnCloseAdmin.setOnClickListener { toggleAdmin() }

        updateBigStatus()
        // First-run: read the gesture help aloud once TTS is up (retry via 3-finger tap / menu).
        if (!uiPrefs.getBoolean("onboarded", false)) {
            uiPrefs.edit().putBoolean("onboarded", true).apply()
            binding.blindSurface.postDelayed({
                announce("Welcome to SecondSense.")
                binding.blindSurface.postDelayed({ speakHelp() }, 1600)
            }, 1200)
        }

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
     * #30 — start the embedded telemetry HTTP server (state.json etc. on port 8085 over the
     * local network). Never crashes the app if the port is busy or no network is up — the
     * assistive pipeline doesn't depend on it. (The on-screen QR/URL were removed; connect a
     * laptop by typing the phone's LAN IP, logged below.)
     */
    private fun startDashboardServer() {
        try {
            dashboardServer = DashboardServer().also { it.start() }
            DashboardServer.localIpAddress(this)?.let {
                android.util.Log.i("SecondSense/dashboard", "telemetry at http://$it:8085/")
            }
        } catch (t: Throwable) {
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
        if (!talkbackOn && binding.adminPanel.visibility != View.VISIBLE) trackMultiFingerTap(ev)
        sceneGestures?.onTouchEvent(ev)   // observe every touch; don't consume
        return super.dispatchTouchEvent(ev)
    }

    /** GestureDetector ignores multi-pointer gestures — detect quick 2- and 3-finger taps here. */
    private fun trackMultiFingerTap(ev: MotionEvent) {
        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                gStartMs = System.currentTimeMillis(); gStartX = ev.x; gStartY = ev.y
                gMaxPointers = 1; gMoved = false
            }
            MotionEvent.ACTION_POINTER_DOWN ->
                gMaxPointers = maxOf(gMaxPointers, ev.pointerCount)
            MotionEvent.ACTION_MOVE ->
                if (kotlin.math.hypot((ev.x - gStartX).toDouble(), (ev.y - gStartY).toDouble()) > 60) gMoved = true
            MotionEvent.ACTION_UP -> {
                val quick = System.currentTimeMillis() - gStartMs < 320 && !gMoved
                if (quick && gMaxPointers >= 3) speakHelp()
                else if (quick && gMaxPointers == 2) { if (menuOpen) closeSpokenMenu() else togglePause() }
            }
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean = when (keyCode) {
        KeyEvent.KEYCODE_VOLUME_UP -> { repeatLast(); true }
        KeyEvent.KEYCODE_VOLUME_DOWN -> { cycleCueVolume(); true }
        else -> super.onKeyDown(keyCode, event)
    }

    // --- blind-first handlers -----------------------------------------------------------

    private fun rootW() = binding.blindSurface.width.takeIf { it > 0 } ?: resources.displayMetrics.widthPixels
    private fun rootH() = binding.blindSurface.height.takeIf { it > 0 } ?: resources.displayMetrics.heightPixels

    /** Speak + remember, so Volume-Up can repeat it. */
    private fun announce(text: String) {
        lastAnnouncement = text
        // When Hindi is the preference, actually translate the English UI line before speaking
        // it — otherwise a hi-IN TTS voice just mangles English words. Falls back to the
        // English string if the on-device model pair isn't downloaded.
        if (langPrefs.speakHindi && langPrefs.translateSigns) {
            translator.localize(text, sourceIsDevanagari = false, wantHindi = true, translateEnabled = true) { spoken, isHi ->
                speakLocalized(spoken, isHi, TextToSpeech.QUEUE_FLUSH, "announce")
            }
        } else {
            speakLocalized(text, langPrefs.speakHindi, TextToSpeech.QUEUE_FLUSH, "announce")
        }
    }

    private fun repeatLast() = announce(lastAnnouncement ?: "Nothing to repeat.")

    private fun toggleSonify() {
        binding.switchSonify.isChecked = !binding.switchSonify.isChecked   // fires existing listener
        haptics.testBuzz()
        announce(if (binding.switchSonify.isChecked) "Cues on." else "Cues off.")
        updateBigStatus()
    }

    private fun togglePause() {
        paused = !paused
        if (paused) { cueEngine.stop(); cueEngine.update(null); announce("Paused.") }
        else { if (sonifying) cueEngine.start(); announce("Resumed.") }
        updateBigStatus()
    }

    private fun toggleWalkExplore() {
        val toExplore = !modeController.acceptsVoiceCommands
        binding.switchMode.isChecked = toExplore   // fires existing listener -> modeController.set
        announce(if (toExplore) "Explore mode. Stopped." else "Walk mode.")
        updateBigStatus()
    }

    private fun enterExploreAndListen() {
        if (!modeController.acceptsVoiceCommands) binding.switchMode.isChecked = true
        announce("Listening. Say what you are looking for.")
        if (hasMicPermission()) startVoiceCapture() else requestMic.launch(Manifest.permission.RECORD_AUDIO)
    }

    /**
     * Phase 3 primary gesture: one tap anywhere on the screen wakes the assistant. A short
     * earcon + buzz confirm it's listening; the transcript is then routed through
     * [IntentInterpreter] -> [handleVoiceIntent]. No mode change here — the intent decides
     * (Find/Recall flip to Scan/Seek, everything else runs in place).
     */
    private fun tapToTalk() {
        if (menuOpen) return
        audio.testTone()
        haptics.testBuzz()
        announce("Listening.")
        if (hasMicPermission()) startVoiceCapture()
        else requestMic.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun toggleAdmin() {
        val show = binding.adminPanel.visibility != View.VISIBLE
        binding.adminPanel.visibility = if (show) View.VISIBLE else View.GONE
        announce(if (show) "Admin panel open." else "Admin panel closed.")
    }

    private fun announceStatus() {
        val mode = if (paused) "paused" else contextManager.context.name.lowercase()
        val cues = if (sonifying && !paused && contextManager.profile.sonification) "cues on" else "cues off"
        val cam = camHealthShort()
        val batt = batteryPercent()
        val seen = lastResult?.detections?.mapNotNull { it.label }?.distinct()?.take(3)
            ?.joinToString(", ")?.ifEmpty { null }
        val goal = vectorToGoal.activeGoal
        announce(buildString {
            append("$mode, $cues, $cam, battery $batt percent. ")
            if (goal != null) append("Looking for $goal. ")
            append(if (seen != null) "I can see $seen." else "Nothing named ahead.")
        })
    }

    private var battPctCache = -1
    private var battPctAtMs = 0L
    private fun batteryPercent(): Int {
        val now = System.currentTimeMillis()
        if (now - battPctAtMs < 15_000L && battPctCache >= 0) return battPctCache
        val i = registerReceiver(null, android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED))
        val lvl = i?.getIntExtra(android.os.BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = i?.getIntExtra(android.os.BatteryManager.EXTRA_SCALE, 100) ?: 100
        battPctCache = if (lvl < 0) -1 else (lvl * 100 / scale)
        battPctAtMs = now
        return battPctCache
    }

    private fun camHealthShort(): String = when (lastCamHealth) {
        ai.secondsense.app.inference.CameraHealth.OK -> "camera ok"
        ai.secondsense.app.inference.CameraHealth.DIM -> "camera dim"
        ai.secondsense.app.inference.CameraHealth.BLOCKED -> "camera blocked"
        ai.secondsense.app.inference.CameraHealth.MISALIGNED -> "camera angle"
    }

    private fun cycleCueVolume() {
        cueVolLevel = (cueVolLevel + 1) % 3
        val max = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        val target = when (cueVolLevel) { 0 -> max * 35 / 100; 1 -> max * 70 / 100; else -> max }
        runCatching { audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, target.coerceAtLeast(1), 0) }
        announce("Loudness ${arrayOf("low", "medium", "high")[cueVolLevel]}.")
    }

    private fun cycleVoiceSpeed() {
        voiceSpeedLevel = (voiceSpeedLevel + 1) % 3
        tts?.setSpeechRate(arrayOf(0.8f, 1.0f, 1.35f)[voiceSpeedLevel])
        announce("Voice speed ${arrayOf("slow", "normal", "fast")[voiceSpeedLevel]}.")
    }

    private fun speakHelp() {
        announce(
            "One tap anywhere wakes me. After the beep, just say what you want. " +
                "For example: find my keys. What's ahead. Read this. Where did I leave my phone. " +
                "Status. I'm sitting. Stop the beeping. Speak Hindi. Or say repeat. " +
                "Double tap for a quick look ahead without the microphone. " +
                "Two finger tap to pause or resume. Three finger tap for this help. " +
                "Swipe up or down to change mode. Long press for the settings menu. " +
                "Volume up repeats the last message. Volume down changes loudness.",
        )
    }

    private fun updateBigStatus() {
        val now = System.currentTimeMillis()
        if (now - lastBigStatusMs < 400 && !menuOpen) return
        lastBigStatusMs = now
        val goal = vectorToGoal.activeGoal
        val big = when {
            menuOpen -> "MENU"
            paused -> "PAUSED"
            goal != null -> "FINDING\n${goal.uppercase()}"
            else -> contextManager.context.name   // WALKING / STANDING / SITTING / HOME / TRANSIT / CONVERSATION
        }
        val cueStr = if (sonifying && !paused && contextManager.profile.sonification) "cues on" else "cues off"
        val sub = "$cueStr · ${camHealthShort()} · batt ${batteryPercent()}%"
        runOnUiThread {
            binding.statusBig.text = big
            binding.statusSub.text = sub
        }
    }

    // --- spoken settings menu ----------------------------------------------------------

    private data class MenuItem(val title: String, val value: () -> String, val activate: () -> Unit)

    private val menuItems: List<MenuItem> by lazy {
        listOf(
            MenuItem("Context", { contextManager.context.name.lowercase() }) {
                contextManager.cycle(System.currentTimeMillis())
            },
            MenuItem("Language", { if (langPrefs.speakHindi) "Hindi" else "English" }) {
                binding.switchHindi.isChecked = !binding.switchHindi.isChecked
            },
            MenuItem("Loudness", { arrayOf("low", "medium", "high")[cueVolLevel] }) { cycleCueVolume() },
            MenuItem("Voice speed", { arrayOf("slow", "normal", "fast")[voiceSpeedLevel] }) { cycleVoiceSpeed() },
            MenuItem("Cues", { if (sonifying) "on" else "off" }) { toggleSonify() },
            MenuItem("Calibrate camera now", { "" }) { binding.btnCalibrate.performClick() },
            MenuItem("Read help", { "" }) { speakHelp() },
            MenuItem("Admin panel", { "" }) { toggleAdmin() },
        )
    }

    private fun openSpokenMenu() {
        menuOpen = true
        menuIndex = 0
        updateBigStatus()
        val it = menuItems[0]
        announce("Settings menu. ${it.title}, ${it.value()}. Swipe up or down to move, double tap to change, two finger tap to close.")
    }

    private fun closeSpokenMenu() {
        menuOpen = false
        updateBigStatus()
        announce("Menu closed.")
    }

    private fun menuMove(delta: Int) {
        menuIndex = (menuIndex + delta + menuItems.size) % menuItems.size
        val it = menuItems[menuIndex]
        val v = it.value()
        announce(if (v.isEmpty()) it.title else "${it.title}, $v")
    }

    private fun menuActivate() {
        menuItems[menuIndex].activate()
        val it = menuItems[menuIndex]
        val v = it.value()
        if (v.isNotEmpty()) announce("${it.title} now $v")
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

    /**
     * The single place every cadence/load knob is set — merges the THERMAL policy (harness
     * heat) and the CONTEXT profile (what the user is doing). Most-conservative wins per knob:
     * whichever says "slower / off" is applied. Called on any thermal or context change.
     */
    private fun applyEffectivePolicy() {
        val t = thermalGovernor.policy
        val c = contextManager.profile

        analyzer.processEveryN = maxOf(t.frameEveryN, c.detectEveryN)
        engine.setDepthEveryN(maxOf(t.depthEveryN, c.depthEveryN))
        engine.setHazardEveryN(if (!c.hazardEnabled) 0 else maxOf(t.hazardEveryN, 1))
        perceptionEnabled = t.auxEnabled && c.auxPerception

        val wantYamnet = t.yamnetEnabled && c.auxPerception
        if (wantYamnet != yamnetWanted) {
            yamnetWanted = wantYamnet
            if (wantYamnet) { if (hasMicPermission()) startHazardDetection() } else hazardDetector.stop()
        }
        if (t.lowRes != lowResActive) {
            lowResActive = t.lowRes
            if (hasCameraPermission()) startCamera()
        }
        // The context can mute the continuous OBSTACLE cue loop — but an explicit spoken goal
        // ("find my chair") must still steer, whatever the activity context. A live voice /
        // memory goal keeps the cue engine running.
        val voiceGoalLive = vectorToGoal.isActive || memoryNavActive
        val cuesLive = (c.sonification || voiceGoalLive) && sonifying && !paused
        if (!cuesLive) { cueEngine.stop(); cueEngine.update(null) } else cueEngine.start()

        updateBigStatus()
        android.util.Log.i(
            "SecondSense/context",
            "effective ctx=${c.label} thermal=${t.label}: frame/${analyzer.processEveryN} " +
                "depth+ hazard=${c.hazardEnabled} aux=$perceptionEnabled cues=$cuesLive",
        )
    }

    /** "What's around me": speak a one-sentence description of the current frame (offline). */
    private fun narrateScene() {
        val text = SceneNarrator.describe(lastResult)
        lastAnnouncement = text   // so Volume-Up can repeat it
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
        voiceCapture.capture { _, transcript, recognizerReady ->
            if (hasMicPermission()) startHazardDetection()
            // Tier-2 safety gate FIRST: a semantic classifier catches "is it safe / can I move"
            // in any phrasing or language and forces the deterministic deflection, bypassing
            // both the grammar and the LLM. Noop until the embedding model ships.
            if (recognizerReady && !transcript.isNullOrBlank() &&
                safetyVectorGate.isReady() && safetyVectorGate.isSafetyQuery(transcript)
            ) {
                android.util.Log.i("SecondSense/voice", "heard=\"$transcript\" -> [vector-gate] SafetyCheck")
                runOnUiThread { handleVoiceIntent(VoiceIntent.SafetyCheck, true) }
                return@capture
            }
            // Phase 3: the whole transcript -> one action from a closed set (offline rule grammar).
            // Whatever the grammar can't place becomes VoiceIntent.Unknown -> Phase 4 LLM.
            val intent = IntentInterpreter.interpret(transcript, contextManager.context)
            android.util.Log.i("SecondSense/voice", "heard=\"$transcript\" -> $intent")
            runOnUiThread { handleVoiceIntent(intent, recognizerReady) }
        }
    }

    /** Execute one interpreted [VoiceIntent] against the existing handlers. */
    private fun handleVoiceIntent(intent: VoiceIntent, recognizerReady: Boolean) {
        if (!recognizerReady && intent !is VoiceIntent.Help) {
            Toast.makeText(
                this,
                "Voice model not loaded — build with -PenableSherpa + add the KWS model (see android/app/src/sherpa/README.md)",
                Toast.LENGTH_LONG,
            ).show()
            return
        }
        when (intent) {
            is VoiceIntent.Find -> {
                modeController.set(OperatingMode.SCAN_SEEK)   // grounding runs only in scan/seek
                vectorToGoal.setGoal(intent.target)
                memoryNavActive = false
                goalSetAtMs = System.currentTimeMillis(); goalEverSeen = false
                sonifying = true
                binding.switchSonify.isChecked = true
                cueEngine.start()                            // context may have stopped it
                announce("Looking for ${intent.target}.")
            }
            is VoiceIntent.Recall -> {
                modeController.set(OperatingMode.SCAN_SEEK)
                goalSetAtMs = System.currentTimeMillis(); goalEverSeen = false
                recallObject(intent.target)
                cueEngine.start()
            }
            VoiceIntent.Describe -> narrateScene()
            VoiceIntent.SafetyCheck -> speakSafety()
            VoiceIntent.ReadText -> readTextNow()
            is VoiceIntent.SwitchContext -> contextManager.set(intent.context, System.currentTimeMillis())
            VoiceIntent.Status -> announceStatus()
            VoiceIntent.RepeatLast -> repeatLast()
            is VoiceIntent.Cues ->
                if (binding.switchSonify.isChecked != intent.on) toggleSonify()
                else announce(if (intent.on) "Cues already on." else "Cues already off.")
            VoiceIntent.Pause -> if (!paused) togglePause() else announce("Already paused.")
            VoiceIntent.Resume -> if (paused) togglePause() else announce("Not paused.")
            VoiceIntent.CancelSeek -> {
                vectorToGoal.setGoal(null)
                memoryNavActive = false
                modeController.set(OperatingMode.FLOW)
                announce("Stopped looking.")
            }
            VoiceIntent.Help -> speakHelp()
            is VoiceIntent.CallContact -> {
                val need = arrayOf(Manifest.permission.CALL_PHONE, Manifest.permission.READ_CONTACTS)
                if (need.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
                    announce(PhoneActions.call(this, intent.name))
                } else {
                    pendingCallName = intent.name
                    requestCallPerms.launch(need)
                }
            }
            is VoiceIntent.SetTimer -> announce(PhoneActions.setTimer(this, intent.seconds))
            is VoiceIntent.SetLanguage -> {
                langPrefs.speakHindi = intent.hindi
                binding.switchHindi.isChecked = intent.hindi
                tts?.setLanguage(if (intent.hindi) Locale("hi", "IN") else Locale.US)
                speakLocalized(
                    if (intent.hindi) "अब हिंदी में बोलूँगा" else "Speaking English now.",
                    intent.hindi, TextToSpeech.QUEUE_FLUSH, "lang",
                )
            }
            is VoiceIntent.Unknown -> handleUnknown(intent.transcript)
        }
        updateBigStatus()
    }

    /**
     * The grammar couldn't place it. If the Phase 4 LLM is loaded, let it reason over the
     * transcript + a snapshot of the scene; otherwise say so honestly.
     */
    private fun handleUnknown(transcript: String) {
        if (transcript.isBlank()) { announce("I didn't catch that."); return }
        if (!llm.isReady()) {
            announce("I didn't understand. Tap and say help for a list.")
            return
        }
        announce("Thinking.")
        val brief = sceneBrief()
        android.util.Log.i("SecondSense/llm", "brief: $brief")
        thread(name = "llm-resolve") {
            val res = runCatching { llm.resolve(transcript, brief) }.getOrNull()
            runOnUiThread {
                when (res) {
                    is LlmResolution.Action -> handleVoiceIntent(res.intent, recognizerReady = true)
                    is LlmResolution.Speak -> announce(res.text)
                    LlmResolution.Defer -> speakSafety()   // model flagged safety / green-lit movement
                    null -> announce("I couldn't work that out.")
                }
            }
        }
    }

    /**
     * DETERMINISTIC, legally-vetted answer to any "is it safe / can I move / is it clear"
     * question (deep-research layers 1 + 3). Never the LLM. Never uses "safe" as an affirmative.
     * Fixed template + the current sensor state (worst in the 1.5 s hysteresis window) + an
     * explicit hand-back of agency to the cane + traffic sounds. English or the vetted Hindi
     * string per the listener's preference — spoken directly, NOT routed through the translator.
     */
    private fun speakSafety() {
        val hi = langPrefs.speakHindi
        val text = safetyAnswer(hi)
        lastAnnouncement = text
        android.util.Log.i(
            "SecondSense/llm",
            "SAFETY deflection -> \"$text\" | dets=${lastResult?.detections?.map { "${it.label}@${"%.2f".format(it.proximity)}" }} " +
                "haz=${synchronized(hazardWindow) { hazardWindow.map { it.second?.name } }} cam=${lastResult?.cameraHealth}",
        )
        speakLocalized(text, hi, TextToSpeech.QUEUE_FLUSH, "safety")
    }

    private fun safetyAnswer(hindi: Boolean): String {
        val r = lastResult
        val camBad = r == null || !(r.cameraHealth == CameraHealth.OK || r.cameraHealth == CameraHealth.DIM)
        val worst = synchronized(hazardWindow) { SafetyGate.mostSevere(hazardWindow.map { it.second }) }
        // A safety readout reports EVERYTHING in view, with a distance qualifier — not just
        // point-blank objects. Sorted nearest-first.
        val seen = r?.detections
            ?.filter { it.label != null && it.proximity >= 0.22f }
            ?.sortedByDescending { it.proximity }
            ?.map { d ->
                val l = d.label!!
                val where = d.box.centerX.let { cx -> if (cx < 0.4f) "on your left" else if (cx > 0.6f) "on your right" else "ahead" }
                val how = if (d.proximity >= 0.6f) "$l close $where" else "$l $where"
                how
            }
            ?.distinct()?.take(3).orEmpty()

        // Sensor-state clause, injected into the fixed template.
        val stateEn: String
        val stateHi: String
        when {
            camBad -> { stateEn = "the camera cannot see clearly right now"; stateHi = "कैमरा अभी साफ़ नहीं देख पा रहा है" }
            worst == ai.secondsense.app.inference.decode.HazardState.DROP_CONFIRMED -> {
                stateEn = "there is a drop-off directly ahead"; stateHi = "आगे एक गड्ढा या सीढ़ी है"
            }
            worst == ai.secondsense.app.inference.decode.HazardState.SCENE_NOT_TRAVERSABLE -> {
                stateEn = "the path ahead looks blocked"; stateHi = "आगे का रास्ता बंद लग रहा है"
            }
            worst == ai.secondsense.app.inference.decode.HazardState.SENSOR_BLOCKED -> {
                stateEn = "the sensors are blocked and cannot tell"; stateHi = "सेंसर बंद हैं और कुछ बता नहीं सकते"
            }
            worst == ai.secondsense.app.inference.decode.HazardState.POSSIBLE_DROP -> {
                stateEn = "there may be a step or drop ahead"; stateHi = "आगे सीढ़ी या गड्ढा हो सकता है"
            }
            seen.isNotEmpty() -> {
                stateEn = "the camera sees ${seen.joinToString(", ")}"
                stateHi = "कैमरे को दिख रहा है: ${seen.joinToString(", ")}"
            }
            else -> { stateEn = "no objects are detected ahead"; stateHi = "आगे कोई वस्तु नहीं दिख रही" }
        }
        return if (hindi) {
            "मैं यह तय नहीं कर सकता कि आगे बढ़ना सुरक्षित है या नहीं। सेंसर बता रहे हैं कि $stateHi, " +
                "लेकिन आगे बढ़ने के लिए आपको अपनी छड़ी, ट्रैफ़िक की आवाज़ और अपने निर्णय पर निर्भर रहना होगा।"
        } else {
            "I cannot decide whether it is safe to move. The sensors show that $stateEn, " +
                "but you must rely on your white cane, traffic sounds, and your own judgement to proceed."
        }
    }

    /** A compact snapshot of what the pipeline currently perceives, for the LLM prompt. */
    private fun sceneBrief(): SceneBrief {
        val r = lastResult
        val objs = r?.detections
            ?.sortedByDescending { it.proximity }
            ?.mapNotNull { d ->
                d.label?.let { lbl ->
                    val cx = d.box.centerX
                    when { cx < 0.4f -> "$lbl to the left"; cx > 0.6f -> "$lbl to the right"; else -> "$lbl ahead" }
                }
            }
            ?.distinct()?.take(4).orEmpty()
        val hazard = when (r?.hazardState?.name) {
            "DROP_CONFIRMED" -> "drop-off ahead"
            "POSSIBLE_DROP" -> "possible drop-off ahead"
            "SCENE_NOT_TRAVERSABLE" -> "path blocked"
            else -> null
        }
        return SceneBrief(
            context = contextManager.context.name.lowercase(),
            objectsAhead = objs,
            hazard = hazard,
            batteryPct = batteryPercent(),
            camera = camHealthShort().removePrefix("camera ").trim(),
            lastSpoken = lastAnnouncement,
        )
    }

    /** One-shot OCR of the latest frame for a spoken "read this" (Phase 3). */
    private fun readTextNow() {
        val src = lastFrameForOcr
        if (src == null || src.isRecycled) { announce("No camera image yet."); return }
        val cfg = src.config ?: android.graphics.Bitmap.Config.ARGB_8888
        val copy = runCatching { src.copy(cfg, false) }.getOrNull()
        if (copy == null) { announce("Can't read right now."); return }
        announce("Reading.")
        perception.readNow(copy) { text, isDeva ->
            runOnUiThread {
                if (text.isNullOrBlank()) announce("I don't see any text.")
                else onSignRead(text, isDeva)
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

        // SAFETY FLOOR — runs in EVERY context, even Sitting/Transit where the hazard pipeline
        // is off: something within arm's reach and closing fast still earns one haptic.
        val imminent = result.detections.any { it.proximity >= 0.90f && it.approaching >= 0.12f }
        if (imminent && System.currentTimeMillis() - lastFloorMs > 3000L) {
            lastFloorMs = System.currentTimeMillis()
            haptics.possibleDrop()
        }

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
        // Specular Trap veto C (safety re-escalation): a chromaticity/flow veto may have held a
        // real drop at POSSIBLE_DROP. If the BAROMETER independently confirms a descent, trust
        // physics over the optical vetoes and treat it as confirmed.
        val hazardState = if (result.hazardState == ai.secondsense.app.inference.decode.HazardState.POSSIBLE_DROP &&
            barometer.descendingConfirmed()
        ) ai.secondsense.app.inference.decode.HazardState.DROP_CONFIRMED else result.hazardState
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
        synchronized(hazardWindow) {
            val now = System.currentTimeMillis()
            hazardWindow.addLast(now to hazardState)
            while (hazardWindow.isNotEmpty() && now - hazardWindow.first().first > HAZARD_WINDOW_MS) {
                hazardWindow.removeFirst()
            }
        }

        // PHASE 4 (#27/#28) — voice goal-seeking, closed-vocab TFLite path. When a spoken goal
        // is active AND we're stopped (SCAN_SEEK, per #25), steer toward the matching COCO
        // detection instead of cueing the nearest obstacle. On arrival: a distinct haptic,
        // clear the goal, fall back to the obstacle spine.
        val goalMatch = if (modeController.acceptsVoiceCommands)
            GoalGrounding.match(result.detections, vectorToGoal.activeGoal) else null
        // Demo safety-net: a spoken Find target that the detector can't recognise (not a COCO
        // class, e.g. "keys") would otherwise just sit there silently. If it's never once been
        // seen within ~7 s, say so and drop it instead of leaving the user waiting.
        if (vectorToGoal.isActive && !memoryNavActive) {
            if (goalMatch != null) goalEverSeen = true
            else if (!goalEverSeen && goalSetAtMs > 0 && System.currentTimeMillis() - goalSetAtMs > 7_000L) {
                val g = vectorToGoal.activeGoal
                vectorToGoal.setGoal(null); goalSetAtMs = 0L
                runOnUiThread {
                    announce("I can't see a $g. I can only spot common things like people, chairs, doors, bottles and bags.")
                    modeController.set(OperatingMode.FLOW)
                }
            }
        }
        val goalCue = goalMatch?.let { m ->
            val prox = calibration.apply(m.proximity)
            if (vectorToGoal.hasArrived(m.box, prox)) {
                haptics.arrived()
                val reached = vectorToGoal.activeGoal
                vectorToGoal.setGoal(null)
                runOnUiThread {
                    announce("You've reached the $reached.")
                    modeController.set(OperatingMode.FLOW)
                }
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
        // A live spoken/memory goal steers regardless of the context's sonification setting.
        val voiceGoalLive = vectorToGoal.isActive || memoryNavActive
        if (sonifying && !paused && (contextManager.profile.sonification || voiceGoalLive)) {
            cueEngine.start()
            cueEngine.update(target)
        }
        if (vectorToGoal.isActive && frameCount % 20 == 0L) {
            android.util.Log.i(
                "SecondSense/goal",
                "goal=${vectorToGoal.activeGoal} mode=${modeController.mode} " +
                    "match=${goalMatch?.label} cue=${goalCue != null} " +
                    "dets=${result.detections.mapNotNull { it.label }}",
            )
        }
        publishDashboardState(result, target)
        updateBigStatus()

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
        contextAutoDetector.stop()
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
        contextAutoDetector.start()
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
        runCatching { llm.close() }
        runCatching { safetyVectorGate.close() }
    }
}
