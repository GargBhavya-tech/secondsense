# SecondSense — On-Device App (`android/`)

Build-map **ticket #6: On-Device App Skeleton + Camera Capture** (Phase 2, 🔴 RED-OK).
Native Android (Kotlin). This is the app that runs on the chest-mounted iQOO 15.

## What this is

The Phase 2 skeleton + Phase 1 TFLite test-path, fully integrated. Gives you:

- **Camera-in** — CameraX live frame stream (rear camera, RGBA_8888, center-crop toggle).
- **Audio-out hook** — stereo `AudioTrack`, test tone on tap. Structured for HRTF pan
  (#18) + pulse-rate distance (#19) + identity pitch (#20), with the Bible's §5.1
  invariant enforced by the API shape: **pitch never encodes distance.**
- **Haptic-out hook** — `VibratorManager` with **graded amplitude** from day one, because
  Bible §5.2 makes haptics a *primary* proximity channel, not a <0.5m backstop (#21).
- **Swappable inference engine** — everything talks to the `InferenceEngine` interface:
  - `MockInferenceEngine` — scripts a realistic scene (drifting target, oscillating
    proximity, cycling identity, WHITE→BLUE→RED tiers). Demo always works, even with no
    models bundled.
  - **`TfliteInferenceEngine`** ✅ — runs the REAL `yolov11_det.tflite` + `depth_anything_v2.tflite`
    on CPU/XNNPACK (or GPU delegate opt-in). Works on ANY Android 11+ device or emulator.
  - `QnnInferenceEngine` — not yet built; reuses the shared `inference/decode/` layer unchanged.
- **Shared decode layer** (`inference/decode/`) — runtime-agnostic; used by both TFLite and
  the future QNN engine:
  - `Preprocess.kt` — letterbox + float buffer + box back-mapping.
  - `YoloDecoder.kt` — layout-sniffing decode + NMS (handles both qai-hub output formats).
  - `DepthSampler.kt` — per-box relative proximity from depth map (inverse-depth aware).
  - `MotionTracker.kt` — coarse moving / approaching flags.
  - `CocoLabels.kt` — 80 COCO classes → SecondSense icon vocab.
  - `RawTensor.kt` — shared flat tensor type.
- **`EngineConfig.kt`** — one-line switch: `Kind.MOCK` / `Kind.TFLITE` / `Kind.QNN`.

### To activate real models right now

1. Open `inference/EngineConfig.kt`
2. Change: `val KIND: Kind = Kind.TFLITE`
3. Sync + run. Watch logcat tag `SecondSense/tflite` for real model shapes.

### Ticket #6 done-condition (from the build map)
> camera-in, audio-out, and haptic-out all work on-device with no laptop attached;
> a test tone + a test vibration both fire on a button tap.

✅ Met + extended: launch → live preview + HUD counting frames → **Test Tone** and **Test Buzz** buttons → real YOLO + Depth inference on any device via TFLite engine.


## Build & run

Open the `android/` folder in **Android Studio** (Koala/2024.1+), let it sync, and run on
a device or emulator (minSdk 30). Or from the CLI once you've added the Gradle wrapper:

```bash
cd android
# first time only: generate the wrapper with a local Gradle 8.7
gradle wrapper --gradle-version 8.7      # or open in Android Studio, which handles this
./gradlew :app:assembleDebug
./gradlew :app:installDebug              # with a device/emulator attached
```

> The Gradle **wrapper JAR** and `gradlew` scripts aren't committed (they're binary /
> environment-specific). Android Studio regenerates them on first open, or run the
> `gradle wrapper` line above. `local.properties` (SDK path) is likewise machine-local.

You do **not** need any Qualcomm token or converted model to build and run this — the mock
engine covers it. Airplane mode works too; nothing here touches the network.

## Where the NPU drops in (ticket #8 / #12)

When `convert.py` produces the first `qnn_context_binary` (YOLOv11, then Depth-Anything-V2):

1. Add `src/main/cpp/` with the QNN load/run code and a `CMakeLists.txt`.
2. Uncomment the `externalNativeBuild { cmake { … } }` block in `app/build.gradle.kts`.
3. Write `QnnInferenceEngine : InferenceEngine` that loads the `.bin` from assets and
   returns the **same** `FrameResult` shape the mock does.
4. In `MainActivity`, change one line:
   ```kotlin
   private val engine: InferenceEngine = MockInferenceEngine()
   // ->  QnnInferenceEngine(context, "yolov11_det.bin", "depth_anything_v2.bin")
   ```

Nothing in the camera loop, the output channels, the targeting logic, or the UI changes.
That is the entire point of the `InferenceEngine` seam.

## Phase 3 — Sonification & Haptics (tickets #18–#22), the crown jewel

Built and wired against the mock, so you can **hear** the whole cue engine before any
model or the phone exists. Flip the **Sonification** switch in the app: the HUD shows the
resolved cue (identity · direction L/C/R · proximity · tier) and the cue loop renders it.

The three orthogonal channels (Bible §5.1), enforced by the code's shape:

| Dimension | Channel | Where |
|---|---|---|
| Direction (azimuth) | equal-power stereo pan | `AudioOutput.playMono(pan=…)` (#18) |
| Distance/urgency | **pulse repetition rate** (faster = closer) | `CueEngine.intervalMsFor()` (#19) |
| Identity | auditory icon / spearcon **timbre** | `AuditoryIcon` + `Spearcon` (#20) |
| Proximity (parallel) | **graded haptics — primary** | `HapticOutput.proximityPulse()` (#21) |
| Confidence | grainy **texture**, never silence | `CueEngine.applyTierTexture()` (§5.3) |

`CueEngine` (#22) runs a continuous loop on its own thread; the vision side just posts the
latest `CueTarget` each frame via `update()`. **Distance changes the RATE, never the pitch;
identity changes the TIMBRE, never the rate; direction changes the PAN, nothing else.** That
one rule is the product.

The channel math is verified numerically (monotonic rate, equal-power pan, provable
orthogonality, all target-selection cases) — see `app/src/test/…/TargetSelectorTest.kt`
and run `./gradlew :app:testDebugUnitTest`.

### #23 — Uncertainty / self-trust layer (WHITE / BLUE / RED)

`TierClassifier` **derives** the tier from the real signal — detection score + whether
depth is present + whether a class came back — rather than trusting an upstream guess, so
it behaves identically once the QNN engine replaces the mock. It applies **hysteresis**
(higher bar to promote than to demote) and a 2-frame stability requirement, so the audio
texture doesn't stutter between clean and grainy at threshold boundaries.

The three tiers are distinguishable by ear:

| Tier | Meaning | Sound |
|---|---|---|
| WHITE | "I see it, I know what it is" | clean auditory icon, full gain |
| BLUE | "Something's there, unsure what" | same icon + grainy texture, slightly lower gain |
| RED | "Something's there, can't name it" | neutral tick, proximity-only, **identity claim dropped** |

`selectWithTier()` stamps the smoothed tier onto the `CueTarget` and, on RED, **nulls the
label** — the system refuses to claim an identity it doesn't have (§5.3), rather than going
silent or faking confidence. The tier is shown per-cue on the HUD via a colored badge (the
full multi-pane dashboard is #30). State machine verified in `TierClassifierTest.kt`.

### #24 — Graceful-degradation ladder

`DegradationLadder.decide()` maps `(tier, proximity, depth, label)` to exactly one audio
rung, TOTALLY — every input lands somewhere, so there's never a dead end:

1. **FULL** — icon + pan + pulse rate (WHITE + depth + label).
2. **PROXIMITY** — drop identity, keep proximity pulse + uncertainty texture (any depth).
3. **SILENT_AUDIO** — nothing audible this frame (no depth).

Crucially, **PANIC is an independent floor, not the bottom of the chain**: a close object
(≥ `PANIC_PROXIMITY`) fires an unmistakable double-buzz *in addition to* whatever audio rung
applies — so a depth-only, unnamed obstacle still gets PROXIMITY audio **and** a panic haptic.
The system can always steer you off a wall even when it can't name it. Verified total in
`DegradationLadderTest.kt`.

### #25 — Two operating modes (flow vs scan/seek)

`ModeController` is the single source of truth for the §4 mode split:

- **FLOW** (walking): center-crop ON, one primary cue, voice off — sparse and urgent-only.
- **SCAN_SEEK** (stopped): whole frame, a small handful of cues, **voice goal-seeking on**
  — this is where Phase 4 (#26–#28) lives.

Toggling the mode drives the camera analyzer's crop and gates voice input from one place,
rather than scattering mode logic through the pipeline. Verified in `ModeControllerTest.kt`.

> Spearcons use on-device `TextToSpeech`; for airplane-mode operation the device's TTS
> voice must be installed (Settings → Accessibility → TTS). If a spearcon isn't baked yet,
> the engine falls back to an honest tick — it never goes silent.

## Package layout

```
ai.secondsense.app
├── inference/
│   ├── Types.kt              # BBox, Detection, FrameResult, ConfidenceTier — pipeline contracts
│   ├── InferenceEngine.kt    # the swap seam
│   └── MockInferenceEngine.kt# realistic synthetic scene; no NPU needed
├── camera/
│   └── FrameAnalyzer.kt      # CameraX ImageAnalysis -> InferenceEngine
├── output/
│   ├── AudioOutput.kt        # stereo; pan=direction, playMono for icons; pitch≠distance
│   └── HapticOutput.kt       # graded proximity buzz — PRIMARY channel
├── sonification/             # Phase 3 (#18–#22)
│   ├── CueTarget.kt          # the resolved thing to cue (dir · prox · id · tier)
│   ├── TargetSelector.kt     # #14 center-crop + #15 closest-in-center + static/dynamic; #23 tier stamping
│   ├── TierClassifier.kt     # #23 WHITE/BLUE/RED derived from signal, with hysteresis
│   ├── DegradationLadder.kt  # #24 total fallback rungs + independent panic floor
│   ├── ModeController.kt     # #25 flow vs scan/seek — single source of mode truth
│   ├── AuditoryIcon.kt       # #20 synthesized identity timbres (offline, asset-free)
│   ├── Spearcon.kt           # #20 sped-up-TTS fallback for classes without a bespoke icon
│   ├── AudioHelpers.kt       # WAV read + time-stretch for spearcons
│   └── CueEngine.kt          # #22 the continuous three-channel loop — the crown jewel
└── ui/
    └── MainActivity.kt       # wiring + #6 test buttons + Sonification toggle + debug HUD
```

## Invariants carried from the Bible (do not regress these)

- Laptop is never a runtime compute dependency — the app is self-contained on the phone.
- Depth is **relative proximity**, never metres — the type is `proximity: Float (0..1)`.
- Pitch = identity; **distance = pulse repetition rate**, never pitch (§5.1).
- Haptics are a **primary** graded channel (§5.2), not a panic backstop.
- Confidence tiering (WHITE/BLUE/RED) is first-class on every `Detection` (§5.3).
- Center-crop ~30% in flow mode keeps the cue stream sparse (#14).
