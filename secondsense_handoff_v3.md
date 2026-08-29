SecondSense — Handoff Context (Session 3, first green build + on-phone)
======================================================================

iQOO Hackathon 2026 · Bengaluru City Battle · Aug 29–30. This is the sequel to
`secondsense_handoff_v2.md` (post-Phase 3). It captures everything decided and built in
**this** session — for you, a teammate, or the next chat.

The one-sentence version: **the whole core spine now compiles, passes unit tests, builds a
real APK, and runs the real TFLite models on a physical phone** — and both engine paths
(TFLite live, QNN scaffolded) plus the Phase 4 voice structure now exist in code.

---

## 1. What SecondSense is (unchanged, for reference)

Offline, on-device assistive navigation for blind/low-vision users. Chest-mounted phone
(iQOO 15 is the target; tested this session on a Redmi), four NPU models (YOLOv11,
Depth-Anything-V2, Whisper-Tiny, OWL-ViT), spatial audio + haptics. The defensible IP is the
**sonification**, not the vision pipeline — three orthogonal channels (direction/pan,
distance/pulse-rate, identity/timbre) plus haptics as a *primary* channel and a
WHITE/BLUE/RED confidence-tier layer. Moat: "they describe, we vector" — continuous
voice-goal-seeking, proven offline via a live airplane-mode toggle.

Source docs (all authoritative):
- `secondsense_bible_v4.md` — the why (425 lines, 25 sections)
- `secondsense_build_map_v3.md` — the what/when (43 tickets, Phases 0–7). v3 adds ticket
  #32 (confidence-tier dashboard overlay) vs v2, renumbering old #32–42 → #33–43.
- `secondsense_handoff_v2.md` — the previous session's handoff
- Phase 0 laptop pipeline (`convert.py`, `models.json`, `export_assets/`, `docs/`)

Two model-name corrections are still **unpatched** in the bible source: the Whisper zoo
module is `whisper_tiny` (not `whisper_tiny_en`) and segmentation is `fastsam_x` (not
`fastsam_s`). Cosmetic; `models.json` already has the right names.

---

## 2. The starting point vs. what handoff_v2 described

handoff_v2 said Phase 2 vision (#12/#13/#16/#17) was NOT started and only the mock existed.
That was already stale on arrival: the repo had grown a **real TFLite path** —
`TfliteInferenceEngine` + a runtime-agnostic `inference/decode/` layer (Preprocess,
YoloDecoder, DepthSampler, MotionTracker, CocoLabels, RawTensor) — with both `.tflite` models
bundled in `assets/models/`. So #12–#15 were effectively already real; this session took it
the rest of the way onto a phone and filled the remaining gaps.

---

## 3. What was built / fixed THIS session

### 3a. New tickets built (real, testable, unit-tested)
- **#7 One-tap calibration** — `sonification/Calibration.kt`: snapshots the center-of-frame
  proximity as a forward-clearance baseline; `apply()` re-references proximity against it
  (identity passthrough when uncalibrated). Wired to a **Calibrate** button. Tested.
- **#16 Temporal smoothing** — `sonification/TemporalSmoother.kt`: a target must persist ~3
  consecutive frames (same identity, ~same azimuth) before its cue fires. Gates flicker,
  then tracks live. Tested. (This was the real gap — it never existed before.)
- **#17 Drop-off / negative-obstacle** — `inference/decode/DropOffDetector.kt`: flags a
  downward discontinuity when the immediate-foreground depth band reads *farther* than the
  mid-ground band. Wired through `FrameResult.dropOff` → a distinct escalating
  `HapticOutput.dropOff()` on the rising edge + a `⚠ DROP-OFF` HUD line. Conservative
  thresholds so a flat floor won't trip it.

### 3b. Both engine paths — the "TFLite now, QNN later" ask
- **TFLite path is ACTIVE** (`EngineConfig.KIND = TFLITE`) and confirmed running the real
  `yolov11_det.tflite` + `depth_anything_v2.tflite` on a phone.
- **QNN path is scaffolded and compiles**: `inference/QnnInferenceEngine.kt` is a near-mirror
  of the TFLite engine reusing the identical decode layer; `inference/qnn/QnnBackend.kt` is
  the native-bridge seam with a `StubQnnBackend` (never ready) so `EngineConfig`'s QNN branch
  gracefully falls back to TFLite/MOCK until the JNI bridge + `.bin` binaries are added. Flip
  `KIND = QNN` on the iQOO once the bridge exists; nothing downstream changes.

### 3c. Phase 4 voice (QNN-targeted, since Whisper/OWL-ViT have no TFLite export)
- `voice/VoicePipeline.kt` — `SpeechRecognizer` + `OpenVocabGrounder` interfaces,
  `TargetNoun` extractor (real, tested), `WhisperQnnRecognizer` + `OwlVitQnnGrounder`
  (QNN-backed, stub until bridge).
- `voice/VoiceCommandCapture.kt` (#26) — real mic capture (AudioRecord, mono 16 kHz PCM),
  gated to SCAN_SEEK, feeds the recognizer + noun extraction. Transcription returns
  "not loaded yet" honestly until the QNN model is live.
- `voice/VectorToGoalController.kt` (#28) — steering-cue + arrival logic, tested.
- Wired: a **Find… (voice)** button, RECORD_AUDIO permission, HUD goal line.

### 3d. Build environment — first green Gradle build ever
No real build had ever run before this session. Established a working toolchain on this
machine (see the `secondsense-android-build` memory for the exact commands):
- System JDK is 24 (too new) → build with **Android Studio's bundled JBR (JDK 21)**.
- No committed wrapper / no gradle on PATH → build via **cached Gradle 8.13**.
- Created `android/local.properties` (SDK path), pinned `buildToolsVersion = "35.0.0"`
  (SDK has 35/36/37, not the AGP-default 34), generated wrapper scaffolding.
- Green: `:app:assembleDebug` + `:app:testDebugUnitTest` both pass. APK ≈ **125 MB**
  (bundles the 99 MB depth + 10 MB YOLO models).

### 3e. Latent bugs the real build/phone exposed and I fixed
- `HapticOutput` (×3, incl. `panic()`): called `Vibrator.vibrate(CombinedVibration…)` —
  invalid overload (that's a `VibratorManager` API). Replaced with `vibrate(VibrationEffect)`.
- `CueEngine`: passed `String?` to `Spearcon.get(String)` — null-safety compile error.
- `Spearcon`: `word in buffers` resolved to `ConcurrentHashMap.contains` (a Kotlin error).
- `TierClassifierTest`: called `classify(score)` with 1 of 3 args — **never compiled**; gave
  `classify()` defaults. (These 4 only surfaced because no build had ever run.)
- **UINT8 dtype bug** (found ON the phone): YOLO's class tensor is `1x8400 UINT8`, but the
  engine sized every output buffer as `elements×4` and read them all as floats → garbage
  class indices. Fixed to size/read each output by its real `dataType()`/`numBytes()`.
- **Frame rotation** (correctness): `FrameAnalyzer` handed the engine the raw
  sensor-orientation bitmap; now rotates by `imageInfo.rotationDegrees` to upright, so boxes
  and the pan/azimuth cue are on the correct axis.

### 3f. Performance + UX on the phone
- **NNAPI hardware delegate** in `TfliteInferenceEngine` (offloads to the phone's NPU/GPU/DSP)
  with a clean CPU/XNNPACK fallback. Confirmed on the test phone: both models loaded on NNAPI,
  no fallback — but NNAPI's first-run graph **compile takes ~17 s** (one-time startup cost).
- **Depth every-2-frames** (reuse the last map between) so YOLO/pan stays responsive; depth is
  the heavy model.
- **KEEP_SCREEN_ON** — a hands-free always-on aid must not let the screen sleep (sleep pauses
  the Activity → stops the camera).

### 3g. Housekeeping
- Deleted `tflite_patch/` (verified byte-identical to the main tree — pure redundancy + a
  stray build cache).
- Cleared `logs/*.log` (they embedded a teammate's absolute path `C:\Users\prith\...`; stale,
  regenerable by `convert.py`). No code contained the path.

---

## 4. Verified on a real phone (Redmi 22071219AI, arm64)
- Builds green; unit tests pass.
- APK installs, launches, **no crash**.
- HUD shows `engine: tflite:yolov11+depth` — the **real models**, not the mock.
- Logcat `SecondSense/tflite`: real shapes — YOLO in `1x640x640x3`, out boxes `1x8400x4` +
  scores `1x8400` + classes `1x8400 UINT8`; depth in `1x518x518x3`, out `1x518x518x1`.
  Confirms **YOLO output = Layout A** (boxes+scores+classes), which `YoloDecoder` handles.
- Both models load on the **NNAPI delegate**, no exceptions.

### What could NOT be verified headlessly (needs a human at the phone)
This is a MIUI device that blocks adb from: `pm grant` (camera permission), and
`input keyevent WAKEUP` (waking the screen). When the screen sleeps the Activity pauses and
the camera stops. So **steady-state fps and live detection labels require you to hold/watch
the phone** — grant Camera on-device (done once already), keep it awake (now enforced), point
at a person, and read the HUD's `infer` ms + `dets`.

---

## 5. What's explicitly NOT done yet
- **QNN native bridge** — `QnnBackend` is a stub; the real `src/main/cpp` JNI + the QNN SDK +
  the `.bin` binaries in `assets/models/` are needed to run the NPU path. Everything above the
  bridge is written and compiled.
- **#11 OWL-ViT → NPU** and **#27 open-vocab grounding** — the conversion failed; the moat
  model needs a working export (or the `yolo_world` fallback) before #27/#28 go fully live.
- **#10/#26 Whisper transcription** — QNN `.bin` exists but no TFLite; live transcription
  waits on the native bridge. (Mic capture + noun extraction are real now.)
- **#30/#32 dashboard + QR** — only the in-app HUD exists; no laptop dashboard / QR web view.
- **#33–#37** (YamNet, auto-ducking, Llama narration, gesture, FastSam) — optional/cut-first,
  none started.
- **#38 metrics table**, **#42 backup recording**, **#43 pitch narrative** — Phase 7 proof
  work, pending a usable live run.
- Non-code / yours: **#1** headphone physical test, **#2** submission-form fields, **#4**
  Office Kit rehearsal, **#5** mount + course, **#29** Office-Kit-live habit, **#31**
  airplane-mode demo flow, **#39** obstacle-course rehearsal, **#40** blindfold caveat +
  real-user feedback, **#41** challenge-response drilling.

---

## 6. Recommended next steps, in order
1. **On the phone (you): read the HUD `infer` ms** with a person in frame. That number
   decides the next lever: if NNAPI is fast → good; if slow (MediaTek NNAPI is weak on the
   depth transformer), flip `useNnapi=false` and/or raise `depthEveryN` in
   `TfliteInferenceEngine` — both one-liners.
2. **Eyeball detection placement.** If boxes are offset, the YOLO Layout-A boxes may be
   normalized rather than model-pixel xyxy (adjust `YoloDecoder.decodeBoxesScoresClasses`);
   if depth proximity feels inverted, flip `DepthSampler(nearIsHigh=false)`.
3. **Run the Qualcomm conversion for OWL-ViT** (retry, or `yolo_world` fallback) and re-export
   Whisper for a runnable target — this unblocks the voice moat (#11/#27).
4. **Build the QNN native bridge** (`QnnBackend` JNI + `.bin` in assets), flip
   `EngineConfig.KIND = QNN`. Nothing else changes.
5. **#30 dashboard + QR**, then **Phase 7 proof** (#38 metrics, #39 rehearsal, #42 backup,
   #43 pitch).
6. Patch the two model-name corrections into `secondsense_bible_v4.md` §23.2.

---

## 7. Build & run cheatsheet (this machine)
```powershell
# Build (JBR JDK 21 required; system JDK 24 fails)
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
cd C:\Master_Brain\Projects\secondsense\android
$g = "C:\Users\bhavy\.gradle\wrapper\dists\gradle-8.13-bin\5xuhj0ry160q40clulazy9h7d\gradle-8.13\bin\gradle.bat"
& $g :app:assembleDebug :app:testDebugUnitTest --console=plain --offline

# Install + launch
$adb = "C:\Users\bhavy\AppData\Local\Android\Sdk\platform-tools\adb.exe"
& $adb install -r app\build\outputs\apk\debug\app-debug.apk
& $adb shell am start -n ai.secondsense.app/.ui.MainActivity
# Then on the phone: grant Camera, point at objects, flip Sonification.

# The engine switch:  inference/EngineConfig.kt  ->  KIND = MOCK | TFLITE | QNN
```

---

## 8. New/changed files this session
```
android/app/src/main/java/ai/secondsense/app/
├── inference/
│   ├── EngineConfig.kt            # KIND=TFLITE; QNN branch wired w/ graceful fallback
│   ├── Types.kt                   # +FrameResult.dropOff
│   ├── TfliteInferenceEngine.kt   # dtype-aware outputs, NNAPI delegate, depth-skip, dropoff
│   ├── QnnInferenceEngine.kt      # NEW — NPU engine (reuses decode layer), backend seam
│   ├── qnn/QnnBackend.kt          # NEW — native-bridge interface + StubQnnBackend
│   └── decode/DropOffDetector.kt  # NEW — #17
├── camera/FrameAnalyzer.kt        # frame rotation to upright
├── output/HapticOutput.kt         # CombinedVibration fix + dropOff() pattern
├── sonification/
│   ├── TemporalSmoother.kt        # NEW — #16
│   ├── Calibration.kt             # NEW — #7
│   ├── TierClassifier.kt          # classify() defaults (test compile fix)
│   ├── CueEngine.kt / Spearcon.kt # null-safety / map-lookup fixes
├── voice/                         # NEW package — Phase 4
│   ├── VoicePipeline.kt           #   recognizer/grounder seams + TargetNoun (QNN-backed)
│   ├── VoiceCommandCapture.kt     #   #26 mic capture
│   └── VectorToGoalController.kt  #   #28 steering
└── ui/MainActivity.kt            # calibrate + find buttons, mic perm, keep-screen-on, wiring
app/src/test/.../NewTicketsLogicTest.kt   # NEW — #16/#7/#28/#26 tests
app/build.gradle.kts               # buildToolsVersion=35.0.0
android/local.properties           # NEW — machine-local SDK path
(deleted) tflite_patch/            # redundant staging copy
```
