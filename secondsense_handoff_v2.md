SecondSense — Handoff Context (Session 2, post-Phase 3)
=========================================================

iQOO Hackathon 2026 · Bengaluru City Battle · Aug 29–30. This is the sequel to the
Phase 0→1 handoff doc you uploaded at the start of this chat. It captures everything
decided and built in this session, and what's left to do — for you, a teammate, or the
next chat.

---

## 1. What SecondSense is (unchanged, for reference)

Offline, on-device assistive navigation for blind/low-vision users. Chest-mounted iQOO 15,
four NPU models (YOLOv11, Depth-Anything-V2, Whisper-Tiny, OWL-ViT), spatial audio +
haptics. The defensible IP is the **sonification**, not the vision pipeline — three
orthogonal channels (direction/pan, distance/pulse-rate, identity/timbre) plus haptics as
a *primary* channel and a WHITE/BLUE/RED confidence-tier layer. Moat: "they describe, we
vector" — continuous voice-goal-seeking, proven offline via a live airplane-mode toggle.

Source docs (all previously uploaded, still authoritative):
- `secondsense_bible_v4.md` — the why (425 lines, 25 sections)
- `secondsense_build_map_v2.md` — the what/when (42 tickets, Phases 0–7)
- `secondsense_phase0_laptop.zip` — verified Qualcomm AI Hub conversion pipeline

Two name corrections from Phase 0 are still **unpatched** in the bible/build-map source
files (they're baked into the code, just not into the docs): the Whisper zoo module is
`whisper_tiny` not `whisper_tiny_en`, and segmentation is `fastsam_x` not `fastsam_s`.
Patch bible §23.2 and build-map tickets #10/#37 when convenient — cosmetic, not blocking.

---

## 2. The one architecture decision made this session

**Phone app framework: native Android (Kotlin), not Python-on-Android or Flutter/RN.**

Reasoning: the compiled models are QNN context binaries (native C++ runtime regardless of
app framework); the product's real IP is latency-sensitive haptic/audio control, which a
bridge layer (Chaquopy, platform channels) actively hurts; and it's a single-device demo,
so cross-platform tax buys nothing. This was the blocking decision from the Phase 0→1
handoff — it's now resolved and the whole codebase reflects it.

---

## 3. What's been built (tickets #6, #18–#25)

All code lives under `secondsense/android/` — a complete, self-contained Android Studio
project. It builds and runs **today**, on an emulator or device, with **no phone-specific
hardware, no Qualcomm token, and no converted model** — everything is validated against a
`MockInferenceEngine` that scripts a realistic scene (drifting target, oscillating
proximity, cycling identity classes, WHITE→BLUE→RED tier cycling, moving/static toggling).

### Ticket #6 — On-device app skeleton + camera capture ✅
- CameraX live frame stream (rear camera, RGBA_8888, 640×480 analysis res).
- `InferenceEngine` interface — **the swap seam**. Ships with `MockInferenceEngine`; a
  future `QnnInferenceEngine` drops in behind the same interface with a one-line change
  in `MainActivity`, and nothing else in the app changes.
- `AudioOutput` (stereo `AudioTrack`) and `HapticOutput` (`VibratorManager`, graded
  amplitude) — both real, both on-device, both testable via the app's Test Tone / Test
  Buzz buttons (the ticket's literal done-condition).
- Debug HUD showing frame count, inference time, detection count, and (as of later
  tickets) the resolved cue + a colored confidence-tier badge.

### Phase 3 — Sonification & haptics, tickets #18–#22 (the crown jewel) ✅
- **#18 Direction** — equal-power stereo pan (`AudioOutput.playMono(pan=…)`). Verified:
  hard-left/center/hard-right correct, L²+R²=1 (constant loudness) at every pan position.
- **#19 Distance** — `CueEngine.intervalMsFor()` maps proximity to inter-pulse interval,
  900 ms (far) → 120 ms (near). Pitch is never touched. Verified monotonic.
- **#20 Identity** — `AuditoryIcon` synthesizes short caricature timbres fully offline, no
  asset files (dog-yip, footstep-thud, wooden knock, door-clack, vehicle rumble, plus a
  neutral "unknown" tick for RED tier). `Spearcon` + `AudioHelpers` (WavIo, TimeStretch)
  give a sped-up-TTS fallback for classes without a bespoke icon — bakes lazily off the
  audio thread via on-device `TextToSpeech`, falls back to the honest tick if not ready.
- **#21 Haptics** — `HapticOutput.proximityPulse()` is graded amplitude, a genuine primary
  channel from day one (not a `<0.5m` backstop).
- **#22 Integration** — `CueEngine` runs a continuous loop on its own thread; vision side
  just posts the latest `CueTarget`. All channels render together, capped at one primary
  cue, orthogonality enforced by the code's shape (distance has no pitch knob; identity
  has no rate knob; direction only ever touches pan).
- Also built: `TargetSelector` (#14 center-crop + #15 closest-in-center + static/dynamic
  priority split) — the "#15 output" #22 needed to integrate against.

### #23 — Uncertainty / self-trust layer ✅
- `TierClassifier` **derives** WHITE/BLUE/RED from real signal (score, depth availability,
  whether a class label came back) rather than trusting an upstream guess — so it behaves
  identically once the mock is replaced by the real engine.
- Hysteresis (higher bar to promote than to demote) + 2-frame stability gate, so the audio
  texture doesn't stutter at threshold boundaries.
- On RED, `TargetSelector.selectWithTier()` **nulls the label** — the system won't claim
  an identity it doesn't have. Tier shown per-cue on the HUD as a colored badge.

### #24 — Graceful-degradation ladder ✅
- `DegradationLadder.decide()` maps `(tier, proximity, depth, label)` to one of three
  audio rungs (FULL / PROXIMITY / SILENT_AUDIO), **totally** — every input combination
  lands somewhere (verified: all 72 combinations tested).
- **PANIC is an independent floor**, not the last rung in a chain: a close object fires
  an unmistakable double-buzz haptic *in addition to* whatever audio rung applies — so a
  depth-only, unnamed obstacle still gets PROXIMITY audio *and* a panic haptic. The
  system can always steer you off a wall even when it can't name it.

### #25 — Two operating modes (flow vs scan/seek) ✅
- `ModeController` is the single source of truth for the §4 mode split: **FLOW**
  (walking) = center-crop on, one cue, voice off; **SCAN_SEEK** (stopped) = whole frame,
  more cues, voice goal-seeking **on**. Drives the camera analyzer via a listener.
- This is the seam Phase 4 (voice goal-seeking) plugs into: `acceptsVoiceCommands` is
  already exposed and already gated correctly.

### Verification approach (repeated every ticket — worth continuing)
Because this sandbox has no Android SDK, no Kotlin compiler, and no network to Google's
Maven repo, a real `./gradlew build` can't run here — that's yours in Android Studio.
What *was* done every ticket:
1. Cross-file static validation — every resource reference, viewBinding id, and Kotlin
   symbol import checked programmatically (no typos, no dangling references, no dupes).
2. XML well-formedness check on every layout/manifest file.
3. **Numerical proof of the core logic** — each channel's math (pan law, rate mapping,
   target selection, tier hysteresis, ladder totality, mode behavior) mirrored in a
   standalone Python harness and proven against edge cases *before* being trusted.
4. Those same proofs committed as JUnit tests (`app/src/test/…`), runnable via
   `./gradlew :app:testDebugUnitTest` — four suites: `TargetSelectorTest`,
   `TierClassifierTest`, `DegradationLadderTest`, `ModeControllerTest`.

### Current file inventory
```
secondsense/android/
├── README.md                          # build instructions, QNN swap point, full ticket map
├── settings.gradle.kts / build.gradle.kts / gradle.properties
├── gradle/wrapper/gradle-wrapper.properties   # Gradle 8.7 pinned
└── app/
    ├── build.gradle.kts               # CameraX, coroutines, JUnit; NDK/CMake block ready but off
    ├── src/main/AndroidManifest.xml
    ├── src/main/java/ai/secondsense/app/
    │   ├── inference/   Types.kt, InferenceEngine.kt, MockInferenceEngine.kt
    │   ├── camera/      FrameAnalyzer.kt
    │   ├── output/      AudioOutput.kt, HapticOutput.kt
    │   ├── sonification/  CueTarget.kt, TargetSelector.kt, TierClassifier.kt,
    │   │                  DegradationLadder.kt, ModeController.kt, AuditoryIcon.kt,
    │   │                  Spearcon.kt, AudioHelpers.kt, CueEngine.kt
    │   └── ui/          MainActivity.kt
    ├── src/main/res/    layout, values, drawable, mipmap (launcher icon)
    └── src/test/java/…/sonification/   4 JUnit test files
```
16 main Kotlin files, 4 test files. Latest packaged deliverables in this chat:
`secondsense_android_phase3_25.zip` (android/ only) and `secondsense_repo_full.zip`
(the whole `secondsense/` tree — Phase 0 Python pipeline + android/ together).

---

## 4. What's explicitly NOT done yet

- **No real Gradle build has been run.** Open in Android Studio, let it sync, confirm it
  actually compiles — this is the single highest-priority action, because everything
  above is validated statically/numerically but never compiled.
- **No model conversion has run.** Tickets #8–#11 (YOLOv11, Depth-Anything-V2,
  Whisper-Tiny, OWL-ViT → NPU) need your Qualcomm account token on your own laptop —
  `qai-hub configure` → `list_devices.py` → `convert.py`. Nothing in this sandbox can
  reach `aihub.qualcomm.com`.
- **No `QnnInferenceEngine` exists.** The seam (`InferenceEngine`) is ready; the real
  implementation needs the converted binaries first, then a `src/main/cpp/` JNI layer
  (the `externalNativeBuild` block in `app/build.gradle.kts` is stubbed but commented
  out for exactly this reason).
- **Phase 2 vision tickets #12/#13/#16/#17** (live YOLO loop, live depth loop, temporal
  smoothing, drop-off detection) are still open — they were skipped because #18–#25 were
  higher-leverage against the mock. They become real once QnnInferenceEngine exists.
- **Phase 4 voice goal-seeking (#26–#28)** — the demo climax — not started.
- **Phase 5 systems/dashboard (#29–#31)**, including the QR-code multi-judge dashboard
  and the airplane-mode proof flow — not started.
- **Phase 0 pre-event items** that are inherently yours (physical, not code): headphone
  decision + physical test (#1), submission logistics sign-off (#2), Office Kit rehearsal
  (#4), mount + demo course build (#5).

---

## 5. Recommended next steps, in order

1. **Open `android/` in Android Studio and build it.** This is the one thing that's been
   deferred every ticket. Confirm `./gradlew :app:assembleDebug` succeeds and
   `./gradlew :app:testDebugUnitTest` passes (4 suites). Fix anything Android Studio
   flags — dependency version bumps are the most likely surprise.
2. **Run the Qualcomm conversion (tickets #8–#11)**, on your own laptop, in a Green Light
   window equivalent: `qai-hub configure --api_token <TOKEN>` → `list_devices.py` →
   `convert.py --device "..." --group core`. This unblocks everything native.
3. **Build `QnnInferenceEngine`** once binaries exist — load the `.bin` via a thin JNI
   layer, return the same `FrameResult` shape the mock does, swap one line in
   `MainActivity`. Nothing else changes.
4. **Phase 2 vision loop (#12/#13/#16/#17)** against the real engine — this is where the
   mock's fabricated data gets replaced by the real YOLO/Depth output flowing into the
   sonification spine you already built and validated.
5. **Phase 4 — voice goal-seeking (#26–#28)**, the demo climax. `ModeController` already
   gates `acceptsVoiceCommands` to SCAN_SEEK; Whisper-Tiny + OWL-ViT slot in behind a
   similar swap-seam pattern to `InferenceEngine`.
6. **Phase 5 — Office Kit dashboard + QR judge access (#29–#31)**, then **Phase 7 proof
   & demo prep (#37–#42)**: fill the real metrics table, rehearse the obstacle course,
   record the backup video, rehearse challenge-response answers.
7. **Patch the two model-name corrections** into `secondsense_bible_v4.md` §23.2 and
   `secondsense_build_map_v2.md` #10/#37 whenever convenient (cosmetic).

If you want to keep working in this chat, the natural continuation is either **step 3/4
(QnnInferenceEngine + real vision loop)** once you've run the conversion, or **step 5
(Phase 4 voice goal-seeking)**, which can still be stubbed against the mock the same way
Phase 3 was — I can build the Whisper/OWL-ViT seam and the vector-to-goal cueing logic
now, ready to swap in real models later, exactly like the pattern that's worked every
ticket so far.
