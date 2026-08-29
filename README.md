# SecondSense


**On-device spatial-audio + haptic navigation for blind and low-vision users.**
Fully offline, runs on a chest-mounted Android phone, zero recurring cost.
Built for the iQOO Hackathon 2026 (Bengaluru City Battle, Aug 29–30).

SecondSense is a **cane *complement*, not a replacement** — it catches what a white cane
structurally cannot: head-height hazards, drop-offs / descending stairs, semantic identity
("what is it", not just "something is there"), and open-vocabulary goal-seeking ("find the
chair"). The defensible IP is the **sonification** — turning proximity, identity, and
direction into something a blind person can act on instantly — not the vision models, which
are a commodity.

---

## Status

| Area | State |
|---|---|
| Vision pipeline (YOLO26s + Depth-Anything-V2) on **TFLite** | live on-device (CPU/XNNPACK for YOLO, NNAPI for depth) |
| Sonification spine — 3 orthogonal channels + haptics + WHITE/BLUE/RED tiers | live, unit-tested |
| Drop-off / negative-obstacle detection (V3: IMU corridor + RGB edge lattice + depth-as-evidence fusion) | live |
| YamNet hazard-sound detection + speech auto-ducking | live (keyword thresholds not yet field-validated) |
| Barometer descent cross-check | works on a device with a barometer (the iQOO 15 has one) |
| Live laptop dashboard + QR (offline, local network) | live |
| Voice goal-seeking — grounding half (spoken noun to COCO detection to steering) | live on TFLite |
| Voice goal-seeking — ASR half (speech to word) | optional: bundle **sherpa-onnx** (`-PenableSherpa`, see [`android/app/src/sherpa/README.md`](android/app/src/sherpa/README.md)); otherwise reports "not loaded" |
| Hexagon **NPU / QNN** path | fully implemented, blocked by an OEM device-access gate (`QNN_DEVICE_ERROR_INVALID_CONFIG`) on retail hardware — kept for a future unblock |
| True open-vocabulary grounding (OWL-ViT / YOLO-World) | roadmap — no working TFLite export; needs the NPU path |

---

## How it works

```
CameraX frame (rotated upright)
  -> InferenceEngine.infer()            <- the swap seam: MOCK | TFLITE | QNN
      |- YOLO26s        -> YoloDecoder (layout-sniffing + NMS)  -> detections
      |- Depth-Anything-V2 (every 2nd frame) -> EMA smooth -> DepthSampler -> per-box proximity
      |- RED-tier synthesis (depth sees something, YOLO named nothing)
      |- Lucas-Kanade ego-motion -> MotionTracker (moving / approaching)
      \- V3 hazard fusion: IMU corridor + EdgeLattice + ground-plane depth evidence
                           + YOLO object-mask suppression -> HazardStateMachine
  -> FrameResult
  -> MainActivity:
      |- hazard state -> distinct haptic (edge-triggered) + barometer cross-check
      |- voice goal active? -> GoalGrounding (match noun to a COCO detection) -> steer
      |- TargetSelector (center-crop, closest, moving-beats-static) -> TierClassifier
      |- Calibration (one-tap baseline) -> TemporalSmoother (~3-frame persistence)
      \- CueEngine  <- continuous loop, own thread
```

**The one rule the `CueEngine` exists to protect — three orthogonal channels, no bleed:**

| Dimension | Channel | Never |
|---|---|---|
| Direction (azimuth) | equal-power stereo **pan** | — |
| Distance / urgency | **pulse repetition rate** (faster = closer) | never pitch |
| Identity | **timbre** — synthesized auditory icon, or sped-up-TTS spearcon fallback | never rate |
| Proximity (parallel) | **graded haptics** — a *primary* channel, not a backstop | — |
| Confidence | grainy **texture** (BLUE/RED) — sounds unsure, never silent | never fakes crispness |

Everything downstream of `InferenceEngine` (the `inference/decode/` layer and the whole
sonification stack) is runtime-agnostic — it speaks in flat `RawTensor`s — so the QNN engine
reuses it unchanged when the NPU is unblocked.

---

## Repo layout

```
android/                         Native Android app (Kotlin) — the thing that runs on the phone
  app/src/main/java/ai/secondsense/app/
    inference/       engine seam + MOCK/TFLITE/QNN engines; decode/ = shared, runtime-agnostic
    sonification/    the crown jewel — CueEngine, TargetSelector, TierClassifier, DegradationLadder,
                     ModeController, AuditoryIcon, Spearcon, TemporalSmoother, Calibration
    output/          AudioOutput (pan), HapticOutput (graded + distinct hazard patterns)
    sensors/         ImuTracker (pitch/roll for the corridor), BarometerMonitor
    audio/           MelSpectrogram (hand-rolled FFT) + HazardSoundDetector (YamNet)
    voice/           Phase 4 — VoiceCommandCapture, GoalGrounding, VectorToGoalController,
                     VoiceRecognizers (picks sherpa KWS if present, else the QNN stub)
    dashboard/       embedded NanoHTTPD dashboard + offline QR
    ui/              MainActivity (text HUD — the real UI is audio/haptics), DebugActivity
  app/src/sherpa/    OPTIONAL offline-ASR module, compiled only with -PenableSherpa (README inside)
  app/src/main/cpp/  QNN native JNI bridge (compiled only with -PenableQnnNative)

convert.py, models.json, scripts/   Laptop -> Qualcomm AI Hub model-conversion pipeline
debug_*.py                          Offline validation harness — run a bundled .tflite against a
                                    real photo in ~2 s (validate every model/algorithm change here first)
docs/                               Phase-0 pre-event tickets + the QAI Hub toolchain runbook
secondsense_bible_v4*.md            The "why": vision, sonification design, ethical invariants, pitch
secondsense_build_map_v3.md         The "what/when": 43 tickets, phases 0-7, per-ticket test steps
secondsense_handoff_v*.md           Session handoffs — live status snapshots
secondsense_research_candidates_v*.md   Adjudicated research: every candidate model/algorithm, graded
```

---

## Build & run

Requires **Android Studio** (Koala 2024.1+). Two host-specific gotchas from `secondsense-android-build`:

- Build with **Android Studio's bundled JBR (JDK 21)** — a system JDK 24 fails.
  `JAVA_HOME="…/Android Studio/jbr"`.
- No committed Gradle wrapper JAR — open in Android Studio (it regenerates it) or run
  `gradle wrapper --gradle-version 8.13` once.

```bash
cd android
# create local.properties (SDK path) if Android Studio hasn't:
echo "sdk.dir=/path/to/Android/Sdk" > local.properties

./gradlew :app:assembleDebug :app:testDebugUnitTest    # green with NO model files (falls back to MOCK)
./gradlew :app:installDebug                             # with a device attached
```

**The engine switch** is one line — [`inference/EngineConfig.kt`](android/app/src/main/java/ai/secondsense/app/inference/EngineConfig.kt):

```kotlin
val KIND: Kind = Kind.TFLITE   // MOCK (no models) | TFLITE (default demo path) | QNN (NPU, blocked)
```

### Model assets (not in the repo — see below)

For `Kind.TFLITE` the app needs these in `android/app/src/main/assets/models/`:

| File | How to get it |
|---|---|
| `yolov11_det.tflite` | `python convert.py --device "<snapdragon>" --runtime tflite --only yolo26_det` (exported under this filename) |
| `depth_anything_v2.tflite` | `python convert.py --device "<snapdragon>" --runtime tflite --only depth_anything_v2` |
| `yamnet.tflite` | `python convert.py --device "<snapdragon>" --runtime tflite --only yamnet` |

`README.txt` and `yamnet_labels.txt` in that folder are kept in the repo. If `TFLITE` is
selected but a model is missing, the app logs it and falls back to `MOCK` — the demo never
hard-crashes.

### Optional native modules

- **QNN / Hexagon NPU** — `-PenableQnnNative=true` + `qnn.sdk.root` in `gradle.properties` +
  the Qualcomm QNN `.so` in `app/src/main/jniLibs/arm64-v8a/`. See
  [`android/README.md`](android/README.md). Currently blocked by an OEM access gate.
- **Offline ASR (sherpa-onnx)** — `-PenableSherpa=true` + wrapper sources + `.so` + KWS model.
  See [`android/app/src/sherpa/README.md`](android/app/src/sherpa/README.md).

---

## Laptop model-conversion pipeline

Turns Qualcomm AI Hub zoo models into NPU-native `qnn_context_binary` and universal `.tflite`.
Compile/profile jobs run on **Qualcomm's cloud** (needs an AI Hub account token + network).

```bash
./setup_qai_hub.ps1        # or ./setup_qai_hub.sh — venv + pinned deps (qai-hub 0.55.0)
python scripts/verify_setup.py                 # works with NO token
qai-hub configure --api_token <YOUR_TOKEN>     # https://aihub.qualcomm.com/
python convert.py --dry-run                    # preview
python convert.py --device "Snapdragon 8 Elite QRD"                    # QNN binaries
python convert.py --device "Snapdragon 8 Elite QRD" --runtime tflite   # universal TFLite
```

## Offline validation workflow

Every model swap / threshold / algorithm change is validated **offline first** — a `debug_*.py`
script loads the exact bundled `.tflite` and runs it against a real photo in ~2 s, before any
Kotlin is written or any phone is touched. The test fixtures (`bottle.jpeg`, `stairs*.jpeg`,
…) are kept in the repo so this reproduces.

---

## Design invariants (do not regress)

1. The laptop is **never** a runtime dependency — the app is self-contained, proven via a live airplane-mode toggle.
2. Depth is **relative proximity `0..1`, never metres** — the type is `proximity: Float`.
3. **Distance = pulse rate, identity = timbre, direction = pan.** No channel drives two dimensions.
4. Haptics are a **primary** graded channel, not a `<0.5 m` panic backstop.
5. Confidence tiering is **derived from signal**; on RED the label is **nulled** — the system never claims an identity it doesn't have.
6. Every degradation path is **total** — never silence-by-omission; PANIC is an independent floor.
7. Validate every model/algorithm change **offline first** (`debug_*.py`).

---

## Documentation

| Doc | What it is |
|---|---|
| [`secondsense_bible_v4.md`](secondsense_bible_v4.md) | The why — vision, sonification design, competitive analysis, pitch |
| [`secondsense_bible_v4_addendum_session4.md`](secondsense_bible_v4_addendum_session4.md) | The reasoning behind the on-device bring-up decisions |
| [`secondsense_build_map_v3.md`](secondsense_build_map_v3.md) | 43 tickets, phases 0-7, per-ticket build/test steps |
| [`secondsense_handoff_v3.md`](secondsense_handoff_v3.md) | Latest session handoff (status snapshot) |
| [`secondsense_research_candidates_v1.md`](secondsense_research_candidates_v1.md) + `v2` | Every candidate model/algorithm, graded verified vs. unconfirmed |
| [`secondsense_phase7_proof_kit.md`](secondsense_phase7_proof_kit.md) | Metrics-table / demo / pitch templates |
| [`android/README.md`](android/README.md) | App-level detail + the QNN drop-in point |
| [`TFLITE_INTEGRATION.md`](TFLITE_INTEGRATION.md) | The TFLite test-path integration notes |

---

## Not in the repo (and why)

- **`android/app/src/main/assets/models/*.tflite` / `*.bin`** — 15–95 MB each; regenerate with `convert.py`.
- **`android/app/src/main/jniLibs/`** — Qualcomm QNN runtime `.so` (proprietary, not redistributed) and, if you add it, the sherpa-onnx `.so`. Obtain from the respective SDK / release.
- **`venv/`, `export_assets/`, `logs/`, `*.pt`, `second.zip`** — generated, downloadable, or large.

If you want turnkey `git clone` -> build, add the model assets via **Git LFS** instead of the
`.gitignore` rules for `assets/models/`.

---

*Prototype / hackathon project. Blindfolded-sighted testing is a proxy, not validation — real
blind/low-vision co-design is the honest next step before any deployment.*
#   s e c o n d s e n s e 
 
 