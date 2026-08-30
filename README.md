<div align="center">

# SecondSense

**A second pair of eyes for blind and low-vision travellers — running entirely on a
chest-mounted Android phone, fully offline, at zero recurring cost.**

![platform](https://img.shields.io/badge/platform-Android%2011%2B-3ddc84)
![network](https://img.shields.io/badge/network-100%25%20offline-blue)
![ai](https://img.shields.io/badge/AI-on--device%20(NPU%20%7C%20CPU)-orange)
![tests](https://img.shields.io/badge/unit%20tests-~118%20passing-brightgreen)
![event](https://img.shields.io/badge/for-iQOO%20Hackathon%202026-black)

*It watches the world, decides what matters, and tells you through spatial sound and graded
vibration — never the screen. And you can just talk to it.*

</div>

---

## Why it exists

A white cane is perfect at one thing: what's touching the ground in the arc in front of your
feet. It tells you nothing about a person crossing your path, a **step going down** before your
foot reaches the edge, a branch at head height, which door is the exit, what a sign says, or
where you set your phone down five minutes ago.

SecondSense is a **cane complement, not a replacement.** It fills exactly those gaps — and it
is built around one hard rule: **the AI is never the last word on safety.** Ask it "is it safe
to cross?" and it will *never* say "yes" — that question is physically fenced off from every
language model in the system (see [The Safety Gate](#the-safety-gate)).

The defensible idea here is not the vision models (those are commodity) — it's the
**sonification**: turning direction, distance, and identity into something a blind person can
act on in under a second, and the layered, honest way hazards are confirmed before anything
makes a sound.

---

## What it does

| Capability | In one line |
|---|---|
| **Obstacle awareness** | Names what's ahead ("person, chair, wall"), how far, which direction, and whether it's *approaching* — as continuous spatial sound. |
| **Drop-off detection** | Detects a step *down* / descending stairs before your foot reaches the edge, using **five independent checks** that must agree. |
| **The Specular Trap** | Puddles, hard shadows, and shiny marble look exactly like a cliff — three physics vetoes (colour, motion, air-pressure) cancel the false alarm. |
| **Overhead hazards** | Branches, open cabinet doors, poles at head height — the white cane's blind spot. |
| **Talk to it** | "What's ahead?", "find a chair", "read that sign", "where did I leave my phone?", "call mum", "I'm on the bus". |
| **Knows what you're doing** | Six activity modes (walking / standing / home / sitting / transit / conversation) — auto-detected, each reconfiguring the whole app. On a bus, hazard detection turns *off* (a moving vehicle creates phantom drop-offs). |
| **On-device reasoning** | A 1-billion-parameter language model (Gemma) runs on the phone for anything the command grammar can't handle — grounded in what the camera sees. |
| **Reads signs, Hindi & English** | Offline OCR in Latin + Devanagari, with on-device translation either way. |
| **Stays fast when it overheats** | A phone strapped to a warm body throttles in ~15 min — a thermal governor sheds non-critical work first and keeps the safety loop alive. |
| **Emergency** | Hold both volume keys 2 s → panic tone + spoken battery/status → "say who to call". |
| **Laptop 3-D demo** | For spectators: the phone streams frames, a laptop builds a live coloured point-cloud of the room. |

---

## How it works

```
   Camera frame  (~5–15 fps, throttled by heat + context)
        │
        ▼
   PERCEPTION ── object detector (YOLO) → boxes + labels
              └─ depth model            → per-pixel near/far
        │
        ▼
   FUSION      per-object: direction · proximity · rate-of-approach · moving?
               + 3-frame stabilisation + WHITE/BLUE/RED confidence tier
        │
        ▼
   HAZARDS     drop-off state machine (5 evidences, 2-of-3 → possible, 3-of-5 → confirmed)
               ├─ Specular-Trap vetoes  (shadow colour · coplanar motion · barometer)
               ├─ overhead / head-height
               └─ camera-health (blocked / dim / knocked off angle)
        │
        ▼
   TARGET      of everything in view, the ONE thing to cue now
               (hazard > voice goal > memory bearing > nearest obstacle), habituation-gated
        │
        ▼
   OUTPUT      Sonification: pan = direction · pulse-rate = distance · timbre = identity
               Haptics:      a distinct graded buzz per hazard class  (never mutable for drops)
               Earcons:      a short tune on every context switch
               Speech:       answers · sign read-outs · the honest "I can't tell"

   Alongside, continuously:
     motion sensors → footsteps + vehicle-vibration → activity context (6 modes)
     barometer      → am I descending?  (independent drop-off check)
     microphone     → siren/horn detection · voice commands · conversation ducking
     thermal governor → shed work as the SoC heats, keep a safety floor
     voice assistant  → wake (one tap) → intent grammar → Gemma for the long tail
     safety gate      → intercept "is it safe?" before ANY model sees it
```

### The one rule the cue engine protects — three channels, no bleed

| Dimension | Channel | Never |
|---|---|---|
| Direction | equal-power stereo **pan** | — |
| Distance / urgency | **pulse repetition rate** (faster = closer) | never pitch |
| Identity | **timbre** — a synthesised auditory icon, or a sped-up spoken word | never rate |
| Proximity (parallel) | **graded haptics** — a *primary* channel | — |
| Confidence | grainy **texture** (BLUE/RED) — sounds unsure, never silent | never fakes crispness |

### The Safety Gate

> A blind person asking a gadget "can I walk?" is the highest-stakes moment in the product.
> A small language model, trained to be agreeable, will say **"yes, go ahead"** over a live
> drop-off warning. So it is never allowed to answer.

Six layers, in order:

1. A **learned n-gram classifier** (37 KB, no download, works in English / Hindi / Kannada
   because it reads character patterns not words) catches the question — tuned for recall.
2. A short obvious-phrasing fast path.
3. The answer is a **fixed, legally-careful template** filled from the *real* sensor state —
   *"I cannot decide whether it is safe to move. The sensors show \[state], but you must rely
   on your white cane, traffic sounds, and your own judgement."* Never the word "safe" as a
   yes.
4. A **1.5-second hazard memory** so a drop-off that flickered during the question still counts.
5. If anything reaches the LLM anyway, its reply is scanned — a "you can go" green-light is
   thrown away and replaced with the deflection.
6. The LLM's prompt also instructs it to refuse.

*(On a real device, Gemma-1B answered "you see nothing there" to "is anything in my way?"
while a book, a person and furniture were all detected. This is why the gate exists.)*

**Full detail on every layer, with the maths and worked examples:**
📖 [`SECONDSENSE_BIBLE.md`](SECONDSENSE_BIBLE.md)

---

## Status

| Area | State |
|---|---|
| Vision pipeline (YOLO + Depth-Anything-V2) on the **Hexagon NPU (QNN)** | **working** on the iQOO 15 (`-PenableQnnNative=true`) |
| Same pipeline on plain CPU (**TFLite**) / **MOCK** for testing | working — the app never hard-crashes; missing models fall back to MOCK |
| Sonification spine — 3 orthogonal channels + graded haptics + tiers | live, unit-tested |
| Drop-off detection (edge lattice + depth + ground-plane + object-suppression + barometer) | live, unit-tested; **not yet field-tested on real drops** |
| Specular-Trap vetoes (shadow chromaticity · ground-flow coplanarity · barometer re-escalation) | live, unit-tested; **needs real puddle/shadow/marble footage** |
| Camera tamper / occlusion / knocked-off-angle detection | live |
| Activity-context system — 6 modes, thermal-merge, earcons | live, unit-tested |
| Context auto-detection (footsteps + vehicle-vibration signature) | live; **threshold needs a real walk + bus ride** |
| Thermal governor (4 signals, warm-up guard, relative baseline) | live; downshift verified; **full soak proof pending** |
| Voice — one-tap wake + ~15-intent grammar (find / read / status / call / timer / mode / …) | live, unit-tested (78-phrase judge sweep) |
| Voice — on-device LLM fallback (Gemma-3-1B-IT, MediaPipe) | works with `-PenableLlm=true` + a side-loaded `.task`; weak — a long-tail helper only |
| **Safety gate** (Tier 1 templates + Tier 2 learned classifier) | live; classifier scores **100 %** on held-out phrases in 3 languages |
| Panic gesture (both volume keys 2 s) | live |
| Sign reading — Latin + Devanagari OCR + on-device Hindi↔English translation | live; translation needs its one-time Wi-Fi model download |
| Blind-first UI — full-screen gesture surface, spoken menu, volume-key mapping, TalkBack fallback | live; **needs a hands-on session with a blind user** |
| Hazard-sound detection (car horns, sirens) + speech auto-ducking | live; thresholds not field-validated |
| Laptop live 3-D room reconstruction demo (`laptop/room3d/`) | works end to end against the live phone; soft on a CPU-only laptop |
| Persistent AR room map (walk-once, route later) | steps 1–4 built; routing not built — **parked** |
| Open-vocabulary grounding + on-NPU Whisper | roadmap — currently uses Android's speech recogniser + the 80 COCO classes (with synonyms) |

---

## Future scope — a 3-D model of your home and workspace

Right now the 3-D reconstruction (`laptop/room3d/`) is a **spectator demo**: the phone streams
frames, a laptop assembles a coloured point-cloud of the room. The direction this points is a
**persistent, on-device 3-D model of the places a user lives and works** — built once by
walking through, then kept and reused.

**How it would be built.** A one-time guided sweep — the user (or a sighted helper) walks each
room slowly while the phone runs Depth-Anything (metric-indoor variant) + ARCore 6-DoF pose.
Frames are fused into a metric mesh + a coarse occupancy grid, objects are detected and pinned
to world coordinates, and rooms/zones are labelled. The whole model lives in `filesDir` — no
cloud, re-anchored on later visits by standing at a known spot.

**What it unlocks:**

| Capability | What changes for the user |
|---|---|
| **Persistent object memory** | "Where are my keys?" answered from the model — *"usually on the kitchen counter, on your right as you enter"* — not just the ~60-second dead-reckoning window it has today. |
| **Indoor routing** | "Take me to the front door" → a path **around** known furniture, delivered through the existing directional cue. This is the routing currently parked in `ar/`. |
| **Zone awareness** | The app knows you're *in the kitchen* vs *the hallway* and can auto-pick the right activity context and vocabulary. |
| **Change detection** | A chair moved into your usual path, a door now closed, a box left on the floor — flagged as *new* against the baseline, which is exactly the kind of trip hazard a cane finds late. |
| **Fewer false drop-offs** | A known floor plan lets the hazard state machine tell "there has always been a step here" from "this edge is new" — tightening the Specular-Trap logic. |
| **Workspace maps** | The same model for an office floor, a workshop, a campus building — shareable with consent, so a venue can publish its own accessibility map. |

**The honest gap.** The reconstruction pipeline works end-to-end as a demo; **persistence,
semantic labelling, on-phone (not laptop) execution, and routing are not built.** The ARCore
walk-once map has steps 1–4 (scan → detect → occupancy grid → save/anchor); step 5 — consuming
it for live navigation — is the parked work this section describes finishing.

---

## Quick start

Requires **Android Studio** (Koala 2024.1+) or a cached Gradle 8.13.

```bash
cd android

# Build with Android Studio's bundled JBR (a system JDK 24 will fail):
export JAVA_HOME="/path/to/Android Studio/jbr"        # or set it in Android Studio

# First build needs network (ML Kit + translate + ARCore deps). After that, --offline is fine.
gradle :app:assembleDebug :app:testDebugUnitTest      # green with NO model files → falls back to MOCK
gradle :app:installDebug                              # with a device attached
```

**Optional build flags:**

| Flag | Adds |
|---|---|
| `-PenableQnnNative=true` | the Hexagon NPU path (needs the Qualcomm QAIRT SDK + the `.so` in `jniLibs/`) |
| `-PenableLlm=true` | the on-device Gemma path + the MediaPipe runtime (~120 MB). The ~530 MB model is **side-loaded**, never bundled — see [`android/app/src/llm/README.md`](android/app/src/llm/README.md) |
| `-PenableSherpa=true` | offline keyword-spotting ASR — see [`android/app/src/sherpa/README.md`](android/app/src/sherpa/README.md) |

**The engine switch** is one line — [`inference/EngineConfig.kt`](android/app/src/main/java/ai/secondsense/app/inference/EngineConfig.kt):

```kotlin
val KIND: Kind = Kind.QNN   // MOCK (no models) · TFLITE (universal) · QNN (Hexagon NPU)
```

### Laptop 3-D room demo

```bash
cd laptop
pip install -r room3d/requirements.txt          # fresh venv; scrcpy optional
python -m room3d.record --source phone:<IP> --out sweep.mp4 --seconds 120   # slow room sweep
python -m room3d.app    --source file:sweep.mp4                             # reconstruct + view
```

The phone serves frames at `http://<phone-ip>:8085/frame.jpg` (same Wi-Fi, no internet). See
[`laptop/room3d/README.md`](laptop/room3d/README.md).

---

## Repo layout

```
android/                          the app that runs on the phone (Kotlin)
  app/src/main/java/ai/secondsense/app/
    inference/        engine seam (MOCK | TFLITE | QNN); decode/ = shared, runtime-agnostic fusion
    inference/decode/ hazard state machine, Specular-Trap vetoes, ground-plane, optical flow
    sonification/     CueEngine, TargetSelector, TierClassifier, Earcons, AuditoryIcon, Spearcon
    output/           AudioOutput (pan), HapticOutput (graded + per-hazard patterns)
    context/          AppContext (6 modes), ContextManager, ContextAutoDetector
    voice/            IntentInterpreter, LlmAssistant + LlmPrompt, SafetyGate, NgramSafetyGate,
                      SafetyGateWeights (generated), GoalGrounding, PhoneActions
    perf/             ThermalGovernor, PerfPolicy
    perception/       MlKitPerception (OCR + faces), OcrTranslator, LanguagePrefs
    sensors/          ImuTracker, BarometerMonitor, PedometerTracker
    ar/               experimental persistent room map (parked)
    dashboard/        embedded NanoHTTPD spectator dashboard + /frame.jpg
    ui/               MainActivity (the real UI is audio + haptics + gestures)
  app/src/llm/        flag-gated MediaPipe LLM assistant (-PenableLlm)
  app/src/main/cpp/   QNN native JNI bridge (-PenableQnnNative)

laptop/
  room3d/             live 3-D room reconstruction demo (Python, Open3D)
  tools/              train_safety_gate.py — trains the Tier-2 classifier, bakes weights into Kotlin

SECONDSENSE_BIBLE.md  the complete technical bible — every layer, the maths, worked examples,
                      edge-case tables, how to test each piece (no code)
debug_*.py            offline validation harness — run a bundled model against a real photo
```

---

## Design invariants (do not regress)

1. The laptop / any network is **never** a runtime dependency — proven via an airplane-mode toggle.
2. Depth is **relative proximity `0..1`**, never metres, in the live cue path.
3. **Direction = pan, distance = pulse rate, identity = timbre.** No channel drives two dimensions.
4. Haptics are a **primary** graded channel, not a "< 0.5 m" panic backstop.
5. Confidence is **derived from signal**; on RED the identity label is nulled — it never claims
   an identity it doesn't have.
6. Every degradation path is **total** — never silence-by-omission. The imminent-collision
   haptic is an independent floor that pause / mute / context / heat cannot disable.
7. **The language model never answers "is it safe."** Six layers enforce it.
8. Many small independent checks that vote — not one monolithic model.

---

## Not in the repo (and why)

- **`android/app/src/main/assets/models/*.tflite` / `*.bin`** — 15–95 MB each; regenerate with
  `convert.py` (Qualcomm AI Hub pipeline).
- **`android/app/src/main/jniLibs/`** — proprietary Qualcomm QNN runtime `.so` (not
  redistributed).
- **The Gemma `.task` model** (~530 MB) — side-loaded to the device, never committed.
- **`*.task`, `*.onnx`, `*.ply`, `venv/`, build output** — generated, downloadable, or large.

---

<div align="center">

*Prototype / hackathon project. Blindfolded-sighted testing is a proxy, not validation — real
blind and low-vision co-design is the honest next step before any deployment.*

</div>
