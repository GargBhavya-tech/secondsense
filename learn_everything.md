# SecondSense — Learn Everything From Basics

> A complete self-contained guide. Read top to bottom. No prior AI/Android knowledge assumed.

---

## CHAPTER 1: What Is This App, In One Paragraph

SecondSense is an Android app that runs entirely on your phone — no internet, no server.
It uses the phone's **camera** to see the world, runs **AI models** on the phone's chip to
understand what it sees (objects, distances, drop-offs), and then tells a blind/visually
impaired person about it through **spatial audio** (stereo sound from left/right ear) and
**haptic vibrations** (phone buzz patterns). The person never has to look at the screen.

---

## CHAPTER 2: The Phone's Hardware (Why iQOO Matters)

A modern smartphone has multiple chips inside it:

```
┌─────────────────────────────────────────┐
│           Snapdragon 8 Elite SoC        │
│  (SoC = System on Chip, everything      │
│   in one package)                       │
│                                         │
│  ┌───────┐  ┌───────┐  ┌─────────────┐ │
│  │  CPU  │  │  GPU  │  │  NPU / HTP  │ │
│  │(brain)│  │(graph-│  │(AI engine)  │ │
│  │       │  │ ics)  │  │             │ │
│  └───────┘  └───────┘  └─────────────┘ │
└─────────────────────────────────────────┘
```

- **CPU** — runs normal code (Kotlin/Java app code, logic)
- **GPU** — renders graphics, also used for some AI
- **NPU / HTP** (Neural Processing Unit / Hexagon Tensor Processor) — a specialized chip
  designed ONLY for running AI models. It's dramatically faster and more power-efficient
  than CPU/GPU for this specific task.

**Why iQOO?** The iQOO 15 has a Snapdragon 8 Elite, which has one of the most powerful
NPUs available. Running YOLO on CPU takes ~200ms per frame. On this NPU it takes ~20ms.
That's the difference between "laggy and useless" and "real-time."

---

## CHAPTER 3: What Is an AI Model?

Think of an AI model as a very complex mathematical function:

```
INPUT IMAGE  →  [millions of math operations]  →  OUTPUT (boxes, labels, scores)
```

You don't write these math operations by hand. Instead, you **train** the model:
show it millions of images with labels ("this image has a person at x=100, y=200"),
and it adjusts its internal numbers (called **weights** or **parameters**) until it
gets good at predicting labels for new images it's never seen.

After training, the model is frozen. You package those weights into a file (`.pt`, `.onnx`,
`.tflite`, `.bin`) and ship it. The app loads that file and runs the math — this is called
**inference**.

**SecondSense uses 3 main AI models:**

| Model | What it does | Trained on |
|---|---|---|
| YOLOv11 | Detects objects in images + draws boxes | COCO dataset (80 object types) |
| Depth-Anything-V2 | Estimates how far away things are | Millions of depth-labelled images |
| YamNet | Classifies sounds (horn, siren, speech) | AudioSet (521 sound types) |

---

## CHAPTER 4: YOLOv11 — The Object Detector

### What YOLO Does

YOLO (You Only Look Once) takes a camera frame and outputs:

```
Input:  640×640 pixel image (RGB)

Output: For each detected object →
        - Bounding box: [x1, y1, x2, y2]  (where is it in the image?)
        - Class index: 0 = person, 1 = bicycle, 2 = car ... (what is it?)
        - Confidence score: 0.0 → 1.0  (how sure is it?)
```

Example output for a frame with a person and a chair:
```
Detection 1: box=[120,80,250,450]  class=0(person)  score=0.91
Detection 2: box=[300,200,500,480] class=56(chair)   score=0.76
```

### How Your App Uses It

1. Camera gives a frame → `FrameAnalyzer.kt` receives it
2. `Preprocess.letterbox()` resizes it to 640×640 (keeping aspect ratio, padding with grey)
3. The resized image is converted to a `ByteBuffer` of float values (each pixel RGB → 3 floats, 0.0 to 1.0)
4. This buffer is fed into the YOLO model
5. `YoloDecoder.decode()` reads the raw output tensor, runs NMS (see below), and returns a list of `Detection` objects

### NMS — Non-Max Suppression

YOLO outputs hundreds of candidate boxes. Most are duplicates. NMS keeps only the best one per object:

```
Before NMS:                After NMS:
┌──────────────┐           ┌──────────────┐
│ ┌──────────┐ │           │              │
│ │ ┌──────┐ │ │           │   ┌──────┐   │
│ │ │person│ │ │   ──────► │   │person│   │
│ │ └──────┘ │ │           │   └──────┘   │
│ └──────────┘ │           │              │
└──────────────┘           └──────────────┘
  3 overlapping boxes         1 clean box
```

Rule: if two boxes overlap by more than 45% (IoU threshold) and detect the same class,
keep only the one with the higher confidence score, discard the rest.

### CocoLabels → SecondSense Icon Vocabulary

YOLO gives you 80 COCO class names. SecondSense maps these to a smaller set for audio:

```kotlin
"car", "truck", "bus", "motorcycle", "bicycle", "train"  →  "vehicle"  →  low rumble sound
"chair", "couch", "bench", "bed", "toilet"               →  "chair"    →  wooden knock sound
"person"                                                  →  "person"   →  footstep thud sound
"dog"                                                     →  "dog"      →  rising yip sound
```

---

## CHAPTER 5: Depth-Anything-V2 — The Distance Estimator

### The Problem With Distance From a Single Camera

A single camera image is 2D — it has no depth information built in. A big object far away
and a small object close up can look identical in the image. This is called the
**monocular depth problem**.

### What Depth-Anything-V2 Does

It was trained on millions of images where depth was measured. It learned patterns:
- Things that look blurry are usually far
- Things with smaller apparent size are usually far
- Shadows, perspective lines, texture gradients all carry depth cues

It outputs a **depth map** — an image where brighter = closer (or further, depending on convention):

```
Camera frame:          Depth map output:
┌─────────────┐        ┌─────────────┐
│  [person]   │        │  [bright]   │   ← person is close
│             │   →    │             │
│  [wall]     │        │  [dark]     │   ← wall is far
└─────────────┘        └─────────────┘
```

**Critical limitation:** The values are RELATIVE, not metric. It can't tell you
"that person is 2.3 metres away." It can tell you "that person is closer than that wall."
Your app is honest about this — it never claims metric distance.

### How Your App Uses Depth

```
Depth map → for each detected YOLO box →
    sample depth at the center of the box →
    normalize to 0.0 (far) .. 1.0 (touching) →
    this is "proximity"
```

This proximity drives:
- **Haptic intensity** (stronger buzz = closer)
- **Audio pulse rate** (faster beeps = closer)
- **Panic threshold** (proximity ≥ 0.80 → always buzz regardless of anything else)

---

## CHAPTER 6: YamNet — The Sound Classifier

### What It Does

YamNet listens to the microphone and classifies what it hears into 521 AudioSet categories.
Your app uses it for two things:

1. **Hazard detection:** car horn, siren, alarm, explosion → fire a distinct 4-tap haptic pattern
2. **Speech detection:** human speech → reduce (duck) the navigation audio by up to 50% so you
   can hear the conversation

### The Audio Pipeline

```
Microphone (16kHz mono)
    │
    ▼
1-second audio chunk (16,000 samples)
    │
    ▼
MelSpectrogram.kt
    ├─ Apply Hann window (smooths the edges of each short segment)
    ├─ FFT (Fast Fourier Transform) — converts time domain → frequency domain
    │   "how much of each frequency is in this 25ms window?"
    ├─ Apply Mel filterbank — 64 filters that mimic how human ears work
    │   (we hear pitch differences better in low frequencies)
    └─ Log compression → [96 × 64] grid of numbers = one "patch"
    │
    ▼
YamNet TFLite model
    │
    ▼
521 raw logit scores → sigmoid → probabilities
    │
    ├─ Check against HAZARD_KEYWORDS list → fire HapticOutput.hazardSound()
    └─ Check against SPEECH_KEYWORDS list → CueEngine.setDucked(true)
```

---

## CHAPTER 7: What Is TFLite?

TFLite (TensorFlow Lite) is Google's framework for running AI models on mobile devices.

### The Problem It Solves

Normal AI models (PyTorch, TensorFlow) are huge and slow. Training a YOLO model requires
a powerful GPU server. But INFERENCE (just using the model, not training) can be made much
faster and smaller for mobile.

TFLite takes a trained model and:
1. Converts it to a `.tflite` file format (optimized for mobile)
2. Provides a Java/Kotlin API to run it on Android

### How TFLite Works in Your App

```kotlin
// Load the .tflite file from assets
val interpreter = Interpreter(loadModel("models/yolov11_det.tflite"))

// Create input buffer (640×640×3 floats)
val inputBuffer = ByteBuffer.allocateDirect(640 * 640 * 3 * 4)

// Fill it with your preprocessed image pixels
// ...

// Run inference
interpreter.run(inputBuffer, outputBuffer)

// outputBuffer now contains raw scores/boxes
```

### NNAPI — The Hardware Acceleration Layer

TFLite by itself runs on CPU. To use the phone's NPU, you add an **NNAPI delegate**:

```kotlin
val options = Interpreter.Options()
options.addDelegate(NnApiDelegate())
val interpreter = Interpreter(model, options)
```

NNAPI is Android's layer that routes model operations to the best available hardware
(NPU, GPU, or CPU). On your test phone, there was a bug where NNAPI returned wrong scores
for YOLO (all saturated to 1.0), so you force CPU for YOLO but keep NNAPI for depth.

---

## CHAPTER 8: What Is Quantization?

### The Basic Problem

A trained model stores its weights as 32-bit floats (FP32). Each weight = 4 bytes.
YOLOv11 has ~25 million weights → 100MB model file → slow to run.

**Quantization** is the process of reducing precision to make models smaller and faster:

```
FP32:  [0.38291047, -0.91847362, 0.12938741, ...]   4 bytes per number
INT8:  [49,          -118,         17, ...]            1 byte per number
                                                      4× smaller, 2-4× faster
```

### How It Works

You find the range of values in each layer (say -1.0 to +1.0) and map it to -128 to +127:

```
float_value → int8_value = round(float_value / scale) + zero_point
int8_value  → float_value = (int8_value - zero_point) * scale
```

A small amount of accuracy is lost, but for practical use (like object detection) the
difference is undetectable.

### The Types Used in SecondSense

| Type | Description | Used For |
|---|---|---|
| **FP32** | Full 32-bit float | Training, debug scripts |
| **INT8** | 8-bit integer | YOLOv11 on NPU |
| **W4A8** | Weights=4-bit, Activations=8-bit | Depth-Anything-V2 (extreme compression) |

**W4A8** is particularly aggressive — model weights are only 4 bits (16 possible values).
This gives a massive speedup (5ms depth inference) with acceptable quality loss for the
distance-estimation task.

---

## CHAPTER 9: What Is QNN?

### TFLite's Limitation

TFLite + NNAPI is a generic path. It works on many Android phones. But it goes through
multiple abstraction layers, and Qualcomm's NPU has features that NNAPI doesn't expose.

### QNN = Qualcomm's Direct NPU API

QNN (Qualcomm Neural Network SDK, also called AI Engine Direct) lets you talk to the
Hexagon NPU directly, without going through NNAPI. This is what gives you the 20ms/5ms
latencies in the bible — those are QNN numbers.

```
TFLite path:  App → TFLite → NNAPI → Qualcomm Driver → NPU
QNN path:     App → QNN SDK → NPU
                    (direct, faster, more control)
```

### The Model Format: `.bin` (Context Binary)

QNN doesn't use `.tflite` files. It uses **precompiled context binaries** (`.bin`).
These are generated by **Qualcomm AI Hub** (a cloud service you use in model conversion):

```
You submit:      PyTorch/ONNX model  →  Qualcomm AI Hub
They give back:  yolov11_det.bin      (compiled for Snapdragon 8 Elite HTP specifically)
                 depth_anything_v2.bin
                 yamnet.bin
```

These `.bin` files are compiled once, for a specific chip. They cannot run on a different
chip. They run at maximum efficiency on the chip they were compiled for.

### How QNN Works in Your App

Your app uses JNI (Java Native Interface) to call C++ code, which calls the QNN SDK:

```
Kotlin (QnnInferenceEngine.kt)
    │  calls via JNI
    ▼
C++ (qnn_backend_jni.cpp)
    │
    ├─ nativeInit():
    │    dlopen("libQnnHtp.so")          ← load Qualcomm's backend library
    │    QnnInterface_getProviders()     ← get the function table
    │    backendCreate()                 ← initialize the HTP backend
    │    deviceCreate()                  ← get a handle to the NPU device
    │
    ├─ nativeLoadModel("yolo", bytes):
    │    contextCreateFromBinary()       ← load the .bin file
    │    graphRetrieve("yolo")           ← get the graph handle
    │
    └─ nativeRun("yolo", inputBuffer):
         [TODO] graphExecute()           ← run inference on NPU
         return output tensors
```

**Current Status:** `nativeInit` and `nativeLoadModel` are written and correct.
`nativeRun`'s tensor binding is a TODO — it needs the physical QNN SDK headers to compile.
Until then, `StubQnnBackend` is used (returns `isReady=false`), and the app falls back to TFLite.

---

## CHAPTER 10: The InferenceEngine Seam — The Most Important Architecture Decision

### Why This Matters

You built 3 implementations of the same interface:

```
             InferenceEngine (interface)
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
  MockEngine    TfliteEngine   QnnEngine
  (fake data)  (current test) (future NPU)
```

All three produce the exact same `FrameResult`. Everything downstream — the decode layer,
the targeting, the audio, the haptics — doesn't know or care which engine is running.

### Why This Is Smart

When you swap from TFLite to QNN:
- Change ONE line in `EngineConfig.kt`
- That's it
- All the hard logic (drop-off detection, tier classification, sound synthesis) stays untouched

```kotlin
// EngineConfig.kt — the entire swap is here:
fun create(context: Context): InferenceEngine =
    when {
        QnnEngine.isAvailable() -> QnnInferenceEngine(context)  // ← flip this on
        else                   -> TfliteInferenceEngine(context)
    }
```

---

## CHAPTER 11: The Decode Layer — Pure Math, No Android

All the "understand what the models output" code lives in `inference/decode/`:

### `Preprocess.kt` — Image → Float Buffer
```
Camera bitmap (any size)
    → letterbox to 640×640 (grey padding, keeps aspect ratio)
    → convert pixels to floats (divide by 255 so values are 0.0..1.0)
    → pack into ByteBuffer in NHWC format
       N=1 (one image), H=640, W=640, C=3 (RGB channels)
    → also records: scale factor, padding offsets (to undo the transform later)
```

### `YoloDecoder.kt` — Raw Tensor → Detection List
The YOLO model outputs a tensor. Its shape depends on the export:

```
Old format (flat):       [1, 8400, 84]  → 8400 candidate boxes, each with 84 numbers
New format (transposed): [1, 84, 8400]  → same data, different axis order
```

`YoloDecoder` sniffs which format it is by looking at the tensor shape, then:
1. Reads each of 8400 candidates
2. Finds the max class score (indices 4..83 in the 84-number row)
3. Filters by confidence threshold (0.25)
4. Converts box from model-pixel coords back to normalized 0..1 frame coords
5. Runs NMS to remove duplicates

### `DepthSampler.kt` — Depth Map → Proximity Score
```
For each detected box:
    1. Find the center point of the box (cx, cy)
    2. Look up the depth value at that pixel in the depth map
    3. Normalize: min depth in frame = 0 (far), max depth = 1.0 (close)
    4. Return as "proximity" (0.0 = far away, 1.0 = right in front of you)
```

### `DropOffDetector.kt` — Find Floor Edges (Drop-offs, Stairs, Curbs)

This is a computer vision algorithm, not a neural network:

```
Step 1: Take the depth map (H×W float values)

Step 2: Apply Sobel filter in Y direction
        Sobel-Y finds HORIZONTAL edges — sudden changes in depth going down the image
        Formula: output[y,x] = depth[y+1,x] - depth[y-1,x]  (simplified)
        Large absolute value = sharp depth discontinuity

Step 3: Look at only the lower half of the frame (the floor region)
        Center column band (35% to 65% of width)

Step 4: Find the row with the strongest vertical depth gradient
        That row is where the floor drops away (a stair, curb, or edge)

Step 5: Check the SIGN of the gradient
        Going from "far" to "close" as you move DOWN = depth is inverted at that row
        = the floor is lower there = a drop-off

Step 6: If confirmed → fire HapticOutput.dropOff(urgency)
```

### `OpticalFlowEstimator.kt` — Lucas-Kanade Optical Flow

This answers: "Is the camera moving, or are the objects moving?"

If you're panning your head (phone), every pixel in the image shifts slightly even if
nothing in the world moved. This could fool the motion detector. Optical flow estimates
this "ego-motion" (camera-caused motion) so you can subtract it.

**Lucas-Kanade algorithm** (the classic, proven method):
```
For a small patch of pixels around a corner feature:
    We assume: pixel brightness doesn't change between frames
    Therefore: Ix*u + Iy*v + It = 0
    where Ix, Iy = spatial gradients, It = time gradient, u,v = flow we want to find

    This is one equation, two unknowns. Solve it over the whole patch (many pixels):
    Build a 2×2 matrix (the "structure tensor"), invert it, get [u, v].
    Iterate 5 times refining the estimate.
```

---

## CHAPTER 12: The Targeting Layer — Who Gets the Cue?

### The Problem
YOLO detects 5–15 objects every frame. You can only give ONE audio cue at a time
(the bible says 1–2 max, and 1 for flow mode). Who gets it?

### TargetSelector — The Rules (in order)

**Rule 1: Center crop** (flow mode only)
Ignore any detection whose center is outside the middle 30% of the frame width.
Walking forward, you only need to know about what's directly ahead.

```
Frame: |-------|[===CENTER 30%===]|-------|
        ignored  ← cued if closest →  ignored
```

**Rule 2: Closest wins**
Among the centered objects, pick the one with highest proximity (closest to you).

**Rule 3: Moving beats static at similar distance**
If a moving object (approaching person) is within 12% proximity of the closest static
object (parked chair), the moving one wins. A person walking toward you is more urgent
than a chair you can walk around.

### TierClassifier — How Confident Are We?

```
Input: detection confidence score (0.0..1.0)

WHITE tier (score ≥ 0.62):  "I see it and know what it is"
BLUE tier  (score 0.32..0.62): "Something's there, not sure what"
RED tier   (score < 0.32 OR no reliable class): "Something's there, can't name it"
```

**Hysteresis** (prevents flickering):
- To ENTER WHITE: score must be ≥ 0.62
- To LEAVE WHITE: score must drop below 0.52 (a 0.10 gap)
- This prevents the audio texture from switching between "clean" and "grainy" every frame

**Stability gate**: a new tier must appear for 2 consecutive frames before it's accepted.

### TemporalSmoother — Prevent Flicker
A detection must appear for 3 consecutive frames with the same label and roughly the
same position (azimuth within 20% drift) before any cue fires. A hand waved through
the camera for one frame never triggers a sound.

---

## CHAPTER 13: The Cue Engine — Turning Data Into Sound

### The Three Channels (The Crown Jewel)

Every piece of "where is it / how close / what is it" maps to exactly ONE audio/haptic channel:

```
DIRECTION → Stereo pan
    azimuth=0.0 → hard left ear
    azimuth=0.5 → both ears equally (center, directly ahead)
    azimuth=1.0 → hard right ear
    Uses "equal-power pan law":
        leftGain  = cos(azimuth × π/2)
        rightGain = sin(azimuth × π/2)

DISTANCE → Pulse repetition RATE (not pitch, never pitch)
    proximity=0.0 (far)  → pulse every 900ms (0.9 seconds apart)
    proximity=1.0 (close) → pulse every 120ms (very fast, like a parking sensor)
    This is the parking-sensor / reversing-car model. Pre-learned. No training needed.

IDENTITY → Auditory icon (timbre/character of the sound)
    "person"  → footstep thud (90Hz + 180Hz body, fast decay)
    "dog"     → rising yip (700..1600Hz sweep, two chirps)
    "vehicle" → low rumble (70Hz + 71.7Hz beating, creates roughness)
    "chair"   → wooden knock (320Hz, sharp decay)
    "door"    → two-part clack (500Hz latch + 150Hz panel)
    unknown   → neutral tick (300Hz, very short) ← honest RED-tier signal
```

**The cardinal rule:** Distance only changes the RATE. Identity only changes the TIMBRE.
Pan only changes the STEREO POSITION. These never bleed into each other.

### Confidence Tier → Audio Texture

| Tier | Sound treatment |
|---|---|
| WHITE | Clean, crisp audio. No modification. |
| BLUE | Amplitude noise mixed in (18% grain). Audibly sounds "unsure." |
| RED | Identity nulled to unknown-tick. Noise mixed in (12% grain). |

```kotlin
// applyTierTexture() — how grain is added:
val grain = if (tier == BLUE) 0.18f else 0.12f
// cheap deterministic noise via LCG (no Random object needed):
seed = seed * 1103515245 + 12345
val n = ((seed ushr 16) and 0x7FFF) / 32768f * 2f - 1f  // -1..+1
output[i] = (mono[i] * (1 - grain) + n * grain)
```

### The Degradation Ladder

```
Is depth available?
    NO  → SILENT_AUDIO (no audio cue this frame, but panic haptic may still fire)
    YES → Is tier WHITE AND label present?
              YES → FULL cue (icon + pan + rate)
              NO  → PROXIMITY cue (bare tick + grain texture + pan)

Is proximity ≥ 0.80?
    YES → panic() fires REGARDLESS of the audio rung above
          (double-hit haptic: buzz-gap-buzz — unmistakable "step back NOW")
```

### Voice Auto-Ducking (Speech Detection)

YamNet continuously listens. When it detects speech (someone talking near you):
- Navigation audio is reduced by up to 50% (never more — safety requirement)
- But ONLY if the object is far. If proximity ≥ 0.60, ducking is disabled entirely.
- RED tier is never ducked regardless.

```
"You're having a conversation on a busy street" → duck navigation audio slightly
"A car is 0.8 proximity away right now" → NO ducking, full volume
```

---

## CHAPTER 14: The Output Channels

### AudioOutput.kt

Uses Android's `AudioTrack` in STREAM mode:
- 44,100 Hz sample rate (CD quality)
- 16-bit signed PCM
- Stereo (2 channels: left + right)
- STREAM mode = push samples in real-time (vs static MODE_STATIC for short sounds)

Every icon is synthesized from scratch using basic wave math:
```kotlin
// A sine wave at frequency f, sample i, sample rate SR:
sin(2π × f × i / SR)

// Applied to "vehicle" (low rumble):
val a = sin(2π × 70.0 × i / SR)
val b = sin(2π × 71.7 × i / SR)  // two close frequencies = "beating" = rough texture
output = (a + b) * 0.5
```

Why synthesized instead of audio files? No asset files needed, works offline, and
the icon can be rendered at the exact millisecond the cue clock fires.

### HapticOutput.kt

Uses Android's `VibrationEffect` API with amplitude control (on supported hardware):

```
proximityPulse(0.3):   short buzz, amplitude = 40 + 0.3×215 = 104,  duration = 30 + 0.7×90 = 93ms
proximityPulse(0.9):   short buzz, amplitude = 40 + 0.9×215 = 233,  duration = 30 + 0.1×90 = 39ms

panic():               [0ms, 60ms buzz at 255, 40ms gap, 60ms buzz at 255]  ← double hit
dropOff(1.0):          [0ms, 70ms at 110, 50ms gap, 70ms at 180, 50ms gap, 110ms at 255]  ← escalating
hazardSound():         [0ms, 50ms at 200, 60ms gap, ×4]  ← 4 evenly-spaced taps
```

Each pattern is deliberately distinct so you can feel which hazard type it is without
looking at the screen.

---

## CHAPTER 15: Voice Goal-Seeking (Phase 4 — Not Yet Active)

### The Vision: "They describe, we vector"

Competitor apps (Seeing AI, Google Lookout) describe what's in a scene: "There is a chair."
SecondSense doesn't just describe — it guides your body toward or around it in real-time.

### How It Works (when QNN bridge is ready)

```
You say: "Find the door"
    │
    ▼
WhisperQnnRecognizer.transcribe(pcm)
    → "find the door"    (speech → text, on-device via Whisper-Tiny NPU)
    │
    ▼
TargetNoun.extract("find the door")
    → removes filler words (find, the, a, me, to, go...)
    → returns last content word: "door"
    │
    ▼
OwlVitQnnGrounder.ground(cameraFrame, "door")
    → returns bounding box of the door in the current frame
    → works for ANY noun, not just COCO classes
    │
    ▼
VectorToGoalController.cueFor(box, proximity)
    → CueTarget(azimuth=box.centerX, proximity=proximity, label="door", tier=WHITE)
    │
    ▼
CueEngine: pan toward the door, pulse rate increases as you approach
    │
    ▼
VectorToGoalController.hasArrived(): |centerX - 0.5| ≤ 0.12 AND proximity ≥ 0.75
    → "arrived" confirmation sound
```

---

## CHAPTER 16: The Debug Dashboard

When you open a browser on your laptop and go to `http://[PHONE_IP]:8085`, you see:

- Engine type and mode (TFLite/QNN/Mock, FLOW/SCAN_SEEK)
- Inference time in milliseconds
- Current cue target (label, direction, proximity)
- Confidence tier (WHITE/BLUE/RED) with color bar — RED pulses red
- Drop-off status (CLEAR or ⚠ AHEAD)
- Last ~50 frames of tier history (bar chart)
- All detections this frame (label, score, proximity)

This runs entirely on the phone's local network. No internet.
`DashboardServer.kt` is a 187-line embedded web server (NanoHTTPD) + HTML page.

---

## CHAPTER 17: The Offline Debug Scripts (Laptop Only)

Before writing any Kotlin code, you validated the math in Python:

### `debug_yolo.py`
Runs the `.tflite` YOLO model on a still photo. Answers:
- Are output boxes in `0..1` coordinates or `0..640` pixel coordinates?
- Which tensor is boxes, which is scores, which is classes?
- What are the top 10 detections on this image?

### `debug_sobel_dropoff.py`
Runs the depth model on a photo. Compares:
- Old method: compare two fixed row bands (FIRES/NO FIRE — no location info)
- New method: Sobel gradient (finds the EXACT ROW of the floor edge)
→ This is what convinced you to write V2 `DropOffDetector`

### `debug_optical_flow.py`
Validates the Lucas-Kanade math by:
- Taking a real photo
- Synthetically shifting it by a known amount (4.0px right, -2.5px down)
- Running LK tracking on a grid of points
- Verifying median estimate ≈ known shift
→ If it passes here, the same math is correct in Kotlin

---

## CHAPTER 18: The Complete Data Flow Summary

```
Camera frame arrives every ~33ms (30fps)
    │
    ▼
FrameAnalyzer.analyze()
    ├─ Rotate bitmap to upright (imageInfo.rotationDegrees)
    └─ engine.infer(bitmap)
         │
         ├─ Run YOLO: bitmap → letterbox → float buffer → model → raw tensor → decode → boxes
         ├─ Run Depth (every 3rd frame): bitmap → letterbox → model → depth map → EMA smooth
         │
         └─ FrameResult {
               detections: [
                  Detection { label, score, box, proximity, moving, approaching, tier }
               ]
               depthAvailable: true/false
               dropOff: true/false
               dropOffUrgency: 0.0..1.0
            }

FrameResult → MainActivity
    ├─ TargetSelector.selectWithTier(frame, classifier) → CueTarget?
    ├─ TemporalSmoother.update(target) → confirmed CueTarget?
    ├─ Calibration.apply(proximity) → re-referenced proximity
    ├─ BarometerMonitor.descendingConfirmed() → cross-check drop-off
    └─ cueEngine.update(target) → CueEngine's AtomicReference gets updated

CueEngine loop (MAX_PRIORITY thread, runs independently):
    ├─ Read current CueTarget (atomic)
    ├─ fireOnePulse(target):
    │    ├─ DegradationLadder.decide() → audioRung + panic flag
    │    ├─ AudioOutput.playMono(icon, pan, gain) ← IDENTITY + DIRECTION
    │    ├─ HapticOutput.proximityPulse(proximity) ← DISTANCE (primary)
    │    └─ if panic: HapticOutput.panic() ← SAFETY FLOOR
    └─ intervalMsFor(proximity) → sleep → repeat ← DISTANCE (rate)

HazardSoundDetector (parallel thread):
    ├─ if hazard keyword → HapticOutput.hazardSound()
    └─ if speech → CueEngine.setDucked(true)
```

---

## Quick Reference: Key Files

| File | What it does |
|---|---|
| `MainActivity.kt` | Orchestrates everything |
| `EngineConfig.kt` | Picks which engine to use |
| `TfliteInferenceEngine.kt` | Runs models via TFLite |
| `QnnInferenceEngine.kt` | Runs models via Qualcomm NPU |
| `qnn_backend_jni.cpp` | C++ bridge to QNN SDK |
| `Preprocess.kt` | Image → float buffer (shared) |
| `YoloDecoder.kt` | Raw YOLO tensor → detection list |
| `DropOffDetector.kt` | Depth map → drop-off warning |
| `TargetSelector.kt` | Which detection gets the cue |
| `TierClassifier.kt` | WHITE/BLUE/RED confidence |
| `CueEngine.kt` | Audio + haptic pulse loop |
| `AuditoryIcon.kt` | Synthesized sound icons |
| `HapticOutput.kt` | Vibration patterns |
| `AudioOutput.kt` | Stereo audio output |
| `HazardSoundDetector.kt` | YamNet sound classification |
| `MelSpectrogram.kt` | Audio → mel spectrogram |
| `DashboardServer.kt` | Live web debug dashboard |
