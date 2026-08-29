SecondSense Bible — Addendum: Session 4 (TFLite bring-up → research-driven hardening)
======================================================================================

This is a companion to `secondsense_bible_v4.md`, not a replacement — the core vision,
sonification design, and ethical invariants in v4 are unchanged and still govern everything
below. This document exists because a huge amount of concrete, load-bearing engineering
happened in one continuous session (real on-device bring-up, real bugs found on real
hardware, real model swaps, real research) and the "why" behind each of those decisions
deserves the same permanent home the rest of the system's reasoning lives in — not just a
line in a ticket tracker.

**Read this alongside `secondsense_bible_v4.md` (the why/vision) and
`secondsense_build_map_v3.md` (the ticket list).** This document is the missing middle layer:
*what we actually built this session, and the reasoning that led to each choice* — in the
same spirit as the Bible, scoped to what changed.

---

## 1. Object detection: why we replaced yolov11n with yolo26s

**What was there before**: `yolov11_det` exported at its default checkpoint, `yolo11n.pt` —
the smallest/fastest/least-accurate size in the YOLO11 family.

**The problem, found on real hardware, not in theory**: testing the app against real photos
(a steel travel bottle, photographed both in a plain close-up and in a cluttered desk scene)
showed the real object scoring only 0.07–0.34 confidence — under the app's detection
threshold. The model wasn't broken; it was just the smallest, least-capable checkpoint,
and it showed on real, ordinary objects, not edge cases.

**What we tried and measured, in order**:
1. Lowered `confThreshold` to 0.25 — recovered the missed detections, but this is a band-aid
   (accepting more noise, not fixing the underlying weak confidence).
2. Investigated `yolo26_det` (Ultralytics' newer generation — anchor-free, NMS-free head) at
   both its `n` (nano) and `s` (small) checkpoints, via a real cloud export through Qualcomm
   AI Hub (`qai_hub_models`), profiled on the actual Snapdragon 8 Elite chipset class.
3. **Measured, not assumed**: ran both new checkpoints against the same real bottle photos
   offline (no phone needed — see §7 below on the test workflow this enabled). `yolo26n`
   was a big win on one photo (0.07→0.725) but a regression on another (0.336→0.097) —
   nano-class models are simply inconsistent scene-to-scene. `yolo26s` won on **every**
   photo tested (0.34–0.57 range, clearing even the original stricter threshold) at only
   5.3ms measured NPU latency in Qualcomm's own cloud profiling.

**What we shipped**: `yolo26s` (checkpoint `yolo26s.pt`), same output tensor contract as
before (`boxes[1,N,4]` + `scores[1,N]` + `class_idx[1,N]`), so zero changes to the decode
layer (`YoloDecoder`, `CocoLabels`) — a pure model swap under an unchanged interface. Bundled
under the SAME filename (`yolov11_det.tflite`) the app already expects, so no Kotlin code
needed to change to point at it.

**`confThreshold`** settled at `0.30f` — between the original `0.35f` and the emergency
`0.25f`, chosen because `yolo26s`'s scores are meaningfully more trustworthy than the old
model's, so we don't need as much slack.

---

## 2. NNAPI: a hardware-delegate bug we found, diagnosed, and isolated

**The bug**: after swapping to `yolo26s`, live on-phone testing showed every single detection
scoring **exactly `1.00`**, across many different classes, on the same frame — a physical
impossibility for real model output (real photos never produce a dozen different classes all
scoring identically 1.0). The same model, run through the same decode code, on the same real
photo, gave correct 0.3–0.7 scores in an OFFLINE CPU-only Python test.

**Diagnosis**: MediaTek's NNAPI hardware delegate on this specific test phone was silently
mishandling an op in `yolo26s`'s larger post-processing graph — not crashing, not falling
back cleanly, just producing garbage. This is a real, documented class of hazard with vendor
NNAPI implementations (confirmed independently later by the deep-research pass — §8), and we
found it ourselves, empirically, before reading about it anywhere.

**Fix — decoupled per-model, not a blanket toggle**: depth had never shown this bug (it's an
unrelated graph, was never touched), so blanket-disabling NNAPI everywhere would have thrown
away real, working acceleration for no reason. `TfliteInferenceEngine` now takes two
independent flags:
```kotlin
private val useNnapiYolo = false   // CPU/XNNPACK — correctness > speed, on THIS test phone
private val useNnapiDepth = true   // NNAPI — never had the bug, keep the speed
```

**Known cost of this fix**: YOLO now runs on plain CPU on this specific (MediaTek) test
phone, which is slow (~1000ms/frame combined with depth). **This is explicitly a
test-phone-only workaround.** The real fix is the QNN native bridge (still not built — see
the pending list), which talks to Qualcomm's own compiler/runtime directly instead of through
Android's generic NNAPI abstraction — and Qualcomm's own cloud profiling already measured
this exact `yolo26s` model at 5.3ms with clean, correct scores on real Snapdragon 8 Elite
hardware. The bug is specific to this test device's NNAPI driver, not to the model or our
code.

---

## 3. Drop-off detection (#17): rebuilt from fixed-band to adaptive, twice

This ticket went through two real, evidence-driven iterations this session — worth recording
both, because the failure of the first design is as instructive as the success of the second.

### V1 (inherited from an earlier session): fixed near/mid depth bands
Compared a fixed "near" band (bottom 80–99% of frame) against a fixed "reference" band
(55–72%) — fires if the near band reads meaningfully farther than the reference band.

**Found broken on real stairs**: tested against a real staircase photo (chest-height,
foyer framing) — the band comparison never fired (diff = -0.09, nowhere near the
0.20 threshold), because the real staircase geometry simply didn't sit inside the detector's
hardcoded assumed zones. A fixed-geometry heuristic is blind to anything outside the geometry
it assumes.

### V2: adaptive Sobel-gradient localization + local sign check
Replaced the fixed bands with a two-step adaptive algorithm:
1. **Scan the lower half of the depth map for the sharpest vertical gradient** (a real,
   weighted 3×3 Sobel kernel — not a plain row-difference, which we initially mis-ported and
   which under-weighted the signal by ~4x, missing real edges entirely until caught and
   fixed) — wherever the discontinuity actually is, not a fixed zone.
2. **Confirm the SIGN**: proximity must read farther just past the located edge than just
   before it (the physical signature of "the ground disappeared here"). This is what
   distinguishes a real drop-off from an ordinary object boundary (e.g., a desk edge, where
   proximity typically reads NEARER past the edge, not farther) — validated on a real desk
   photo (local_diff = -0.084, correctly rejected) against a real staircase photo
   (local_diff = +0.058, correctly fires).

**Enrichment on top of the fire decision**: the located edge's row position (0.5 = just
entering the search zone, 1.0 = right at the feet) now scales haptic urgency — a distant edge
buzzes softer, an imminent one buzzes at full intensity — instead of one flat-intensity
warning regardless of distance.

**Known remaining gap, found and left honestly open**: a second real staircase photo (viewed
from the top, looking down, own foot in frame — the actual "walking toward a descending
staircase" scenario) still doesn't fire. Diagnosis: the depth model's OWN estimate reads the
wrong physical sign on this specific shot (low light, unusual surface texture) — our
gradient/sign logic is working correctly against what the depth model reports; the depth
model itself is the weak link here, not our detector. This is why the barometer cross-check
(§4) exists.

---

## 4. Barometer cross-check (research-driven, §7 of the research candidates doc)

**Why this exists**: directly targets the exact unresolved gap above. A depth model can be
fooled by lighting and texture; a barometer cannot — it measures real atmospheric pressure,
which reliably rises by ~0.12 hPa per metre of descent, independent of anything the camera
sees.

**What it does**: `BarometerMonitor` reads `Sensor.TYPE_PRESSURE` continuously, keeps a
rolling 3-second window, and reports whether the recent trend is consistent with the user
physically descending right now.

**How it's used — deliberately NOT a replacement for vision**: the camera/depth pipeline
stays the primary, safety-critical trigger (a barometer alone is too slow/noisy to be the
sole signal — you don't want to wait seconds of walking before any warning fires). When
vision fires a drop-off AND the barometer independently confirms real descent, haptic urgency
is bumped to maximum — two independent physical signals agreeing is a stronger basis for
urgency than one uncertain camera guess alone.

**Degrades honestly, not silently**: if a device has no barometer (confirmed: our current
MediaTek test phone has none), the app now says so plainly in the HUD (`baro: no sensor on
this device`) rather than staying quietly invisible about a feature that simply can't run.
**Confirmed via spec lookup: the iQOO 15 (the actual target device) does have a barometer**,
alongside its Snapdragon 8 Elite Gen 5 — this feature is real and testable once you're on
that hardware, just not provable on the current test phone.

---

## 5. Laptop dashboard (#30) + confidence-tier overlay (#32)

**Why**: the pitch/demo needs a way for judges to SEE what the pipeline is doing while the
presenter (potentially blindfolded, per the demo plan) walks the course — the phone's own
screen isn't the right audience-facing surface, and the whole system must work with zero
internet (the airplane-mode demo requirement, #31).

**What was built**: the phone runs its own tiny embedded HTTP server (`NanoHTTPD`, fully
local — no cloud dependency anywhere), serving a self-contained dashboard page. A QR code
(generated on-device via `ZXing`, also fully offline) lets a laptop on the same Wi-Fi/hotspot
join instantly. Verified end-to-end: phone hotspot → laptop connects → scans QR → live
telemetry streaming, confirmed with the phone's hotspot actually active (not just USB
tethering).

**#32 on top of #30**: a full-page color bar/glow reflecting the live WHITE/BLUE/RED
confidence tier — visible from across a room, not buried in a small badge — plus a rolling
50-frame tier-history strip, so a judge can see STABILITY over time (a steady white run reads
very differently from a flickering mess, even at the same instantaneous tier).

**A real bug found and fixed along the way**: the dashboard initially showed nothing (no QR,
no URL) despite the server actually running correctly — traced to MIUI restricting
`NetworkInterface` enumeration without `ACCESS_NETWORK_STATE`/`ACCESS_WIFI_STATE` declared,
even though raw socket use alone doesn't formally require them. Fixed by declaring both
permissions plus a `WifiManager`-based fallback path.

---

## 6. Model conversion pipeline: from theoretical to actually configured

At the start of this session, `qai-hub configure` had never been run on this machine — the
bundled `.tflite` models had been produced elsewhere and just copied in. This session:
- Installed `ultralytics` (required for YOLO26 export, wasn't in `requirements.txt`).
- Configured this machine's `qai-hub` CLI with a real API token, confirmed against
  Qualcomm's device farm (`qai-hub list-devices` returns real Snapdragon targets).
- Used this real, working pipeline to export and profile `yolo26n`/`yolo26s` against actual
  cloud-hosted Snapdragon 8 Elite hardware — not simulated, not estimated.

**This machine can now perform real model conversions end-to-end**, which it could not do
before this session. This directly unblocks any future model swap (§8's research candidates
that need zoo verification can now actually be checked, not just estimated from docs).

---

## 7. The offline test workflow — arguably the most valuable thing built this session

**The problem it solves**: every hypothesis about a bug used to require a full
rebuild→install→relaunch→NNAPI-warmup cycle on the phone — 60–90 seconds per guess, with a
MIUI device that additionally blocks `adb` from waking the screen or granting permissions
headlessly.

**What was built**: standalone Python scripts (`debug_yolo.py`, `debug_sobel_dropoff.py`,
`debug_dropoff_v2.py`) that load the exact same `.tflite` models the app bundles, using
`ai-edge-litert` (a lightweight, offline TFLite interpreter, no TensorFlow install needed),
and run them against a real photo in ~2 seconds. Every model swap, every threshold change,
every algorithm rewrite this session was validated this way FIRST, before any Kotlin was
written or any phone touched.

**Why this matters beyond convenience**: it's what let us catch our own mistakes fast — the
Sobel under-weighting bug, the wrong-threshold-after-kernel-swap risk with Scharr, the actual
confidence numbers behind the yolo26n vs yolo26s decision — all found and fixed in seconds,
not in a rebuild cycle. This workflow should be the default first move for any future model
or algorithm change.

**On-device debug tooling**: a companion `DebugActivity` (isolate YOLO-only/depth-only/full
pipeline, live box overlay, individual audio/haptic/spearcon test buttons) exists for the
things that genuinely need a real phone (camera framing, real hardware haptics, real audio
output) — but anything about model *accuracy* should be checked offline first.

---

## 8. Deep research pass — what we evaluated, what we acted on, what we didn't

A structured deep-research pass covered all 10 pipeline stages (see
`secondsense_research_candidates_v1.md` for the full catalog — every algorithm/model
surfaced, with an explicit assessment of each). Summary of what actually happened as a
result:

- **Scharr operator** (proposed replacement for our Sobel gradient) — tested offline against
  all 4 real test photos. Found to produce IDENTICAL edge locations to Sobel, just a uniform
  4x scale factor — no demonstrated accuracy benefit on our actual data, real re-tuning risk.
  **Rejected**, on evidence, not assumption.
- **Barometer cross-check** — evaluated as the single highest value-to-effort item in the
  whole document (real physics, no new model, directly targets a confirmed bug). **Built**
  (§4 above).
- **YOLOE-26 (open-vocab)**, **Depth Pro** (metric depth, claimed to fix staircase inversion),
  **PIDNet/RepViT** (walkable-path segmentation), **EfficientAT** (better hazard-sound
  classifier) — all flagged as **unverified against our actual `qai_hub_models` zoo** and not
  pursued this session. The research document's specificity (exact parameter counts, exact
  claimed export paths) outran what we'd actually confirmed real.
- **VINS-Mono (full visual-inertial SLAM)**, **conformal-prediction confidence calibration** —
  both legitimate, real techniques, both explicitly assessed as the wrong scope for the
  remaining event timeline (multi-week systems work, or requiring a labeled calibration
  dataset we don't have).

The full reasoning per item lives in `secondsense_research_candidates_v1.md` — this section
is just the summary of what changed as a result.

---

## 9. Depth EMA fusion (research candidate, built and confirmed working)

**Why**: raw single-frame monocular depth is noisy pixel-to-pixel even when nothing in the
scene changed (sensor noise, lighting flicker, model non-determinism). That noise feeds
directly into `DropOffDetector`'s gradient search, so smoothing it reduces false-positive
hazard triggers from noise alone.

**Validated before writing Kotlin**: simulated 10 consecutive frames of realistic jitter on a
real depth map, applied EMA (alpha=0.3) — confirmed 47% frame-to-frame noise reduction with
negligible lag on the real underlying signal.

**What shipped**: `DepthTemporalSmoother.kt` in the shared, runtime-agnostic `decode/` layer
(TFLite and the future QNN engine share it unchanged). Applied *before* `DepthSampler.parse()`
so both the percentile normalization and `DropOffDetector`'s gradient search see the cleaned
signal. Resets automatically on a debug-mode switch or `close()` so stale state never blends
into a fresh scene.

**How to verify it yourself (built into the Debug Panel specifically for this)**: open
DebugActivity, point the phone anywhere, and read the new HUD line:
```
depth center: raw=0.XXX  smoothed=0.XXX  (watch smoothed hold steadier)
```
Hold the phone still for 5-10s and watch both numbers — `raw` should visibly jitter while
`smoothed` moves more gently. For a more dramatic test, point at something static, then
suddenly point at your hand very close up, then hold still — `raw` jumps instantly, `smoothed`
visibly eases into the new value over about a second. **User-confirmed working** this session.

---

## 10. YamNet hazard-sound detection (ticket #33) — a new sensing modality entirely

**Why this ticket, and why now**: everything else in the pipeline is vision-based — it is
structurally blind to anything outside the camera's field of view. A car horn from behind, a
siren around a corner, an alarm in an adjacent room: none of these can ever be caught by
YOLO or depth, no matter how good those get. Audio is a genuinely independent sensing channel,
not an incremental improvement on vision.

### The path to a working export — a real dependency chain, not a simple wire-up
Every previous model swap this session (`yolo26s`, the depth candidates) needed at most one
missing pip package. YamNet needed four, escalating in difficulty:
1. `resampy` — normal PyPI package, one-line fix.
2. `soundfile` — same.
3. `torch_audioset` — **not on PyPI at all.** Confirmed via `pip install` failing outright,
   and confirmed it isn't even declared in YamNet's own `requirements.txt` (a real packaging
   gap in this release of `qai_hub_models`). Verified the correct source via web search
   before touching anything — [w-hc/torch_audioset on GitHub](https://github.com/w-hc/torch_audioset)
   — and installed it directly from there **only after explicit user authorization**, since
   installing from an unofficial source is treated differently from a normal package registry.
4. A Windows console encoding bug (`UnicodeEncodeError` on a `⏳` progress character) that
   looked like an export failure but wasn't — the actual cloud compile/inference jobs had
   already succeeded; only the local progress-printing crashed. Fixed with
   `PYTHONIOENCODING=utf-8` (the same fix `convert.py` already carries, which is exactly why
   it's there).

### The export itself — genuinely excellent numbers
Real Qualcomm cloud profiling on Snapdragon 8 Elite QRD:
- **0.2ms inference time** (yes, sub-millisecond — YamNet is a small, efficient classifier)
- **Runs entirely on-NPU**, zero CPU/GPU fallback ops
- **PSNR 65.97dB** on-device-vs-reference accuracy match (>30dB is considered good; this is
  more than double that)

### The real engineering: YamNet doesn't take raw audio
The exported model's input shape is `[1,1,96,64]` — a **log-mel spectrogram** (96 time frames
× 64 mel-frequency bins over 0.96 seconds), not a waveform. This is genuinely new signal-
processing work, not a model file swap:
1. Found the exact preprocessing spec by reading `torch_audioset`'s own source (the code that
   produced the reference PSNR numbers above, so provably correct): 16kHz sample rate, 25ms
   STFT window / 10ms hop, 512-point FFT, 64 mel bins spanning 125-7500Hz, MAGNITUDE (not
   power) spectrogram, `log(mel + 0.001)`.
2. **Validated a from-scratch Python implementation against the real exported model BEFORE
   writing any Kotlin** (`debug_yamnet.py`) — ran it on YamNet's own reference recording
   (`speech_whistling2.wav`) and confirmed it correctly identified "Speech" in the first
   second, then "Whistling" in the middle — matching the filename's actual content exactly.
3. **Ported the validated math to Kotlin** (`MelSpectrogram.kt`) — a from-scratch radix-2
   Cooley-Tukey FFT (pure Kotlin, no external DSP library, since 512 is a clean power of 2)
   plus a triangular mel filterbank computed once at load time.
4. Discovered YamNet's raw output is **logits**, not probabilities (real values ranged
   roughly -9 to +5 in testing — outside any valid 0..1 probability range) — applied sigmoid
   before thresholding, matching the model's own published `CLASSIFIER_ACTIVATION = 'sigmoid'`
   spec.

### What got built
- `MelSpectrogram.kt` — the validated DSP pipeline (decode-layer style: pure signal
  processing, no Android dependency).
- `HazardSoundDetector.kt` — continuous 1-second non-overlapping audio capture (matches the
  reference implementation's own `PATCH_HOP_SECONDS=1.0`, chosen specifically because
  overlapping windows aren't needed for inference), classification, and a curated keyword
  match against YamNet's 521 AudioSet classes (vehicle horns, car/civil-defense/police/fire
  sirens, alarms, explosions, gunshots — not the hundreds of irrelevant classes like music or
  animal sounds YamNet also recognizes).
- `HapticOutput.hazardSound()` — a new, distinct 4-tap pattern, deliberately different from
  `dropOff()`'s 3 escalating pulses and `panic()`'s 2 sharp hits, so a hazard SOUND (something
  you can't see) is tactilely distinguishable from a hazard you can see or are about to hit.
- Wired into the HUD (`hazard listen: <label> (<score>)` updating every ~1s, plus a
  `⚠ LAST HAZARD` line on a real trigger), the laptop dashboard's JSON feed, and the mic
  permission flow (starts automatically the moment RECORD_AUDIO is granted, whether via the
  existing voice-search button or independently).

### Honest limitation, stated plainly
The model and the preprocessing math are both validated against real audio (the PSNR number
above, and the Speech→Whistling reference-recording test). The specific **hazard keyword
list and the 0.3 score threshold have not been validated against a real horn, siren, or alarm
recording** — no such sample was available this session. Treat the threshold as a sensible
starting point that may need tuning once real hazard-sound data is available, not a proven
value the way `confThreshold=0.30f` for YOLO was (that one WAS validated against real bottle
photos).

### How to verify it yourself
1. Install the latest build and launch the app.
2. Tap **"Find… (voice)"** once — this grants `RECORD_AUDIO` (MIUI blocks `adb` from doing
   this headlessly, so it must be a real tap) and, as a side effect of the same permission
   grant, also starts hazard-sound listening.
3. Watch the main HUD for a new line: `hazard listen: <label> (<score>)` — this updates
   roughly once per second with whatever YamNet's single most confident guess is for the
   current second of audio, HAZARD OR NOT (e.g. it might say "Speech" while you're talking
   near the phone — that's expected, it proves the pipeline is alive).
4. To test an actual hazard trigger: play a car horn, siren, or alarm sound near the phone's
   mic (a YouTube clip works fine). Watch for:
   - A toast: `⚠ Hazard sound: <label>`
   - A new `⚠ LAST HAZARD: <label> (<score>)` line appearing in the HUD
   - A distinct 4-tap buzz pattern (clearly different from the drop-off/panic patterns)
5. If nothing ever matches even with an obvious siren/horn sound playing close to the mic,
   that's exactly the "honest limitation" above surfacing — the threshold or keyword list
   would need tuning against real captured hazard audio, which is the natural next step for
   this ticket in a future session.

---

## 11. Lucas-Kanade ego-motion compensation (research candidate) — working, tunable, not final

**Why**: `MotionTracker`'s `moving` flag previously compared a detection's raw box-position
shift against a fixed threshold — but a chest-mounted camera shifts a static object's box
just as much when the WEARER turns their body as when the OBJECT actually moves. That's a
real, structural false-positive source: a static pole could read as "moving" purely because
you turned your head toward it.

**What it does**: estimates the camera's own motion each frame (a robust median of sparse
Lucas-Kanade flow over a background point grid — median specifically because it's outlier-
resistant, so the minority of points that land on actually-moving objects don't skew the
estimate), then subtracts that ego-motion from each detection's raw box shift before deciding
if it's genuinely moving.

**Validated before any Kotlin was written**: a from-scratch Python LK implementation was
tested against a KNOWN synthetic pixel shift (4.0, -2.5) applied to a real photo — recovered
(3.98, -2.50) via median across 12 tracked points. The Kotlin port is the same math.

**A real bug caught during the port, before it ever reached a build**: the first Kotlin draft
used `return@repeat` inside a `repeat { }` block to try to break out of the iterative
refinement loop early — in Kotlin, `return@repeat` only skips to the NEXT iteration, it does
NOT exit the loop, so the early-exit-on-convergence and early-exit-on-out-of-bounds logic
would have silently done nothing. Caught by re-reading the code before building, not by a
runtime failure — fixed with proper labeled loops (`iterations@ for (...) { ... break@iterations }`).

**User-confirmed this session**: "working fine for now but not good" — i.e., functionally
wired correctly (no crash, ego-motion visibly tracks real panning, static objects mostly stay
`moving=no` during a pan) but not yet tuned for real accuracy. Below is exactly what's tunable
and what would meaningfully improve it.

### The tunable parameters ("weights") — what they are, and what changing each one does

**In `OpticalFlow.kt`:**
- `WINDOW_RADIUS = 7` — the half-size (in pixels, at the 160×120 downsampled resolution) of
  the patch used to track each point. Bigger = more robust to noise but blurs over small/thin
  objects and is slower; smaller = more precise localization but noisier on flat/low-texture
  regions. 7px on a 160×120 frame is a fairly large fraction of the image — worth trying
  smaller (4-5) if tracking feels too "smeared."
- `MAX_ITERS = 5` — how many Newton-Raphson refinement passes per point. More = more accurate
  convergence for LARGER motions, at linear cost. 5 is a reasonable default for small
  frame-to-frame motion; if real walking-pace testing shows flow visibly lagging real motion,
  raising this (or, better, adding a coarse-to-fine pyramid — see below) is the fix.
- `CONVERGE_EPS = 0.01f` — the per-iteration displacement below which refinement stops early
  (saves compute once a point has converged). Lower = more precise but slower.
- `det < 1e-3` threshold (in `trackPoint`) — the minimum gradient-matrix determinant a window
  needs to be considered trackable. Too low = tracks flat/textureless regions unreliably
  (garbage flow vectors); too high = rejects legitimate-but-subtle-texture regions, so the
  ego-motion grid could end up with too few valid points on a plain wall or floor.
- `FLOW_GRAY_W/H = 160×120` (in the two engine files, not OpticalFlow itself) — the
  grayscale downsample resolution flow tracking runs at. Higher = more precise motion
  estimates but proportionally slower (LK cost scales with resolution); lower = faster but
  coarser. 160×120 was chosen for speed, not validated for accuracy tradeoff specifically.

**In `MotionTracker.kt`:**
- `matchGate = 0.12f` — how close (normalized frame distance) a detection must be to last
  frame's same-label detection to count as "the same object." Too tight = loses track on fast
  motion or a jittery box; too loose = could match two DIFFERENT same-class objects to each
  other (e.g. two different people), scrambling the motion signal entirely.
- `movingThreshold = 0.02f` — how much ego-motion-COMPENSATED shift counts as genuinely
  moving. This is the single most important tuning knob for the "not good yet" feedback —
  too low and normal detection jitter (YOLO's box wobbling a few pixels frame to frame even
  on a static object) reads as movement; too high and slow-but-real motion gets missed.
  **This is the first thing to tune** with real walking-pace test data.
- `approachGain = 4f` — unrelated to optical flow (this scales the DEPTH-based `approaching`
  signal), included here only because it lives in the same file and is easy to confuse with
  the motion-flow parameters above.

### Concrete ways to make it meaningfully better (ranked by effort)

1. **Tune `movingThreshold` against real recorded walking-pace data** (cheapest, do this
   first) — capture what YOLO's box jitter looks like on a genuinely static object at normal
   distance, and set the threshold comfortably above that noise floor.
2. **Add a coarse-to-fine image pyramid** (the standard, textbook Lucas-Kanade robustness fix)
   — track at a very small/blurred scale first for a rough estimate, then refine at
   progressively finer scales. This is THE standard fix for exactly the "not good enough yet"
   feedback when motion is larger than a few pixels between frames (fast pans, or frame skips
   under load) — single-scale LK (what's built now) only converges reliably for small
   inter-frame motion, which was an explicit, documented tradeoff at build time, not an
   oversight.
3. **Track detection-specific windows sized to the box**, not a fixed 7px radius for
   everything — a small, distant object and a large, close one have very different apparent
   motion scales; a fixed window size is a real simplification.
4. ~~Use RANSAC instead of plain median for the ego-motion estimate~~ — **DONE, same session.**
   `OpticalFlow.estimateEgoMotion` now exhaustively tries each tracked background point's own
   flow as a hypothesis (cheap: the grid is small, ~9-16 points, so O(n²) is negligible),
   counts how many other points agree within 1.5px, and returns the MEAN of the LARGEST
   agreeing cluster — replacing the old per-axis median entirely. This fixes the specific
   failure mode median had: median computes the X-estimate and Y-estimate independently, so
   it could land on a "phantom" vector matching NO actual tracked point when the grid splits
   between background and a large object — RANSAC instead always returns an actual, physically
   coherent cluster's mean. **Honest limitation this does NOT fix** (inherent to any
   majority-vote method, not a bug): if the moving object covers MORE than half the
   background grid points, the largest consensus cluster could legitimately be the object's
   motion rather than the camera's — a genuinely harder problem needing a different signal
   (e.g. depth-aware point weighting) to solve properly.
5. ~~Feed the compensated flow into `approaching` too~~ — **DONE, same session.**
   `MotionTracker` now adds `flowApproachGain * compensatedShift` on top of the depth-based
   approaching value — but ONLY when the depth signal is already non-negative (not receding).
   Reasoning: a single 2D box-centroid flow vector can't tell us the SIGN of depth change on
   its own (someone walking briskly ACROSS your path looks the same, flow-wise, as someone
   moving toward you) — so flow is used purely as an urgency BOOST on an already-approaching
   or flat depth signal, specifically to compensate for depth's lag (it only updates every
   Nth frame, see `depthEveryN`) versus a fast object's real-time motion. It never flips a
   receding read into an approaching one.

Both #4 and #5 build directly on top of the validated optical-flow foundation from earlier in
this section — neither needed new offline validation of their own, since the underlying flow
math was already proven correct against the known-shift test.

**Remaining roadmap (not done yet)**: #1 (tune `movingThreshold` against real data), #2 (image
pyramid for larger motions), #3 (per-detection window sizing) are all still open — none were
skipped by oversight, each is a real, scoped tradeoff made explicitly to ship a working,
validated version this session rather than an unbounded research project.

---

## 12. Voice Auto-Ducking (ticket #34) — built by reusing existing infrastructure

**Why this one, and not #35/#36/#37**: the build map's own text marks #35 (Llama narration),
#36 (gesture control), and #37 (FastSAM) as "Seed/pitch-only," "Cut first," "not worth
risking the spine" — the team's own prioritization already decided against building these
under time pressure, and #37 additionally contradicts this session's own research conclusion
(§8: FastSAM/SAM-family models are unsuited for continuous real-time use). #34 is marked
"Supporting" priority and was flagged feasible specifically BECAUSE it's a thin layer on top
of infrastructure already built this session (YamNet, §10) — not a new subsystem.

**What it does**: `HazardSoundDetector` already classifies audio every ~1 second (built for
#33). This adds a second, independent check on the SAME classification pass: scan the full
521-class score array for speech-related classes ("Speech," "Conversation," "Narration,
monologue," "Child speech," "Shout," "Whispering") above a 0.3 threshold — the same
scan-the-full-array pattern `checkHazards()` already uses for hazard sounds, not just the
single top-1 class (so speech ducks the cues even on a frame where something else happened
to score marginally higher overall).

### REVISED after a real safety concern was raised (same session)

The first version applied a flat 80% gain cut to EVERY cue whenever ANY speech was detected —
matching the ticket's literal spec, but with a real problem the user caught immediately:
YamNet can't distinguish "someone is talking TO me" from "ambient chatter I'm not even part
of." A blanket duck in a noisy street would near-silence navigational audio in exactly the
environment that needs it most, and a flat cut applies just as hard to a close-range warning
as to a comfortable-distance one — dropping identity/direction audio right when a user is
distracted by conversation AND something is nearby is a genuine safety regression, not a
UX nicety.

**Fixed with two changes, applied together:**
1. **Max duck reduced from 80% to 50%** (`DUCK_MIN_GAIN = 0.5f`) — never drops toward
   near-silent, whatever else is true.
2. **Urgency-gated, not blanket**: `duckFactorFor(target)` computes the actual multiplier
   PER CUE, not once globally. Ducking fades OUT entirely as `target.proximity` approaches
   `DUCK_SAFETY_PROXIMITY = 0.6f`, and is fully disabled outright for RED tier (identity
   unknown but close). A target far away and comfortably WHITE/BLUE tier gets the full 50%
   reduction; a target at or above the safety-proximity threshold gets NO reduction at all,
   regardless of how much speech is happening nearby.

**Still deliberately does NOT touch haptics** — Bible §5.2's primary "how close" channel is
completely unaffected by any of this either way.

**Known remaining limitation, stated plainly**: this still can't distinguish directed speech
from ambient chatter — it only softens the CONSEQUENCE of that ambiguity (bounded duck depth,
safety floor near urgent targets) rather than solving the ambiguity itself. Solving that
properly would need real audio source/direction discrimination (e.g. sound source
localization toward the user's own mouth vs. elsewhere), which is out of scope here.

**Not offline-validated the way earlier features were** — reuses the SAME classification
pipeline already validated for #33 (real Speech/Whistling recognition confirmed against a
reference audio file); the new logic (keyword scan + proximity-gated gain) is simple enough
to be low-risk without a separate offline pass.

### How to test it
1. Grant mic permission (tap "Find… (voice)" once, same as for #33) so hazard/speech
   listening starts.
2. Turn Sonification ON so cues are actually playing.
3. Watch the HUD for a new line: `ducking: off` or `ducking: ON — speech detected, cues quiet`.
4. **Start talking near the phone** (or have someone talk near it) — within ~1 second, the
   line should flip to `ON`, and the audio cues (not haptics — those keep buzzing normally)
   should become noticeably quieter.
5. **Stop talking** — within ~1 second, it should flip back to `off` and cues return to full
   volume.
6. Confirm haptics (test buzz, proximity pulse) are completely unaffected by ducking state at
   any point — that's a deliberate design invariant, not an oversight if it doesn't duck.

---

## 13. What's still open (honest status, not aspirational)

- **QNN native bridge**: still a stub. This is the actual fix for the NNAPI bug (§2) and the
  unblock for Whisper/OWL-ViT voice goal-seeking. Highest-leverage remaining item.
- **Depth model wrong-sign bug on real descending stairs**: not fixed, only cross-checked
  (barometer, §4). The underlying depth model limitation is still there.
- **Voice pipeline** (`#10/#11/#26/#27`): unchanged this session, still blocked on the native
  bridge.
- **`#33` (YamNet hazard sounds)**: built this session (§10 above), pending real-audio
  threshold validation. **`#34` (voice auto-ducking)**: built this session (§12 above).
  **`#35` (Llama narration), `#36` (gesture control), `#37` (FastSAM)**: deliberately NOT
  built — all three are marked "cut first"/"pitch-only" in the build map itself, and #37
  additionally contradicts this session's own research conclusion (§8). Skipped on purpose,
  not by oversight; see §12 for the full reasoning.
- **Phase 7 proof work** (`#38` metrics, `#39` rehearsal, `#42` backup, `#43` pitch): all
  still pending real on-phone runs.

This addendum captures reasoning, not just status — for a live status snapshot at any given
moment, the handoff docs (`secondsense_handoff_v3.md` and onward) remain the source of truth.
