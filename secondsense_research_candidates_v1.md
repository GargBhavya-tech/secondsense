SecondSense — Research Candidates (Deep Research Synthesis, v1)
================================================================

Source: a deep-research pass run against the 8-part prompt covering every stage of the
pipeline (object detection, depth, classical CV, segmentation, audio, speech/voice, sensor
fusion, confidence calibration, NPU optimization, prior art). This document captures **every**
algorithm/model the research surfaced, plus my own assessment of each — verification status,
feasibility given our actual `qai_hub_models` deployment path, and whether it's realistic
before the Aug 29-30 event.

**Read the "MY ASSESSMENT" line on each entry before acting on anything.** The research doc
contains some very specific claims (exact API names, exact parameter counts, exact physical
constants) that I have NOT independently verified against the real model zoo or real hardware.
Given this session's own history (NNAPI silently corrupting scores, a Sobel scaling bug that
looked fine on paper) I'm treating every unverified specific as "plausible, not confirmed"
until we've actually checked it — the same standard we held our own code to.

---

## 1. Object Detection

### YOLO26 (Nano / Small) — current baseline
- **What research says**: NMS-free, DFL-module removed, ProgLoss + STAL for small/close-up
  target calibration, MuSGD optimizer, up to 43% faster than YOLO11.
- **MY ASSESSMENT**: This is what we're ALREADY running (`yolo26s`, validated this session
  with real photos — see the `debug_yolo.py`/offline test history). The NMS-free claim is
  plausible (matches Ultralytics' own YOLO26 release notes) but we validated our actual
  export empirically rather than trusting the marketing; our own decode still runs its own
  NMS (`YoloDecoder.nms()`), so if the export genuinely pre-NMSes, that's a redundant
  no-op today, not a bug, not urgent to change.
- **Priority**: Already shipped. No action.

### RF-DETR, RT-DETRv2, D-FINE
- **What research says**: Higher COCO mAP than YOLO26 via transformer/deformable-attention
  backbones (DINOv2, FDR refinement).
- **MY ASSESSMENT**: Research itself concludes these are NOT mobile-NPU-deployable
  ("structurally hostile," "quantization brittleness," "GPU bound"). I agree with this
  conclusion on priors — ViT attention layers are the single most common cause of NNAPI/QNN
  op-fallback we could hit (echoes the exact class of bug we found this session with
  NNAPI+yolo26s). Not verified hands-on, but the research's own reasoning is sound here.
- **Priority**: Skip. Research agrees, I agree.

### YOLO-World, OWL-ViT
- **What research says**: Open-vocabulary detection, but heavy cross-attention hurts mobile
  latency.
- **MY ASSESSMENT**: `owl_vit` is literally in our `models.json` already — and its conversion
  **already failed** in an earlier session (noted in the pending-tickets list). This matches
  the research's stated concern. `yolo_world` is also already flagged in our own `models.json`
  as the fallback. Nothing new here, just confirms our existing plan.
- **Priority**: Already known/scoped (#11 fallback). No new action from this research.

### YOLOE-26 (RepRTA — Re-Parameterizable Region-Text Alignment)
- **What research says**: Folds text embeddings into the classification head at export time,
  eliminating the text-encoder at runtime — open-vocab detection at closed-set-YOLO latency.
- **MY ASSESSMENT**: **UNVERIFIED — check before committing any time.** Our own zoo listing
  earlier this session showed `yoloe_seg` (segmentation), not a confirmed "YOLOE-26" det
  variant with this specific RepRTA export path. The mechanism described is architecturally
  plausible (Ultralytics does ship a real YOLOE family), but "RepRTA" as a specific
  productized export flag is not something I've confirmed exists in `qai_hub_models` 0.61.0.
  **Five-minute check before any commitment**: `python -m qai_hub_models.models.yoloe_seg.export --help`
  (or equivalent det variant) to see what's actually exportable.
- **Priority**: Verify first, then reconsider.

---

## 2. Depth / Monocular Geometry

### Depth-Anything-V2 — current baseline
- Already shipped, validated extensively this session (offline Python tests, real photos).
- **Known real bug**: produces an inverted/wrong-sign gradient on a genuine descending
  staircase (`stairs2.jpeg`), confirmed via our own `debug_dropoff_v2.py`.

### Depth Pro (Apple)
- **What research says**: 6.1M params, absolute METRIC depth (not just relative/affine),
  convex upsampling for sharp boundaries, explicitly solves the staircase-inversion class of
  problem by anchoring to real-world scale instead of ordinal-only depth.
- **MY ASSESSMENT**: **UNVERIFIED — highest-value item to check, if real.** This is the one
  candidate in the whole document that, if it actually works as described, directly targets
  our one confirmed unresolved bug. But: (a) I have no confirmation Depth Pro is in
  `qai_hub_models` at all — it's an Apple research release, not a Qualcomm zoo staple; (b) the
  claim that metric depth alone "solves" staircase inversion is a bit too clean — inversion
  on affine-invariant models happens because of ambiguous local geometry, and metric training
  data reduces but doesn't mathematically eliminate that ambiguity for a single frame. Worth
  checking zoo availability before spending more time on the claim itself.
- **Priority**: Verify zoo availability first. If absent, this becomes a "would need a from-
  scratch conversion," which is a different (much bigger) time commitment than "swap a model
  name."

### Metric3D, ZoeDepth, Marigold
- Mentioned as comparison points in the research's own citations, not recommended as
  integration targets (Metric3D flagged as "heavy memory," others just background).
- **MY ASSESSMENT**: Not evaluated further — research itself doesn't push these as
  actionable, and I have no independent reason to look harder at them under time pressure.
- **Priority**: Skip for now.

### Temporal depth fusion (EMA/Kalman + sparse Lucas-Kanade corner tracking)
- **What research says**: Track Shi-Tomasi/FAST corners frame-to-frame on CPU, use the
  resulting homography to warp/align consecutive depth maps, then smooth with a Kalman
  filter or EMA — reduces frame-to-frame depth jitter that causes false-positive drop-off
  triggers.
- **MY ASSESSMENT**: Architecturally sound and CHEAP (pure classical CV, no new model, no
  NPU/export risk at all). This is exactly the kind of "near-zero compute, real payoff" item
  we should trust more than the exotic model swaps, precisely because it doesn't depend on
  any unverified external claim — it's just standard, well-understood signal processing we
  can build and test ourselves offline first.
- **Priority**: Reasonable to prototype, moderate effort (new code, not a model swap).

### Stereo-from-Motion / multi-frame SfM
- Research itself rejects this as too heavy for real-time mobile (bundle adjustment memory
  cost). I agree — this is genuinely a different class of system.
- **Priority**: Skip.

---

## 3. Classical CV / Signal Processing (near-zero compute cost)

### Scharr operator (vs. our current Sobel)
- **What research says**: Better rotational symmetry than Sobel, catches diagonal
  drop-offs/curbs approached at an angle with less noise amplification.
- **MY ASSESSMENT**: This is a genuinely trivial, low-risk change — literally swapping the
  3×3 kernel weights in `DropOffDetector.kt`'s gradient step, which we JUST built and
  validated this session. Easy to test offline first with the same `debug_dropoff_v2.py`
  workflow before touching Kotlin. No new dependency, no export risk.
- **Priority**: **Do now** — cheapest, safest, most directly testable item in the whole doc.

### Laplacian of Gaussian (LoG)
- Mentioned as an alternative edge operator, no strong claim of superiority over Scharr for
  our specific use case.
- **MY ASSESSMENT**: Not obviously better than Scharr for this; would need the same offline
  A/B test to justify over Scharr. Not worth chasing both at once.
- **Priority**: Skip unless Scharr underperforms in testing.

### RANSAC ground-plane fitting
- **What research says**: Back-project the depth map to a 3D point cloud, fit a floor plane
  with RANSAC, flag any cluster that drops meaningfully below that plane as a negative
  obstacle — a more principled replacement for our whole `DropOffDetector` approach.
- **MY ASSESSMENT**: This is real, well-established robotics technique (ground-plane RANSAC
  is textbook). It's a genuinely BETTER long-term design than our current gradient+sign-check
  heuristic — but it's a bigger rewrite (3D back-projection needs camera intrinsics we don't
  currently model, plus RANSAC iteration cost per frame). Not a "swap a threshold" change.
- **Priority**: Consider for a future session, not this week — too much new surface area to
  validate before the event given our current budget.

### Lucas-Kanade sparse optical flow (collision trajectory)
- **What research says**: Track only YOLO bounding-box centroids frame-to-frame; compare
  each box's flow vector against the ego-motion field (radial expansion) to distinguish "I'm
  walking toward a static object" from "something is moving toward me independently."
- **MY ASSESSMENT**: Clever, cheap (tracks only a handful of points, not a dense field), and
  directly upgrades our existing (currently coarse) `MotionTracker`'s `approaching` field.
  Doesn't need a new model. Real implementation effort though — needs an actual ego-motion
  estimate, which without IMU fusion (see VINS-Mono below) is itself approximate.
- **Priority**: Reasonable future item; meaningfully more code than the Scharr swap.

### Hough transform
- From my original research prompt, folded into "line/plane-fitting" in the research answer;
  not elaborated separately.
- **MY ASSESSMENT**: Same bucket as RANSAC — useful for stair-edge/curb-line detection, not
  independently urgent beyond what RANSAC/Scharr already covers.
- **Priority**: Skip for now, revisit only if RANSAC plane-fitting gets built later.

---

## 4. Segmentation

### SAM / FastSAM / MobileSAM
- **What research says**: Explicitly RECOMMENDS AGAINST these for our use case — optimized
  for interactive/prompted segmentation, not continuous real-time scene parsing.
- **MY ASSESSMENT**: Agree with the research's own conclusion. Matches my own earlier
  assessment of `fastsam_x` (already in `models.json` as "optional," never prioritized).
- **Priority**: Skip — research and I agree.

### PIDNet
- **What research says**: Three-branch (detail/context/boundary) architecture from the
  autonomous-driving literature, fast real-time drivable-area segmentation.
- **MY ASSESSMENT**: **UNVERIFIED for our zoo.** PIDNet is a real, published architecture
  (I recognize the name from AD literature), but I have no confirmation it's available in
  `qai_hub_models` or has any validated mobile export path. This is a genuinely new
  subsystem (continuous free-space segmentation), not a swap — real UX payoff (a "safe path"
  audio texture) but real scope.
- **Priority**: Interesting, but too much new scope for this week. Future session.

### RepViT
- **What research says**: ViT-design-principles-in-a-CNN, exports cleanly to TFLite/ONNX,
  good for the same walkable-path task.
- **MY ASSESSMENT**: Same bucket as PIDNet — unverified against our actual zoo, and either
  way this is a new subsystem, not a tweak.
- **Priority**: Future session, pair with PIDNet evaluation if pursued.

---

## 5. Audio Classification & Hazard Localization

### YAMNet — already scoped, never built
- Already sitting in `models.json` as ticket `#33`, optional, never implemented.

### AST (Audio Spectrogram Transformer)
- Research itself rejects this for edge use (>200ms latency, 86M params). Agree.
- **Priority**: Skip.

### EfficientAT
- **What research says**: MobileNetV3-based distillation of AST, ~12ms inference, much
  better AudioSet mAP than YAMNet at similar/smaller size.
- **MY ASSESSMENT**: Plausible-sounding (knowledge distillation from AST teachers into
  MobileNet students is a real, published technique), but **we haven't even built the YAMNet
  baseline yet.** This would be optimizing a feature that doesn't exist in the app. Also
  unverified against our zoo.
- **Priority**: Skip until `#33` (basic hazard-sound detection) is actually built first.

### GCC-PHAT (sound source localization via TDOA)
- **What research says**: Cross-correlate the two-mic signals with phase-transform weighting
  to get a precise time-delay-of-arrival, giving hazard DIRECTION not just presence.
- **MY ASSESSMENT**: Real, well-established DSP technique (GCC-PHAT is textbook acoustic
  localization), and it's genuinely cheap (just FFTs). But it depends on a specific dual-mic
  physical geometry we haven't verified on the target phone, AND depends on the (unbuilt)
  hazard-sound classifier existing first.
- **Priority**: Skip until `#33` exists; revisit as a `#33` enhancement later.

### SALSA-Lite / SALSA-Mel features
- Mentioned as a way to fuse phase-difference features directly into the audio classifier's
  input for joint classify+localize.
- **MY ASSESSMENT**: Same dependency chain as above — not actionable before the base feature
  exists.
- **Priority**: Skip for now.

---

## 6. Speech Recognition & Open-Vocabulary Grounding

### Whisper-tiny — current baseline
- Already in `models.json`, name-corrected (`whisper_tiny` not `whisper_tiny_en`), already
  scoped as `#10/#26`, blocked only on the native QNN bridge (not on the model choice).

### Moonshine
- **What research says**: Better latency/accuracy ratio than Whisper for short command
  phrases specifically.
- **MY ASSESSMENT**: Possible, but swapping the ASR model adds NEW export/validation risk to
  a ticket that's already blocked on something else entirely (the native bridge). Fixing the
  bridge unblocks Whisper immediately; swapping to Moonshine first would just add a second
  unknown on top of the first.
- **Priority**: Skip — not the bottleneck. The bridge is the bottleneck.

### Levenshtein fuzzy matching for ASR-to-target mapping
- **What research says**: Compare transcribed text against a cached target dictionary with
  edit-distance fuzzy matching, to tolerate ASR typos/homophones.
- **MY ASSESSMENT**: This is a genuinely cheap, sensible, LOW-RISK piece of glue code — no
  model, no export, just a standard string-matching algorithm. Worth doing whenever the voice
  pipeline actually goes live (i.e., after the native bridge lands), since it's basically
  free robustness.
- **Priority**: Cheap future addition, gate it on the bridge landing first (no ASR = nothing
  to fuzzy-match yet).

### Bounding-box-area disambiguation ("nearest instance wins")
- **What research says**: When multiple instances of the requested class exist, default to
  the largest bounding box (closest proximity) as the goal.
- **MY ASSESSMENT**: Sensible, simple, already broadly consistent with how
  `VectorToGoalController` is likely to need to behave. Free to add once voice goal-seeking
  is actually live.
- **Priority**: Cheap future addition, same gate as above.

---

## 7. Sensor Fusion

### VINS-Mono (Visual-Inertial Odometry)
- **What research says**: Tightly-coupled camera+IMU state estimation with IMU
  pre-integration, stabilizes vision against body sway, gives metric-scale ego-motion.
- **MY ASSESSMENT**: This is a REAL, well-known academic SLAM system (VINS-Mono is a
  legitimate, widely-cited HKUST project) — but it's a serious, multi-week systems
  integration even for an experienced robotics team, not a mobile-app feature you bolt on.
  The research doc somewhat undersells this by presenting it in the same breath as one-line
  kernel swaps.
- **Priority**: Skip entirely for this event. Correct long-term architecture, wrong
  timeline.

### Barometer-assisted stair-descent confirmation
- **What research says**: ~0.12 hPa pressure change per 1m of altitude change; cross-check a
  detected downward depth gradient against a real barometer pressure trend to confirm actual
  descent, resolving depth-model ambiguity.
- **MY ASSESSMENT**: **This is the single best idea in the entire document.** It's cheap
  (Android's `Sensor.TYPE_PRESSURE` is a standard, already-present sensor on most phones,
  no permission beyond normal sensor access), needs no new model, no export, no new
  dependency — and it directly targets our ONE CONFIRMED, STILL-OPEN BUG (`stairs2.jpeg`'s
  depth model reading the wrong sign). The exact hPa-per-meter constant is a standard
  atmospheric physics figure, easy to verify independently rather than trust blindly.
- **Priority**: **Do now**, alongside the Scharr swap — highest value-to-effort ratio of
  anything in this document, and it's real physics, not a marketing claim about a model we
  haven't tested.

---

## 8. Confidence Calibration & Uncertainty

### Conformal Prediction / Split Conformal Prediction (SCP)
### Sequential Conformal Risk Control (SeqCRC)
- **What research says**: A statistically rigorous, distribution-free way to turn raw
  detector scores into calibrated prediction sets with formal coverage guarantees — proposed
  as the mathematical foundation for the WHITE/BLUE/RED tier system.
- **MY ASSESSMENT**: This is legitimate, real statistical methodology (conformal prediction
  is an active, respected ML research area) and would genuinely be the "correct" way to
  build a trustworthy confidence-tier system. But it requires a labeled CALIBRATION DATASET
  (the research suggests ~500 representative images) that we do not have, do not have time
  to collect+label before the event, and building the calibration pipeline itself is real
  engineering work on top of that. This is the most over-scoped recommendation in the
  document relative to our actual remaining time.
- **Priority**: Skip for this event. If you want SOME calibration benefit cheaply: manually
  eyeball a handful of `debug_yolo.py` runs and pick a slightly better threshold constant by
  hand (we already did exactly this earlier this session, moving `confThreshold` from 0.35
  to 0.30 based on real photo evidence) — gets ~80% of the practical benefit in minutes, not
  days.

### Temperature scaling / Platt scaling
- Mentioned generally as lighter-weight calibration alternatives to full conformal
  prediction.
- **MY ASSESSMENT**: More tractable than full conformal prediction, but still needs a
  validation set and a fitting step we haven't budgeted time for. The manual-threshold
  approach we already did this session is the pragmatic version of this.
- **Priority**: Skip — we already did the practical equivalent by hand.

### Test-Time Augmentation (TTA)
- Research itself rejects this (multiplies inference cost, breaks real-time budget). Agree.
- **Priority**: Skip.

---

## 9. On-Device / NPU Optimization

### Subgraph fragmentation (NPU/CPU graph splitting on unsupported ops)
- **What research says**: LayerNorm/Softmax/GELU-type ops get rejected by the Hexagon NPU
  compiler, forcing a CPU/NPU graph split with heavy DMA overhead.
- **MY ASSESSMENT**: This description is consistent with (though not identical to) the exact
  class of bug we found ourselves this session — NNAPI silently degrading `yolo26s`'s
  outputs rather than cleanly falling back. Different failure mode (silent corruption vs.
  described latency fragmentation), but same underlying lesson: don't trust a delegate to
  handle unusual ops gracefully, TEST it. We already learned this the hard way; the research
  doc's warning is a useful confirmation, not new information for us.
- **Priority**: Already internalized from this session's own debugging. No new action, but
  validates our "test everything offline first" workflow going forward.

### W8A16 mixed-precision quantization
- **What research says**: 8-bit weights + 16-bit activations, ~1.4% accuracy loss vs. ~8%
  for full INT8, natively accelerated on Hexagon.
- **MY ASSESSMENT**: Attempted this session. Two real findings: (1) **W8A16 is QNN-only,
  NOT TFLite-compatible** — confirmed via the export CLI's own error message. Since we don't
  have the native QNN bridge built yet, we couldn't run a W8A16 model on-device even if we
  had one — this whole item is genuinely blocked on the bridge landing first, not just a
  nice-to-have alongside it. (2) Getting even the CLOUD PROFILING numbers (without running
  on-device) hit a real dependency chain: quantization needs calibration data, which pulls in
  the full COCO val2017 dataset (800MB+) plus a chain of missing packages one at a time
  (`omegaconf` → `safetensors` → `aiofiles` → `pycocotools`, discovered only by retrying
  after each failure). Stopped after the 4th missing dependency — not one-flag-away as
  originally assessed, this needs a proper one-time environment setup pass.
- **Priority**: **Deferred, not skipped.** Revisit once the QNN native bridge exists (so
  there's actually somewhere to run the result) — at that point, first do
  `pip install pycocotools` (last confirmed blocker) and retry the export; the COCO
  calibration data should already be cached locally from this session's attempt.

### INT8 (plain)
- Research itself warns this causes ~8% accuracy collapse for our model classes. Consistent
  with general quantization literature. Not recommending it.
- **Priority**: Skip — use W8A16 instead if quantizing at all.

### AOT compilation via Qualcomm AI Hub
- Not new information — this is literally the exact pipeline (`qai_hub_models` cloud
  compile) we already used this session to produce `yolo26n`/`yolo26s`.
- **Priority**: Already our standard workflow. No new action.

---

## 10. Prior Art / Existing Assistive Navigation Systems

- **Microsoft Soundscape** — GPS+cloud POI, no real-time hazard sensing. Validates our
  stereo-pan design choice, confirms GPS-only isn't sufficient (which we already knew).
- **OrCam MyEye / Envision Glasses** — cloud-dependent, discrete query-response, not
  continuous — exactly the gap SecondSense's offline+continuous design is meant to fill.
- **Biped** — active IR/ToF sensing, accurate but expensive/bulky hardware; validates why
  we chose passive monocular RGB instead (accessibility via a commodity phone).
- **MY ASSESSMENT**: Plausible and consistent with general public knowledge about these
  products; I have not independently fact-checked the specific technical claims about each
  (e.g., Biped's exact sensor suite), but none of this changes our roadmap — it's
  contextual validation, not a new engineering input.
- **Priority**: Informational only. Useful for your pitch narrative (`#43`), not
  actionable engineering.

---

## Summary — Prioritized Action Table

| Item | New model/export needed? | Verified real? | Effort | Priority |
|---|---|---|---|---|
| **Barometer stair-descent cross-check** | No | Yes (standard physics/sensor) | Low | **Do now** |
| **Scharr operator swap** | No | Yes (standard DSP) | Very low | **Do now** |
| **W8A16 quantization test** | No (existing pipeline) | Yes (our own export CLI) | Low | Try next |
| Temporal depth EMA/Kalman fusion | No | Yes (standard technique) | Medium | Good next session |
| YOLOE-26 (open-vocab) | Maybe | **Unverified in our zoo** | Unknown | Verify first |
| Depth Pro (staircase fix) | Yes, unclear if in zoo | **Unverified in our zoo** | Unknown | Verify first |
| Lucas-Kanade collision trajectory | No | Yes (standard technique) | Medium | Future session |
| RANSAC ground-plane fitting | No | Yes (standard technique) | Medium-high | Future session |
| PIDNet / RepViT segmentation | Yes | **Unverified in our zoo** | High | Future session |
| EfficientAT / GCC-PHAT audio | Yes | Plausible, unverified | High | After #33 baseline exists |
| Moonshine ASR swap | Yes | Plausible, unverified | Medium | Skip — not the bottleneck |
| VINS-Mono (full VIO) | N/A (algorithm, not model) | Yes (real system) | Very high | Skip — wrong timeline |
| Conformal prediction calibration | No | Yes (real methodology) | High (needs dataset) | Skip — over-scoped for event |

**Bottom line**: two items (barometer cross-check, Scharr swap) are cheap, real, and directly
useful — start there. Everything requiring a new model needs a zoo-availability check before
any time is spent on it, because this document's specificity outran what I've actually
verified.
