SecondSense — Research Candidates v2: Domain Adaptation, Fine-Tuning, Negative-Obstacle Detection
====================================================================================================

Companion to `secondsense_research_candidates_v1.md` (same format, same discipline). Source:
a second deep-research pass specifically on "how do real assistive-navigation products/
research groups actually increase accuracy beyond a generic pretrained checkpoint" — domain-
specific datasets, fine-tuning, hardware quantization pitfalls, and negative-obstacle
detection methods.

**Same rule as v1: every "MY ASSESSMENT" is my own read, not a re-statement of the research
doc's confidence.** This report is far more specific and citation-heavy than the first —
which makes it MORE useful if the citations hold up, and MORE risky to act on blindly if
they don't. Several claims below (especially "downloadable pre-trained checkpoints exist")
have NOT been independently verified yet — that verification is the explicit first next step
after this document, not assumed true here.

---

## 1. The domain-shift problem itself

**What research says**: COCO-trained models lack the taxonomy (negative obstacles, tactile
paving, curb ramps) and viewpoint alignment (chest/waist-mounted vs. eye-level) that BLV
navigation actually needs. Real systems (SightWalk) reframe problems as classification
("drift left/right/on-path") instead of raw detection to cut compute.

**MY ASSESSMENT**: This framing is correct and consistent with what we found ourselves this
session — `yolo26s` is a generic COCO detector, and its confidence genuinely was weaker on
our own real, non-COCO-typical test photos (§1 of v1). The "classification instead of
detection" reframing (SightWalk's sidewalk-drift approach) is a legitimate compute-saving
idea, but it's a different KIND of feature (lane-keeping guidance) than anything currently
built — not directly actionable as a swap into the existing pipeline.
**Priority**: Informational — confirms a problem we already independently found, doesn't
change our roadmap by itself.

---

## 2. Domain-specific datasets

| Dataset | Claimed size/format | Claimed focus | MY ASSESSMENT |
|---|---|---|---|
| **Project Sidewalk** | 300k+ images, Street View crops | Curb ramps, missing ramps, surface problems, includes "null crops" for negative sampling | **UNVERIFIED.** Plausible — Project Sidewalk is a real, known academic crowdsourcing initiative (University of Washington). Whether it's actually packaged as a ready-to-use detection training set (vs. crowdsourced accessibility survey data in a different format) needs checking before assuming it's YOLO-ready. |
| **OD (Obstacle Dataset)** | 7,900 images, 40k boxes, 15 classes | Blind-sidewalk obstacles | **UNVERIFIED, HIGH PRIORITY TO CHECK** — claimed to be "pre-formatted for YOLO/SSD," which is the most actionable claim in this whole document if true. |
| **VIDVIP** | 32k images, 540k instances, 39 classes | Japanese cities, tactile paving/signal buttons | Plausible but geography-specific (Japan) — tactile paving standards and street furniture differ from India; would need domain re-adaptation even if real. |
| **SideSeeing** | 325k video frames + IMU | Multi-modal, hospital-area sidewalks (Brazil/USA) | Interesting for sensor fusion research, but multi-modal (needs synchronized IMU) — more complex to use than a plain image dataset. |
| **BlindWays** | 1,029 clips, 3D human poses | BLV pedestrian motion | Research doc itself flags this as suited for trajectory forecasting, not object detection — not directly relevant to our pipeline. |
| **Mendeley Indian road obstacles** | 38,295 images | Animals, auto-rickshaws, irregular crosswalks, tractors — India-specific | **UNVERIFIED**, but if real, directly relevant to Bengaluru streetscape (the actual demo environment). Worth checking. |
| **Roboflow "Indian Road Obstacles"** | 500+ images | India-specific, YOLO-formatted | **UNVERIFIED**, smaller and more immediately usable if real (Roboflow datasets are typically already YOLO-ready by platform convention). |

**Overall assessment**: None of these are verified to exist/be accessible in the form
claimed. Roboflow-hosted datasets are the easiest category to verify FAST (Roboflow has a
public, browsable catalog) — that's the natural first check.
**Priority**: Verify Roboflow-hosted ones first (fastest to check), then Project
Sidewalk/OD Dataset/Mendeley.

---

## 3. Downloadable pre-fine-tuned checkpoints — THE highest-value claim to verify

**What research says**: the OD Dataset's authors published trained YOLO/SSD weights directly
on GitHub; Roboflow Universe hosts community-contributed YOLOv8/v11 checkpoints for curb
detection, pothole/manhole detection ("Pothole_Detection", "NEEVD").

**MY ASSESSMENT (at the time this was written, before verification)**: this looked like the
single most consequential claim in the report — a pre-trained checkpoint needs ZERO training,
a direct model swap testable with our existing offline workflow, bypassing the "no GPU"
blocker entirely.

### VERIFIED — full result, both sub-claims checked directly

**Why we checked this before anything else**: "a GitHub repo with YOLO weights exists" or "a
Roboflow project has a downloadable model" are exactly the kind of specific-sounding claims
that turn out overstated in research summaries — a training config mistaken for actual
weights, a broken link, or (as happened here) real weights that turn out to be structurally
unusable for our specific offline architecture. Verifying before building on top of a claim
is the same discipline applied to every other item this session.

**1. OD Dataset's GitHub weights — FALSE.** Fetched
[TW0521/Obstacle-Dataset](https://github.com/TW0521/Obstacle-Dataset) directly. Confirmed the
LABELED DATASET is real (7,915 images, VOC + YOLO annotation formats, 15 real obstacle
classes — matches the claimed numbers almost exactly). But **no trained weight files**
(`.pt`/`.h5`/`.onnx`) exist anywhere in the repo, despite the report's specific claim that
"detection models can be obtained" there. Data-labeling is solved if we ever wanted to
fine-tune (with an external GPU); "download and use immediately" is not true for this one.

**2. Roboflow-hosted checkpoints — real models exist, but structurally unusable for us
as-is.** Searched Roboflow Universe's actual catalog and confirmed genuine, high-quality
trained models exist, including **`YOLO_stairs`** (216 images, `stairs` class,
**85.6% mAP@50, 91.6% precision** — directly relevant to our own confirmed staircase-depth
bug) and a `curb` detector. But: every project exposes its model **only through Roboflow's
hosted cloud inference API** (an `API_KEY`-gated endpoint), not a visible "download weights"
button on the public page. Confirmed via Roboflow's own documentation
([docs.roboflow.com](https://docs.roboflow.com/deploy/download-roboflow-model-weights)) that
direct weight export (PyTorch/ONNX/**TFLite**/etc.) is a feature of **paid Basic/Growth
plans** — free-tier accounts don't get a direct download button; the documented free-tier
path is forking the dataset and training your own version instead.

**Why this matters architecturally**: even if weights were freely downloadable, the DEFAULT
access path (hosted cloud inference) is an ONLINE dependency — directly contradicts the app's
offline requirement (Bible §16). The only usable path for us is a genuinely local export
file, which needs either a paid plan or the fork-and-train-your-own-free-version route.

### How to get the weights yourself (steps, for whenever this gets picked back up)
1. Create a free Roboflow account at `app.roboflow.com` (a real account, standard email/
   Google signup — not something to do via automation, a real person needs to do this).
2. Go to the PUBLIC project page (not your own workspace):
   `https://universe.roboflow.com/hohyoo/yolo_stairs` (or `.../curbstone/curb-zjoev`).
3. Click **"Fork Dataset"** — copies it into your own workspace now that you're logged in.
4. Back in your workspace (`app.roboflow.com/<your-username>/projects`), the forked project
   should now appear.
5. Inside it: **Versions → Generate New Version** (required even to reuse the same
   images/labels as-is).
6. Free accounts include a monthly training-credit allowance (observed: 15 credits/month on
   a fresh account) — use it to train, then check the resulting version for an
   export/download option. **Select TFLite if offered** (matches our pipeline exactly,
   no conversion needed); PyTorch/ONNX also work, convertible via our existing tooling.
7. Hand off whatever file results — at that point this becomes a normal offline-test-first
   verification (`debug_yolo.py`-style) before any Kotlin integration, same as every other
   model this session.

**PAUSED, not abandoned** — the user is running this signup/fork flow themselves; picking
this back up requires a resulting downloaded file, not further engineering work right now.
**Priority**: On hold pending the user's own Roboflow export attempt. Session moved on to
building V-disparity + RANSAC (§8) instead, which needs no external account.

---

## 4. Real commercial/academic systems — what they reportedly do

- **OrCam MyEye**: task-specific inference cascades (dedicated OCR model, few-shot face
  metric learning), not one general detector. Plausible architecture pattern, unverified
  specifics.
- **Envision Glasses**: Google MLKit + AR spatial anchors + LLM integration for
  conversational queries. Plausible, matches publicly known product positioning.
- **Biped NOA**: prioritizes objects by intersection with the user's PROJECTED PATH, not
  raw presence in frame — a genuinely different targeting philosophy than our current
  "nearest/most-central" `TargetSelector`. Interesting idea, real scope to adopt (would
  need trajectory prediction, not built).
- **WeWalk / Sunu Band**: ultrasonic-only, no semantic classification — cited as the
  contrast case for why vision-based classification matters. Consistent with our own
  Bible's stated design rationale.
- **SightWalk (academic, open-source)**: Jetson Xavier + waist camera, sidewalk-drift
  classification instead of full segmentation. Real, citable academic project.
- **MY ASSESSMENT overall**: These are architecture/philosophy descriptions, not something
  directly portable into our codebase — useful context for the pitch narrative (`#43`), not
  an engineering action item. I have not independently verified any of these specific
  technical claims against primary sources.
**Priority**: Pitch-narrative material only, no engineering action.

---

## 5. NPU quantization: SiLU → ReLU6 replacement

**What research says**: YOLO's SiLU activation is unbounded on the positive axis, which maps
poorly into INT8's 256 discrete bins, causing the 7-15% mAP drops commonly reported on INT8
YOLO quantization. Replacing SiLU with ReLU6 (bounded output, max 6) fixes this with near-zero
quantization loss. Also recommends decoupling detection heads (keep box-regression layers in
FP16/FP32 even if the backbone is INT8) and "degradation-aware calibration" (mixing blurred/
noisy images into the calibration set).

**MY ASSESSMENT**: The underlying problem description (SiLU's unbounded range hurting INT8
quantization) is a real, well-known issue in the quantization literature — plausible on its
face. However: **this is architecture surgery on the model itself** (swapping activation
functions means re-exporting from source, not a flag on our existing `qai_hub_models` export
CLI) — meaningfully more invasive than anything else we've done this session. Also directly
blocked by the same wall we already hit: W8A16 export itself is stuck on a dependency chain
(`pycocotools`, deferred in v1 §9), so there's nowhere to apply this technique yet even if we
wanted to.
**Priority**: Remember for later, not actionable now — blocked on the same W8A16/QNN bridge
prerequisite already documented.

---

## 6. Depth Anything V2 mobile precision

**What research says**: dense regression tasks like depth estimation are more sensitive to
INT8 quantization than classification; recommends FP16 or mixed precision, not full INT8, for
depth models specifically.

**MY ASSESSMENT**: Consistent with our own existing setup — we're already running
`depth_anything_v2.tflite` as a float export (never attempted INT8 quantization on it this
session), so we're already doing what this recommends by default, not by explicit design
choice. No action needed; useful confirmation that we haven't been doing something wrong.
**Priority**: Already aligned, no action.

---

## 7. Synthetic data generation (Unreal Engine + NDDS)

**What research says**: UMass Amherst's RITA lab generated tens of thousands of synthetic
training images (tactile paving, curb ramps, fire hydrants) via Unreal Engine + NVIDIA's NDDS
plugin, with pixel-perfect auto-generated labels; fine-tuning YOLOv8m on just 3,000 synthetic
images improved detection of sidewalk features from novel viewpoints.

**MY ASSESSMENT**: Plausible and describes a real, known synthetic-data technique (NDDS is a
real NVIDIA tool). But this requires: Unreal Engine expertise, 3D asset creation/sourcing, a
render pipeline, and — same as every fine-tuning path — a GPU to actually train on. This is
real infrastructure we don't have, not a quick add.
**Priority**: Out of scope for this event, real technique for a future/funded phase.

---

## 8. Negative-obstacle / curb detection alternatives — directly relevant to our own #17 work

### V-disparity + RANSAC
**What research says**: accumulate a depth/disparity map's rows horizontally into a
"V-disparity image" — a flat ground plane always projects as a straight diagonal line in that
space. Fit that line with RANSAC; anything deviating above or below the line is an obstacle
or drop-off. Cheap (CPU-only), fast, doesn't need a new model.

**MY ASSESSMENT**: **This is the most directly actionable, well-justified item in the whole
document.** It's the SAME underlying idea as "RANSAC ground-plane fitting," which v1 §3
already flagged as a legitimate future improvement to `DropOffDetector` but left too vague to
act on ("needs camera intrinsics we don't currently model"). V-disparity is a more precise,
proven, well-established version of that same idea — and notably does NOT need real 3D
back-projection/camera intrinsics the way a full point-cloud RANSAC would (that was the exact
blocker v1 flagged); it works directly on the 2D depth map by accumulating rows, which is
something we can compute directly from `DepthSampler.Frame` with no new geometric modeling.
This is a genuine, meaningful upgrade path for our own already-built #17 detector, using our
own already-established offline-validate-before-Kotlin workflow.
**Priority**: **Second thing to build**, after verifying item #3 (checkpoints).

### Surface normal prediction (NDDepth)
**What research says**: predict per-pixel surface normals (not just depth) so a staircase's
horizontal treads and vertical risers show mathematically distinct, orthogonal normal
vectors — physically prevents depth from "smoothing over" a staircase into a ramp illusion.

**MY ASSESSMENT**: Elegant and well-motivated (matches the exact staircase-inversion bug we
found and left unresolved this session), but requires a DIFFERENT depth model output entirely
(NDDepth-style networks output surface normals, `depth_anything_v2` does not) — this is a
model swap/retrain, not a post-processing addition like V-disparity. Real future direction,
not this week.
**Priority**: Future session — would need a new model, unlike V-disparity.

### CurbNet / RampNet (point-cloud-based)
**What research says**: LiDAR/point-cloud curb detection with attention modules and
polynomial curve fitting.

**MY ASSESSMENT**: We have no LiDAR — this is explicitly designed for a different sensor
modality than our monocular-camera-only phone setup. Not applicable.
**Priority**: Skip — wrong sensor class for our hardware.

---

## 9. Active learning / continuous improvement pipeline

**What research says**: shadow-mode disagreement triggers (heuristic vs. NN disagreement →
flag for review), consent-gated edge-cropping (privacy-preserving partial uploads), cloud
auto-labeling with large VLMs, periodic OTA model updates.

**MY ASSESSMENT**: A legitimate, real production ML pattern — but requires a real user base,
a cloud backend, a labeling pipeline, and consent/privacy infrastructure, none of which exist
for a hackathon prototype. Good to know the shape of "what comes after the hackathon," not
actionable now.
**Priority**: Post-event roadmap item only.

---

## Summary — Prioritized Action Table

| Item | Needs GPU/training? | Verified? | Effort | Priority |
|---|---|---|---|---|
| **Downloadable pre-trained checkpoints (OD Dataset, Roboflow)** | No | **Unverified — check first** | Low if real | **Verify NOW** |
| **V-disparity + RANSAC drop-off detector** | No | N/A (technique, not a claim to verify) | Medium | **Build next**, offline-validated first |
| Roboflow-hosted Indian-road datasets | Yes (to fine-tune) | Unverified | High | Check availability, defer training |
| Project Sidewalk / OD Dataset / Mendeley datasets | Yes (to fine-tune) | Unverified | High | Defer — needs external GPU |
| SiLU→ReLU6 quantization fix | No (but needs re-export) | Plausible, real technique | Medium-high | Blocked on W8A16/QNN bridge anyway |
| Surface normal prediction (NDDepth) | Yes (different model) | Real technique, unverified fit for us | High | Future session |
| Synthetic data via Unreal Engine/NDDS | Yes | Real technique | Very high | Out of scope for this event |
| CurbNet/RampNet | N/A | Wrong sensor modality | N/A | Skip |
| Active learning pipeline | N/A (needs infra) | Real pattern | Very high | Post-event roadmap |

**Bottom line**: two items are genuinely actionable this session — verify the downloadable
checkpoints (cheap, potentially high payoff, zero training needed), then build V-disparity +
RANSAC as a real upgrade to our own #17 drop-off detector, following the same offline-first
validation discipline as everything else built this session.
