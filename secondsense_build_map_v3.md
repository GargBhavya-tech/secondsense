# SecondSense — Detailed Build Map & Step-by-Step Implementation Guide (v3)

**For anyone building SecondSense who has NOT read the full Bible.** This file is self-contained and ordered so each ticket can be built and tested on its own before moving to the next. The companion `secondsense_bible_v4.md` has the deep "why" — read it if a step's rationale is unclear.

**Build model:** 3-person team, single **30-hour sprint** — the **iQOO Hackathon 2026 · Bengaluru City Battle · Aug 29–30**. This is a real sprint, not an open-ended runway, and the sprint is **phase-gated**: the organizers toggle between Red Light (phone-only) and Green Light (both devices) on *their* schedule, and there are **two graded checkpoints along the way** (Eval R1 ≈ hour 8, Eval R2 ≈ hour 22). Build order targets **demo-ready by hour 8**, then iterates — not "build toward one ending."

This map is granular on purpose: bigger pieces are split so each ticket is buildable and independently testable in a single sitting, and every ticket has an explicit **Test** step, not just a "done when" description.

---

## How to Read This Map

Tickets are numbered in strict dependency order — `Blocked by` only ever points to a lower ticket number, so the logical build sequence never leaves you stuck on something undone.

### Priority tag

| Tag | Meaning |
|---|---|
| 🟥 **CORE** | The spine, or a headline differentiator. If you build nothing else, build these. |
| 🟨 **SUPPORTING** | Real, but shown as one cue / pane / mode. Build after the spine works end-to-end. |
| ⬜ **SEED / PITCH-ONLY** | Build a *token* version only, or narrate it on stage — the real version is future work, not a hackathon target. |

### Light tag (which venue window a ticket can be done in)

| Tag | Meaning |
|---|---|
| ⚪ **PRE-EVENT** | Must be decided/sourced/rehearsed before doors open. Doing this on the clock is a wasted hour. |
| 🟢 **GREEN** | Needs the laptop — model conversion/quantization, big installs, dashboard polish. Only doable in a Green Light window. |
| 🔴 **RED-OK** | Phone-only is enough. Fine to build during Red Light; keep the Office Kit screen-mirror running while you do (that usage is scored). |

### Per-ticket fields

Each ticket has: **Build** (what to actually make), **Watch out** (the specific mistake the Bible already flagged for this piece), **Test** (how to verify it works, standalone), **Done when** (the pass condition). Plus a **Checkpoint** line: the latest eval it must be ready for.

**Golden rule:** don't start a ticket until its blockers show green on their own Test step. A ticket that "mostly works" is not done — the next ticket silently inherits its bugs.

**Scheduling rule (SecondSense-specific):** because Red/Green is toggled by the organizers, you can't always take the next number in sequence. **Always keep one Red-ready ticket and one Green-ready ticket queued**, split across the three people, so no light change ever leaves anyone idle. Get model conversion (Phase 1, all 🟢) done in your *first* Green window — nearly every on-device ticket is blocked on it.

---

## Phase 0 — Pre-Event Decisions & On-Device Foundations

### #1: Ear-Occlusion / Headphone Decision
Priority: 🟥 CORE
Type: Discuss
Light: ⚪ PRE-EVENT
Checkpoint: Before doors open
Blocked by: —

**Build:** Pick and physically test one answer to the ear-occlusion problem: (A) wired bone-conduction, or (B) haptics-primary so audio load drops enough that open-ear buds are viable. Source the actual headset and the USB-C / USB-C→3.5mm path, and pre-test it on the demo phone.
**Watch out:** a blind person's primary navigation sense is hearing — do not blindly mandate fully-occluding wired headphones. Not having an answer on stage is worse than any specific answer.
**Test:** wear the chosen setup, walk a corridor, confirm you can still hear traffic/voices AND the app's cues at once.
**Done when:** one approach is chosen, sourced, and tested — and you can say the sentence "we thought about ear occlusion, here's our answer" out loud.

### #2: Submission Logistics Locked
Priority: 🟥 CORE
Type: Discuss
Light: ⚪ PRE-EVENT
Checkpoint: Before doors open
Blocked by: —

**Build:** Finalize four things and stop revisiting them: the name **SecondSense** across form/deck/app UI; the **track** (HealthTech, with the one-line "assistive tech is health & independence, not convenience" justification ready; Open Innovation as fallback); the **3-person role split** (§20); and student/professional bucket registration.
**Watch out:** registration caps teams at 3 — the old 4-role split must collapse to 3 (merge Systems/Bridge with Demo/Wow, since both touch Office Kit and choreography).
**Test:** each of the four items has a single written, agreed answer with no open "we'll decide later."
**Done when:** name, track, roles, and registration are settled in writing before you arrive.

### #3: Qualcomm AI Hub Toolchain Confirmed
Priority: 🟥 CORE
Type: Research
Light: 🟢 GREEN
Checkpoint: Before doors open (ideally)
Blocked by: —

**Build:** Install the `qai-hub-models` CLI on the laptop, authenticate, and confirm you can actually download and convert at least one target model to the phone's NPU format.
**Watch out:** discovering a broken download or an auth wall *during* a Green window is a catastrophic time sink — nearly all of Phase 1 depends on this working.
**Test:** run one full download → convert → deploy cycle for the smallest model end-to-end, before the event if possible.
**Done when:** you have a proven, repeatable command sequence that gets an NPU-native model onto the phone.

### #4: Office Kit Workflow Rehearsed
Priority: 🟥 CORE
Type: Prototype
Light: ⚪ PRE-EVENT
Checkpoint: Before doors open
Blocked by: —

**Build:** Practice the full Office Kit loop (screen mirror, remote control, clipboard, file transfer) between the demo phone and laptop until it's fast and reflexive.
**Watch out:** Office Kit usage is **10% of the score, measured from device telemetry** — and speed here is directly scored. This is not a day-of thing to figure out.
**Test:** time yourself mirroring the phone, pushing a file both directions, and driving the phone from the laptop — all under a minute, cold.
**Done when:** every team member can run the Office Kit loop without hesitation.

### #5: Mount + Demo Course Prepped
Priority: 🟥 CORE
Type: Prototype
Light: ⚪ PRE-EVENT
Checkpoint: Before doors open
Blocked by: —

**Build:** A lanyard/chest strap that holds the phone camera chest-forward and steady. Plus the physical demo kit: a dark chair on a light floor, tape for a "curb/drop-off" line, a marked "door," and a head-height obstacle rig.
**Watch out:** arm-swing from a handheld phone breaks depth mapping — mount, don't hold. The chest position is the whole cane-blind-spot argument; don't compromise it.
**Test:** wear the mount, walk 10 steps, confirm the camera stays forward and level without hand support.
**Done when:** the mount is stable hands-free and the high-contrast course elements exist physically.

### #6: On-Device App Skeleton + Camera Capture
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: —

**Build:** A minimal app on the iQOO that opens the rear camera, pulls a live frame stream, and has empty hooks for an audio-out channel and the haptic motor.
**Watch out:** the laptop is never a compute dependency for the running app — everything the app does at demo time runs on the phone alone.
**Test:** launch the app, confirm live frames arrive and a test tone + a test vibration both fire on a button tap.
**Done when:** camera-in, audio-out, and haptic-out all work on-device with no laptop attached.

### #7: One-Tap Calibration
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #6

**Build:** Hold the phone level once, tap to set the baseline "forward" and a proximity reference. Not full gyroscope auto-correction — one tap.
**Watch out:** monocular depth is relative, not metric (§#13) — calibration sets the *baseline* the relative-proximity ordering is measured against, which is exactly what you claim on stage.
**Test:** tap-calibrate, then confirm the same object reads as consistent "forward + baseline distance" across repeated taps.
**Done when:** a single tap reliably sets forward + proximity baseline.

---

## Phase 1 — Model Conversion (all Green Light, do these first in your first Green window)

### #8: YOLOv11 (INT8) → NPU
Priority: 🟥 CORE
Type: Research
Light: 🟢 GREEN
Checkpoint: Eval R1 (hr 8)
Blocked by: #3

**Build:** Download, quantize to INT8, convert, and deploy YOLOv11 to the Hexagon NPU for real-time object detection & classification.
**Watch out:** the vision pipeline is a commodity — anyone can wire up YOLO. It is *not* your differentiator; get it working and move on to Phase 3, which is.
**Test:** run inference on a still image on-device; confirm boxes + class labels return in roughly the expected ~20ms.
**Done when:** YOLOv11 runs NPU-native, offline, on the phone.

### #9: Depth-Anything-V2 (W4A8) → NPU
Priority: 🟥 CORE
Type: Research
Light: 🟢 GREEN
Checkpoint: Eval R1 (hr 8)
Blocked by: #3

**Build:** Quantize to W4A8, convert, and deploy Depth-Anything-V2 for monocular relative-distance estimation.
**Watch out:** this gives *relative/inferred* depth, not calibrated metres — the whole product's honesty (§#13, §5.4) depends on never overclaiming this.
**Test:** run on a still image on-device; confirm a relative depth map returns (near vs far ordering is correct) in roughly the expected ~5ms.
**Done when:** Depth-Anything-V2 runs NPU-native, offline, producing a usable relative-depth map.

### #10: Whisper-Tiny → NPU
Priority: 🟥 CORE
Type: Research
Light: 🟢 GREEN
Checkpoint: Eval R2 (hr 22)
Blocked by: #3

**Build:** Convert and deploy Whisper-Tiny for on-device speech recognition of short voice commands.
**Watch out:** this is half of the co-headline voice-search feature (§7) — it's not a stretch goal, but it can slip to R2. Don't let it slide past that.
**Test:** speak "find the door" on-device in airplane mode; confirm the transcript comes back correct, offline.
**Done when:** Whisper-Tiny transcribes short commands on-device with no network.

### #11: OWL-ViT → NPU
Priority: 🟥 CORE
Type: Research
Light: 🟢 GREEN
Checkpoint: Eval R2 (hr 22)
Blocked by: #3

**Build:** Convert and deploy OWL-ViT for open-vocabulary zero-shot object grounding (locate an arbitrary named thing in the frame).
**Watch out:** open-vocabulary is the moat — "vector toward anything I can say" is what describe-only incumbents (Seeing AI, Lookout) can't do. This model *is* the differentiator; protect its build time.
**Test:** feed a frame + the word "door" on-device; confirm it returns a bounding region for the door, offline.
**Done when:** OWL-ViT grounds an arbitrary spoken word to a screen region, NPU-native and offline.

---

## Phase 2 — Core Vision Pipeline (on-device)

### #12: YOLO Live Inference Loop
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #8, #6

**Build:** Run YOLOv11 continuously on the live camera stream, emitting detections (class + box) per frame.
**Test:** point the phone at the demo chair and a person; confirm both are detected live, labeled, at a stable frame rate.
**Done when:** a continuous stream of labeled detections comes off the live camera on-device.

### #13: Depth Live Inference Loop
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #9, #6

**Build:** Run Depth-Anything-V2 continuously on the live stream, producing a per-frame relative-depth map, and expose a "relative proximity of the thing in front of me" value + rate-of-approach.
**Watch out:** claim relative-proximity ordering and rate-of-approach against the #7 baseline — never absolute metres. That honest framing is what makes everything else believable.
**Test:** walk toward a wall; confirm the proximity value rises smoothly and rate-of-approach spikes as you close in.
**Done when:** live relative-proximity + rate-of-approach are available per frame.

### #14: Center-Crop Targeting
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #12

**Build:** In flow (walking) mode, only prioritize objects in the center ~30% of frame.
**Watch out:** processing the whole frame floods the user with cues for things they'll walk past — the center crop is what keeps flow mode sparse and urgent-only.
**Test:** place objects at the frame edges and one dead center; confirm only the center object is prioritized.
**Done when:** only centered objects drive cues during flow mode.

### #15: Closest-in-Center Priority + Static/Dynamic Split
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #14, #13

**Build:** Combine detection (#14) with proximity (#13) to pick the single closest centered target, capping simultaneous cues at 1–2. Add a coarse "is this box moving" flag (adapted from a prior project's separate tracked-object layer, Bible §13.2) and give a moving target priority over a static one at equal distance — a person walking toward the user should win over a parked chair at the same range.
**Watch out:** full Time-to-Collision physics is a rejected stretch goal (§11) — simple closest-in-center covers the common case; don't build velocity tracking beyond a cheap moving/static flag. **Hidden cost to budget for:** the flag still requires matching a box to the same box in the previous frame (lightweight box-to-box association across ~2 frames). This is the lightest possible form of tracking, but it is *not* free — budget it as real work, not a one-liner, and keep the association logic dumb (nearest-box-of-same-class between consecutive frames is enough).
**Test:** *(note — run this AFTER #16's temporal smoothing exists, or use a pre-recorded clip; a live hand-moved "approaching object" will flicker without smoothing and give a false failure, which is exactly the fast-motion weakness §11 declines to solve).* Using a slow, smooth approach (or a recorded clip): place two objects at the same distance dead center, one stationary and one slowly approaching; confirm the moving one wins priority. Separately confirm the original closest-wins behavior still holds when both are static.
**Done when:** the pipeline resolves to the closest centered target by default, and a slowly-moving target at equal distance to a static one wins priority. Fast-motion robustness is explicitly out of scope (§11) — do not chase it here.

### #16: Temporal Smoothing
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #15

**Build:** Require a target to persist ~3 consecutive frames before its cue fires.
**Watch out:** without this, detections flicker and cues stutter — the exact instability a fast-moving-object demo would expose (which is why the thrown-object test was cut, §19.1).
**Test:** wave a hand briefly through frame (should NOT fire a cue) vs. hold an object steady for a second (should fire).
**Done when:** brief flickers are filtered; steady targets confirm.

### #17: Drop-Off / Negative-Obstacle Detection
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #13

**Build:** Use the depth map to detect *downward* discontinuities — curbs, downward stairs, unguarded platform edges, potholes — as a semantically distinct hazard from "object in the way."
**Watch out:** nothing else in the pipeline answers the downward case, and it's life-safety-critical in India — this closes the pothole/edge loop the problem statement (§2) opened.
**Test:** point the phone at a taped "curb line" / a real step-down; confirm a distinct drop-off flag fires (different from an obstacle flag).
**Done when:** downward drop-offs raise their own dedicated hazard flag.

---

## Phase 3 — Sonification & Haptics (the crown jewel — §5)

> This is the defensible IP. The vision pipeline is a commodity; *how proximity, identity, and direction become something a blind person can act on instantly* is the product. Build each channel standalone against synthetic inputs first, then integrate at #22.

### #18: Direction Channel — Stereo/HRTF Pan
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #6, #1

**Build:** Map a target's azimuth (left↔right position) to stereo/HRTF panning of the cue.
**Watch out:** direction gets its *own* channel — don't let it bleed into the distance or identity signal. One dimension, one channel.
**Test:** feed synthetic azimuths (hard left, center, hard right); confirm the cue pans convincingly to each with eyes closed.
**Done when:** a listener can point to where a synthetic target is, from pan alone.

### #19: Distance/Urgency Channel — Pulse Repetition Rate
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #6

**Build:** Map proximity/urgency to **pulse repetition rate** — faster pulses = closer. The parking-sensor / reversing-car model.
**Watch out:** do NOT use pitch for distance. Pitch is reserved for identity (#20); overloading it is the single biggest ambiguity in the original design (§11). Repetition-rate is culturally pre-learned in India — nobody has to be taught it.
**Test:** feed synthetic distances sweeping near→far; confirm pulse rate speeds up as distance drops, with no pitch change.
**Done when:** distance is legible from beep rate alone, pitch untouched.

### #20: Identity Channel — Auditory Icons (+ Spearcon Fallback)
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #6

**Build:** Map object class to a short **auditory icon** — a dog-yip for a dog, a footstep-thud for a person, a low wooden knock for furniture. Keep single-word **spearcons** (sped-up speech) as a fallback for classes without a bespoke icon.
**Watch out:** use icons/spearcons, not arbitrary tones — nobody should have to memorize "beep pattern 4 = vehicle." Icons have a real, citable learnability edge.
**Test:** play the icon set to someone cold; confirm they can name the object from the sound with no training on the common classes.
**Done when:** identity is recognizable from timbre alone, with a spearcon fallback wired for rare classes.

### #21: Proximity Haptics — Primary Channel
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #6

**Build:** Drive the phone's haptic motor with **graded intensity/rhythm** as a *primary* how-close channel (richer than a binary buzz), running in parallel with the audio proximity pulse.
**Watch out:** the original design used vibration only as a `<0.5m` panic backstop — that undersells it badly. Haptics and hearing don't compete for bandwidth; audio does. Pushing "how-close" onto haptics is what resolves the ear-occlusion tension (§6).
**Test:** eyes closed, no audio, approach an obstacle; confirm graded vibration alone conveys getting-closer.
**Done when:** proximity is fully legible from haptics alone, as a real second channel.

### #22: Cue Engine Integration
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #18, #19, #20, #21, #15

**Build:** Wire the real targeting output (#15) into all three orthogonal channels at once: pan (direction) + pulse rate (distance) + icon (identity), with haptics carrying proximity in parallel. Cap at 1–2 simultaneous cues.
**Watch out:** this is the moment the three channels must stay orthogonal under real data — don't let a shortcut collapse two dimensions back onto one signal.
**Test:** walk the phone toward the demo chair off to one side; confirm you simultaneously hear *where* (pan), *how close* (rate + haptics), and *what* (furniture knock) — and can act on all three.
**Done when:** a live target produces a correct, non-overloaded three-channel cue.

### #23: Uncertainty / Self-Trust Layer — Tiered Confidence Tagging
Priority: 🟨 SUPPORTING
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R2 (hr 22)
Blocked by: #22

**Build:** Implement three named confidence tiers per detection (adapted from a prior team project's colour-tag system — see Bible §5.3): **WHITE** (direct, high-confidence — full clean three-channel cue), **BLUE** (inferred/lower-confidence — same channels with a deliberately grainy/degraded audio texture layered in), **RED** (unknown, honestly flagged — proximity-only pulse, no identity claim). The system *sounds unsure* on BLUE and *admits it doesn't know* on RED, instead of going silent or faking confidence on either.
**Watch out:** an aid that lies about its confidence is how people get hurt. "The device knows when it doesn't know" is a genuinely strong stage claim, and citing that the team previously **built and open-sourced** a tiered-confidence system in a different project (PHANTOM-ECHO REVEAL) is a credible answer if a judge probes this — don't throw either away by silently failing or flattening the tiers back into one binary "degraded" state. Say "built" or "open-sourced," never "shipped": the prior work is a public repo, not a released product, and a judge who clicks through to a hackathon-grade repo after hearing "shipped" will discount every other claim you make. Match the verb to what's provable.
**Test:** feed high-confidence, borderline, and no-classification detections in turn (clear object / partially covered lens / lens fully obscured but depth still returning); confirm each produces a distinct, correct tier (WHITE full cue / BLUE degraded cue / RED proximity-only).
**Done when:** all three tiers are distinguishable by ear per detection. *(Rendering the tiers visually on the dashboard is a separate step — see #32 — since it needs both this ticket and the R1 dashboard #30 to exist.)*

### #24: Graceful-Degradation Ladder
Priority: 🟨 SUPPORTING
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R2 (hr 22)
Blocked by: #23, #21

**Build:** Formalize the fallback ladder: (1) full — icon + pan + pulse rate; (2) low visual confidence — drop identity, keep proximity-pulse + uncertainty texture (#23); (3) total failure — haptic-only panic threshold (`<0.5m`).
**Watch out:** don't let the ladder have a dead end — each rung must degrade to the next automatically.
**Test:** force each failure level in turn; confirm the system steps down one rung at a time, never to silence.
**Done when:** all three rungs are reachable and step down cleanly.

### #25: Two Operating Modes — Flow vs. Scan/Seek
Priority: 🟨 SUPPORTING
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R2 (hr 22)
Blocked by: #22

**Build:** A mode switch: **flow** (walking) = sparse, urgent-only cues (the #14–#22 default); **scan/seek** (stopped, on-demand) = richer, exploratory, and the home for voice goal-seeking (Phase 4).
**Watch out:** state this as a deliberate HCI decision mirroring real blind-user behavior (stop-to-orient, then move) — not an implementation detail.
**Test:** toggle modes; confirm flow stays minimal while scan/seek opens up the richer "what's around me" behavior.
**Done when:** the two modes are distinct and switchable, and scan/seek is ready to host Phase 4.

---

## Phase 4 — Voice Goal-Seeking (co-headline — §7, demo climax)

### #26: Voice Command Capture
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R2 (hr 22)
Blocked by: #10, #25

**Build:** In scan/seek mode, capture a short spoken command via Whisper-Tiny and extract the target noun ("door", "seat", "exit").
**Test:** say "find the door" offline; confirm the target word "door" is extracted correctly.
**Done when:** a spoken goal reliably becomes a target word, on-device and offline.

### #27: Open-Vocabulary Grounding
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R2 (hr 22)
Blocked by: #11, #26

**Build:** Feed the target word (#26) + live frame to OWL-ViT; get the screen region of the named object.
**Watch out:** this is the moat — it works for anything you can say, not a fixed object library. Don't quietly restrict it to a hardcoded class list; that throws away the whole differentiator.
**Test:** say three different objects in the room in turn; confirm each is grounded to the correct region.
**Done when:** an arbitrary spoken object is located live in the frame.

### #28: Vector-to-Goal Cueing
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R2 (hr 22)
Blocked by: #27, #22

**Build:** Continuously steer the user toward the grounded target by reusing the cue engine (#22) — pan toward it, pulse/haptics as they close in, confirmation when reached.
**Watch out:** *they describe, we vector* — this is the single clearest differentiator over Seeing AI / Lookout. The output is continuous body-guidance, not a one-shot description.
**Test:** say "find the door", start facing away; confirm you're audibly/haptically turned and walked to it, with a clear "arrived" cue.
**Done when:** a named goal produces continuous guidance all the way to the target.

---

## Phase 5 — Systems, Office Kit & Dashboard (directly scored)

### #29: Office Kit Live During Red Light
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #4, #6

**Build:** Keep the Office Kit screen-mirror actively running while building/testing on the phone during Red Light windows, for visibility and debugging.
**Watch out:** 25% of the total score is device telemetry — Office Kit usage (10%) and creative phone use (15%) are measured, not pitched. Idle-but-mirrored still beats not-mirrored.
**Test:** confirm the phone screen mirrors to the laptop live and stays up through a full build session.
**Done when:** Office Kit is a running habit during every Red window, not an afterthought.

### #30: Live Debug Dashboard + QR-Code Multi-Judge Access
Priority: 🟥 CORE
Type: Prototype
Light: 🟢 GREEN
Checkpoint: Eval R1 (hr 8)
Blocked by: #29, #12, #13

**Build:** A dashboard showing raw camera feed, detection boxes, depth heatmap, and a radar-style view of active audio cues — streamed to the laptop via Office Kit. Also expose it as a local-network web view with a QR code (adapted from a prior project's demo pattern) so judges can watch live on their own phones instead of crowding one laptop screen. *(The WHITE/BLUE/RED confidence-tier overlay is deliberately NOT part of this R1 ticket — the tiers don't exist until #23, which is R2. The overlay is added later in #32.)*
**Watch out:** the dashboard is a *debug/visibility* aid and a scored Office-Kit surface — never a compute dependency the running app needs. The QR-code path is additive convenience, not a replacement for the Office-Kit-scored laptop stream — keep both. **The QR path only stays cheap if the dashboard is built as a local web view from the start.** Decide the dashboard's form factor at this ticket's kickoff, not later — retrofitting a native/desktop dashboard into a served web view is a Green-Light-only task that will compete directly with model conversion, and it is not the "just print a QR code" the pattern makes it sound like.
**Test:** trigger a detection on the phone; confirm boxes, heatmap, and radar cue view all update live on the laptop. Separately, scan the QR code on a second phone and confirm the same live state appears there too.
**Done when:** the dashboard reflects the phone's live camera/detection/depth/cue state in real time on both the laptop (Office Kit) and any device that scans the QR code.

### #31: Airplane-Mode Offline Proof
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R1 (hr 8)
Blocked by: #22

**Build:** A clean flow where you toggle the phone to airplane mode in front of judges and the full pipeline keeps working.
**Watch out:** claim 100%-offline against the *specific* named competitors, not vision apps in general (§4 honest-framing note). Don't overreach the claim.
**Test:** enable airplane mode, run the obstacle course; confirm every cue still fires with zero network.
**Done when:** the demo runs identically with the internet physically disabled.

### #32: Confidence-Tier Dashboard Overlay (WHITE/BLUE/RED)
Priority: 🟨 SUPPORTING
Type: Prototype
Light: 🟢 GREEN
Checkpoint: Eval R2 (hr 22)
Blocked by: #23, #30

**Build:** Add a per-detection WHITE/BLUE/RED confidence-tier badge to the debug dashboard (#30) and the QR-accessible web view, driven by #23's tiering. This is the *visual* proof of the "the device knows when it doesn't know" claim — a judge watching the dashboard should see a detection drop from WHITE to BLUE to RED live as you dim the lights or partially cover the lens.
**Watch out:** this ticket exists as its own step precisely because the tiers (#23, R2) don't exist when the core dashboard (#30, R1) is first built — do not try to fold it back into #30 or you'll block an R1 must-have on an R2 feature. It is purely additive and cut-safe: if hour 22 is tight, the tiers still work *by ear* (#23) without the visual overlay.
**Test:** dim the lighting on a detected object; confirm the dashboard badge steps WHITE → BLUE → RED in sync with the audio texture change, on both the laptop and a QR-scanned phone.
**Done when:** the live confidence tier is visible on the dashboard, matching what the audio is doing.

---

## Phase 6 — Optional / Cut-First (only if the spine is green)

### #33: YamNet Hazard Listening
Priority: 🟨 SUPPORTING
Type: Prototype
Light: 🟢 GREEN
Checkpoint: Stretch
Blocked by: #3, #22

**Build:** Convert YamNet, run it async to flag hazard sounds (horns, sirens) as an extra alert channel.
**Test:** play a horn/siren clip; confirm an async hazard flag fires without stalling the main pipeline.
**Done when:** background hazard sounds raise a non-blocking alert.

### #34: Voice Auto-Ducking
Priority: 🟨 SUPPORTING
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Stretch
Blocked by: #33

**Build:** Reuse the always-on mic to drop spatial-audio volume ~80% when human conversation is detected nearby.
**Test:** start talking near the phone; confirm cue volume ducks, then restores when talk stops.
**Done when:** conversation automatically ducks the audio cues.

### #35: Scene Narration (Llama-3.2-1B)
Priority: ⬜ SEED / PITCH-ONLY
Type: Prototype
Light: 🟢 GREEN
Checkpoint: Cut first
Blocked by: #3, #25

**Build:** On double-tap in scan/seek, speak a one-sentence scene description. Build only if everything above is done; otherwise narrate as roadmap.
**Watch out:** this is first to cut — don't let it eat time the sonification spine needs.
**Done when:** either it works on double-tap, or it's a clean roadmap line in the pitch.

### #36: Gesture Control (MediaPipe)
Priority: ⬜ SEED / PITCH-ONLY
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Cut first
Blocked by: #6

**Build:** Hands-free pause/mute via a hand gesture. Build only as a last luxury.
**Watch out:** expect version/setup friction; not worth risking the spine. Cut first alongside #35.
**Done when:** either a single gesture pauses/mutes, or it's dropped without regret.

### #37: FastSam-S Segmentation
Priority: ⬜ SEED / PITCH-ONLY
Type: Research
Light: 🟢 GREEN
Checkpoint: Stretch
Blocked by: #3, #13

**Build:** Pixel-level masking to sharpen targeting precision, if time allows.
**Done when:** either masks measurably tighten targeting, or it's left as future work.

---

## Phase 7 — Proof & Demo (do not skip these)

### #38: Target-Metrics Table Filled
Priority: 🟥 CORE
Type: Research
Light: 🔴 RED-OK
Checkpoint: Eval R2 (hr 22)
Blocked by: #22, #17

**Build:** Measure and fill real numbers into the §12 table: detection range, min obstacle size, angular resolution of audio cues, drop-off detection distance, obstacle-course completion time, collision count, false-alarm rate. Report each as either a **held-out** number (first run on an arrangement no one has walked/tuned against during development) or a **development-time** number, and label which is which — don't blend them into one unqualified figure (Bible §12, non-circular validation discipline, same rigor applied on a prior team project).
**Watch out:** a measured-against-target table reads as engineered, not just demoed — but every number must be real, not estimated, and the held-out/development-time distinction must be honest, not just labeled for show. **Assign one owner for held-out integrity** (the Systems/Demo person, per §20): a single held-out obstacle layout, kept physically separate and never walked, tuned against, or rehearsed on during development until this ticket runs. In a 30-hour sprint with three people sharing one course, an unowned held-out set gets contaminated by accident by hour 20 — the discipline is real on paper and violated in practice unless one person guards it.
**Test:** re-run two measurements and confirm they reproduce within a sane margin. Confirm at least the core collision-avoidance metrics have one genuine held-out run (fresh obstacle layout, no prior walkthroughs) logged separately from development-time numbers, and confirm with the held-out owner that the layout was genuinely untouched before this run.
**Done when:** every row has a real, reproducible number, and the held-out vs. development-time split is explicit in the table.

### #39: Obstacle-Course Demo Built & Rehearsed
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R2 (hr 22)
Blocked by: #22, #17, #28

**Build:** The static course run: veer around a chair → duck/stop for a head-height obstacle → stop at the taped drop-off line → climax: a judge says "find the door" and the volunteer is vectored to it (§#28).
**Watch out:** the thrown-object test was cut for being highest-variance — do NOT reintroduce fast-moving objects. Static course only.
**Test:** run the full course end-to-end twice under real lighting; log whether each completed without manual intervention.
**Done when:** two consecutive clean live runs.

### #40: Blindfold Caveat + Real-User Feedback
Priority: 🟥 CORE
Type: Discuss
Light: ⚪ PRE-EVENT
Checkpoint: Eval R1 (hr 8)
Blocked by: —

**Build:** A stated pitch line owning that a blindfolded sighted volunteer is a proxy, not a validation — plus, if at all possible, one conversation's worth of feedback from an actual blind/low-vision person, cited on stage.
**Watch out:** co-design credibility is the thing most assistive-tech hackathon projects skip — naming the caveat yourself beats a judge catching it.
**Done when:** you can say the caveat out loud, ideally backed by one real user conversation.

### #41: Challenge-Response Answers Rehearsed
Priority: 🟥 CORE
Type: Discuss
Light: ⚪ PRE-EVENT
Checkpoint: Eval R1 (hr 8)
Blocked by: —

**Build:** Rehearse the §10 answers cold: "isn't this biped on a phone?", "you occluded their ears", "is that real metres?", "why not Seeing AI/Lookout?", "what if the phone dies?", "you tested a blindfolded person." Add one more: "how do you know your confidence tiers/metrics are real, not just a demo trick?" — answer: cite the tiered WHITE/BLUE/RED confidence system and the non-circular held-out/development-time metric split as patterns the team previously **built and open-sourced** in a different project (PHANTOM-ECHO REVEAL). Never say "shipped" — it's a public repo, not a released product, and the overclaim is more damaging than the credibility it buys.
**Watch out:** name the competitive landscape (biped, .lumen, Seeing AI, Lookout, Soundscape, SmartCane, The vOICe) *before* a judge names it first.
**Done when:** every team member can answer each challenge in one clean sentence, including the new credibility question.

### #42: Backup Recording
Priority: 🟥 CORE
Type: Prototype
Light: 🔴 RED-OK
Checkpoint: Eval R2 (hr 22)
Blocked by: #39

**Build:** Record a full successful obstacle-course run (including the airplane-mode toggle and the voice-search climax) as a fallback if the live demo fails on the day.
**Test:** play the recording start to finish; confirm it clearly shows avoidance → drop-off → "find the door" → arrival.
**Done when:** a usable backup video exists.

### #43: Pitch Narrative — Both Formats
Priority: 🟥 CORE
Type: Discuss
Light: 🔴 RED-OK
Checkpoint: Top-10 stage (Sun 13:30)
Blocked by: #38, #39, #41

**Build:** Two collateral forms: the hands-on table walkaround (Eval R1/R2) and a timed stage narrative (top-10 only, likely no live blindfold run). Both lead with the differentiators (§24), cite metrics from #38, and state the honest-limits lines (relative depth, offline-vs-named-competitors, ear-occlusion answer). Frame the whole degradation philosophy (§13.3, #23, #24) as the Bible's **"Prove → Measure → Imagine"** line: the system **proves** what direct detection confirms (WHITE), **measures** what a lower-confidence-but-real signal gives it (BLUE), and stays honestly **unknown** (RED) rather than fabricating an answer — "we eliminate and admit what's left unknown, we don't hallucinate certainty."
**Watch out:** say "Imagine" as the thing the system *refuses* to do, not a feature it has — the demo only ever shows Prove and Measure. Phrased as a capability it sounds like a missing feature; phrased as a discipline it's a credibility win.
**Test:** dry-run each format once against someone who hasn't seen the project; confirm every spoken number matches #38's table, and confirm the Prove→Measure→Imagine line lands as "here's the honesty discipline," not "here's a feature we didn't finish."
**Done when:** both formats are rehearsed and every claim traces to a real number or a stated design decision.

---

## Future Work (out of scope for this hackathon build)

These aren't rejected ideas — they're real extensions that don't fit this build's time/complexity budget. Worth naming if asked, worth returning to after the event:

1. **Crowdsourced hazard mapping** — a shared, community-updated map of fixed hazards. Pitch it as a roadmap line only; never build it here (Bible §21 marks it "pitch-only, never build").
2. **Full metric depth calibration** — true distance in metres. Deliberately rejected (§11): it overclaims precision the monocular sensor can't deliver, and relative-proximity ordering is the honest, sufficient answer for collision avoidance.
3. **Full Time-to-Collision physics** — frame-to-frame object tracking for true TTC. Rejected for this build (§11); simple closest-in-center distance logic covers the common case.
4. **Bone-conduction vs. open-ear as a shipped hardware SKU** — the event uses whatever #1 decides; a productized headset choice with field testing is post-event work.
5. **Onboarding mode + verbosity dial as a trained arc** — sensory substitution has a real weeks-long training curve (vOICe). A full novice→expert onboarding system is future work; for the event, state it openly as a maturity signal rather than implying zero learning curve (§15).
6. **Real blind/low-vision co-design program** — beyond the one feedback conversation (§#40), a proper participatory design cycle with trained cane users is the honest next step before any real deployment.

---

*This build map is a companion to `secondsense_bible.md`. The Bible explains WHY; this map tells you WHAT to build, in WHAT order, how to test each piece, which venue window (Red/Green) and checkpoint each belongs to, and what's genuinely future work. When a step and the Bible disagree, the Bible is the source of truth on intent — but follow this map's ticket order, Light tags, and Test steps for execution.*
