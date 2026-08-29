# SECONDSENSE — Project Bible v4
### iQOO Hackathon 2026 · Bengaluru City Battle · Aug 29–30
### On-Device Spatial Audio + Haptic Navigation for the Visually Impaired

> **Name note:** Renamed from the earlier working title "Echo" to **SecondSense** to avoid the Amazon Echo naming collision — a bad collision for an audio-first accessibility product. SecondSense also better reflects the actual positioning: this is a second sense, complementing the cane, not an echo/sonar gimmick.

---

## 1. One-Line Pitch

**SecondSense turns any iQOO smartphone into a real-time "extra sense" for blind and visually impaired people — a cane-complement, not a cane-replacement, that catches what a white cane physically cannot (head-height hazards, drop-offs, and open-vocabulary goal-seeking), delivered through spatial audio and haptics, entirely offline, zero recurring cost.**

---

## 2. The Problem

- India has an estimated **4.95 million blind individuals** and **70 million visually impaired persons**. Economic loss from blindness in India is estimated at **₹845 billion** (GNI loss), with cumulative lifetime loss estimated at **₹22,565 billion**.
- Indian streets — uneven pavements, unmarked construction, chaotic traffic, encroaching vendors — are a uniquely hostile, unmapped environment.
- **The white cane's structural blind spot:** a cane sweeps the ground plane. It cannot detect head-and-torso-height hazards (overhanging branches, open cabinet doors, truck mirrors, scaffolding, signage) — a well-documented cause of real facial and head injuries. A chest-mounted forward camera is positioned exactly where the cane isn't.

---

## 3. Positioning — Complement the Cane, Don't Replace It

**Don't pitch this as "better than a cane."** No blind user abandons a cane for a phone, and "replacement" framing invites the fatal question: *what happens when the phone dies?* The defensible frame is complement:

- **The cane owns:** the ground plane, tactile confirmation, zero-battery reliability.
- **SecondSense owns what the cane is blind to:** head-height hazards, semantic identity ("what is it," not just "something is there"), drop-offs and negative obstacles (curbs, stairs, unguarded platform edges — high-relevance in India), and open-vocabulary goal-seeking ("find the door").

### The real competitive landscape (name these on stage — don't let a judge name them first)

| Product | What it actually does | How SecondSense differs |
|---|---|---|
| **biped.ai / NOA** | Worn harness, 3D cameras, spatial audio via **bone-conduction** headphones, obstacle detection at multiple heights, GPS turn-by-turn | Zero dedicated hardware — runs on a phone people already own; adds open-vocabulary object search a fixed-function harness doesn't have |
| **.lumen glasses** | Camera + IMU + GPS "PAD AI," guides primarily via **forehead haptics**, audio as secondary channel, "Find Me" object-locate feature | Validates our haptics-forward instinct independently; SecondSense is phone-native (no dedicated headset) and open-vocabulary rather than a fixed object library |
| **Microsoft Seeing AI / Google Lookout** | On-device/phone scene description — point, get a description of what's there | **They describe, we vector.** They tell you a chair is present; SecondSense continuously guides your body toward or around it in real time. This is the single clearest differentiator — say it explicitly. |
| **Microsoft Soundscape (GPS)** | Macro-navigation, audio beacons toward destinations | GPS drift of 3–5m in urban India — useless for the pothole in front of you right now |
| **IIT Delhi SmartCane (ultrasonic)** | "Something is there," ~₹3,500 | Zero semantic awareness — can't distinguish a wall from a person from a dog |
| **The vOICe (Meijer, 1992–present)** | Decades-old sensory-substitution research: image → soundscape mapping | Cite this to show you know the field predates the current AI wave — it buys credibility, not competition |

**Honest framing note:** cloud vision apps and even some "offline" competitors above still lean on connectivity for their richest features. SecondSense's 100%-offline claim, proven live via airplane mode, is a real point of difference — but say it about the *specific* competitors above, not vision apps in general.

---

## 4. Core Concept — Two Modes, Not One

Blind users don't walk in one continuous mode — they **stop to orient, then move.** SecondSense should mirror that:

1. **Flow mode (walking):** sparse, urgent-only cues. Obstacle avoidance, drop-off warnings, head-height hazards. Audio stays minimal; haptics carry proximity.
2. **Scan/seek mode (stopped, on-demand):** richer, exploratory. "What's around me?" or "Find the door" — the open-vocabulary goal-seeking capability (§7) lives here.

This mode split is itself good, defensible HCI — state it as a deliberate design decision, not an implementation detail.

---

## 5. Sonification & Haptic Design — The Actual Product

**This is the crown-jewel section.** The vision pipeline (detection + depth) is increasingly a commodity — anyone can wire up YOLO and a depth model. The defensible IP is *how proximity, identity, and direction get turned into something a blind person can act on instantly.* Right now this was three bullet points; it should be the best-argued section in the document.

### 5.1 Three information dimensions → three orthogonal channels

Don't make one parameter (pitch) do double duty. Split cleanly:

| Dimension | Channel | Why |
|---|---|---|
| **Direction (azimuth)** | Stereo/HRTF pan | Directly spatial, intuitive |
| **Distance/urgency** | **Pulse repetition rate** (not pitch) — faster pulses = closer | This is the parking-sensor / reversing-car model. It's culturally pre-learned in India already; nobody has to be taught what a faster beep means |
| **Identity ("what is it")** | **Auditory icons** (a short dog-yip for a dog, a footstep-thud for a person, a low wooden knock for furniture) — not abstract tones | Auditory icons have a real, citable learnability advantage over arbitrary earcons (no need to memorize "beep pattern 4 = vehicle"). **Note:** the literature also shows short sped-up speech ("spearcons") can match or beat auditory icons on raw accuracy/reaction time — worth keeping single-word spearcons as a fallback option for identity, especially for less-common object classes where you don't have time to design a bespoke icon. |

Freeing pitch from proximity duty and giving it to identity (via icon timbre) resolves the biggest ambiguity in the original design.

### 5.2 Promote haptics from fallback to a primary channel

The original design used vibration only as a `<0.5m` panic backstop. That undersells it. Haptics and hearing **don't compete** for bandwidth — audio does. Push the *how-close* channel onto graded haptic intensity/rhythm (richer than a binary buzz) and reserve audio primarily for *what* and *where*. This directly resolves the ear-occlusion tension in §6 below — it's not just a safety net anymore, it's a deliberate second channel.

### 5.3 A self-trust / uncertainty layer — tiered confidence tagging

When lighting is poor or detection confidence drops, the system should sound *unsure*, not go silent or fake confidence. An aid that lies about its confidence is how people get hurt — and "the device knows when it doesn't know" is a genuinely strong, defensible claim on stage.

**This gets a real, named tiering system, not just one binary "degraded" state** — borrowed and adapted from a prior team project (PHANTOM-ECHO REVEAL, occlusion-aware 3D reconstruction), which used a five-tier colour-coded confidence tag on every reconstructed point. SecondSense adapts the same *pattern* — not the code, the domains are unrelated — to its own signal types:

| Tier | Source of the cue | Audio/haptic treatment |
|---|---|---|
| **WHITE** — direct, high-confidence | Object cleanly detected + confident depth reading | Full three-channel cue (icon + pan + pulse rate), clean and crisp |
| **BLUE** — inferred/lower-confidence | Detection present but confidence below threshold (poor light, partial occlusion) | Same channels, but with the deliberately grainy/degraded texture layered in — audibly "less certain" |
| **RED** — unknown, honestly flagged | Depth signal present but no reliable classification at all | Proximity-only pulse + uncertainty texture, no identity claim — falls through to §13.3's ladder rung 2 |

This isn't just a UX nicety — it's a **provable design lineage**: the team has previously built and shipped a working tiered-confidence system in a different domain, which is a genuinely rare, credible thing to say on stage when a judge asks "how do you know your system isn't overconfident." Point to it directly if asked.

**The "Prove → Measure → Imagine" framing** (also adapted from the same prior project) is useful pitch language for the whole degradation philosophy in §13.3: SecondSense doesn't *predict* what it can't see — it **proves** what direct detection confirms (WHITE), **measures** what a lower-confidence-but-real signal gives it (BLUE), and when neither holds, it stays honestly **unknown** (RED) rather than fabricating an answer. Say this explicitly — "we eliminate and admit what's left unknown, we don't hallucinate certainty" — it's a sharper, more defensible framing than "we have an uncertainty mode."

### 5.4 Honest claims on depth

Monocular depth (Depth-Anything-V2) gives **relative/inferred** depth, not calibrated metric distance. When a judge asks "is that real distance in metres?" — the credible answer is: *"We don't claim absolute metric range. We claim reliable relative-proximity ordering and rate-of-approach against a one-tap calibration baseline — which is what collision avoidance actually needs."* Stating the limit up front makes everything else you claim more believable.

---

## 6. The Ear-Occlusion Tension (state your answer explicitly)

A blind person's primary navigation sense is hearing. Mandating fully-occluding wired headphones for latency reasons blocks the sense they depend on most for everything *else* — traffic, voices, footsteps. This is a real design tension, not a minor UX note, and a judge who knows the space will ask about it.

You don't have to solve it perfectly, but you do need a stated position:
- **Option A:** wired bone-conduction (what biped uses) — keeps the ear canal open, still wired for latency.
- **Option B:** lean on §5.2's haptics-primary redesign so audio load (and therefore occlusion) drops substantially, making open-ear earbuds viable even if not bone conduction.
- **Whichever you pick, say the sentence "we thought about ear occlusion and here's our answer" out loud in the pitch.** Not having an answer is worse than any specific answer.

---

## 7. Voice Search Is the Headline Capability, Not a Wow Feature

"Warn me about hazards" is a crowded category — every competitor above does some version of it. **"Vector my body toward a goal I named out loud"** (find the empty seat, the exit, the door, the person waving) is a fundamentally different, more valuable primitive — and it's exactly what describe-only incumbents (Seeing AI, Lookout) don't do. Open-vocabulary goal-seeking is the moat.

- Promote it from priority #9 to a **co-headline** alongside obstacle avoidance in the pitch, the architecture section, and the demo climax.
- It's also your best indoor story (§9) — GPS-free, controllable, demoable.

---

## 8. Add: Drop-Off / Negative-Obstacle Detection

Curbs, downward stairs, and — especially relevant in India — unguarded train-platform edges are semantically distinct from the "object in the way" case, and they're life-safety-critical. Depth estimation is well-suited to catching *negative* obstacles, but nothing in the original pipeline explicitly handles the downward case. Potholes are in the problem framing (§2) but nothing in the build currently answers them directly — this closes that loop and adds a high-drama, high-relevance capability.

---

## 9. Indoor Is the Strongest Use Case, Not an Afterthought

The original framing is street-first. But the most controllable, most reliably-demoable, most differentiated use case is **indoor**: find the door, the seat, the exit — exactly where GPS-based tools (Soundscape) are useless, and exactly where your demo physically happens anyway. Make indoor navigation a first-class, explicitly-stated use case, not just an implied byproduct of "it also works indoors."

---

## 10. What We Say When Challenged

Write these answers now — this is how the room gets won.

| Challenge | Answer |
|---|---|
| "Isn't this just biped on a phone?" | Zero dedicated hardware — runs on a phone people already own — plus open-vocabulary goal-seeking a fixed-function harness doesn't have. |
| "You occluded their ears." | [State your §6 answer — bone conduction, or haptics-primary reducing audio load.] |
| "Is that real distance in metres?" | We claim relative-proximity ordering and rate-of-approach, not calibrated metric range. That's what collision avoidance actually needs. |
| "You tested a blindfolded sighted person, not a blind user." | Acknowledged directly — see §12. Blindfolded testing is a proxy, not a validation. |
| "Why not just use Seeing AI / Lookout?" | They describe. We vector. Point-and-describe vs. continuous real-time guidance are different products. |
| "What happens when the phone dies?" | The cane is the fallback of record — SecondSense is explicitly a complement, never pitched as a replacement. |

## 11. Considered and Rejected

| Idea | Why rejected |
|---|---|
| Full metric depth calibration | Overclaims precision the sensor can't deliver; relative-proximity ordering is honest and sufficient |
| Pitch-for-proximity | Collides with pitch-for-identity; repetition-rate is more intuitive and frees pitch for identity |
| Silence on low-confidence detections | Silently failing is worse than admitting uncertainty — degraded/grainy audio texture instead |
| Full Time-to-Collision physics | Needs frame-to-frame tracking; simple centered-distance logic covers the common case |
| Replacing the cane in the pitch narrative | Invites "what if the phone dies" objection; complement framing sidesteps it entirely |

---

## 12. Target Metrics (fill in with real numbers before the event)

| Metric | Target |
|---|---|
| Detection range | — |
| Minimum obstacle size reliably detected | — |
| Angular resolution of audio cues | — |
| Drop-off detection distance | — |
| Obstacle-course completion time | — |
| Collision count (test runs) | — |
| False-alarm rate | — |

A measured-against-target table reads as engineered, not just demoed.

**Non-circular validation discipline (adapted from a prior team project's methodology):** report a clear split between numbers measured on a *held-out* condition the system hasn't been tuned against live (e.g. a fresh obstacle arrangement no one has walked through during development) versus numbers from repeated development-time runs. Don't blend them into one inflated headline figure — state both, and say plainly which is which if a judge asks "how do we know these numbers are real." This is the same discipline PHANTOM-ECHO REVEAL used (separating real held-out data from synthetic self-consistency checks) and it reads as far more credible than a single unqualified number.

---

## 13. Technical Architecture (pipeline — unchanged from v1, ignore feasibility)

### 13.1 Model Pipeline (Qualcomm AI Hub, all NPU-native, all offline)

| Stage | Model | Function | Latency (NPU) |
|---|---|---|---|
| 1 | **YOLOv11 (INT8)** | Object detection & classification | ~20 ms |
| 2 | **Depth-Anything-V2 (W4A8 quantized)** | Monocular relative-distance calculation, incl. drop-off/negative-obstacle cases | ~5 ms |
| 3 | *(optional)* **FastSam-S** | Pixel-level masking for precision | ~3 ms |
| 4 | **Whisper-Tiny + OWL-ViT** | Voice-triggered zero-shot open-vocabulary goal-seeking — **now a co-headline feature, not a stretch goal** | ~12.5 ms |
| 5 | *(if time)* **YamNet** | Background audio hazard detection (horns, sirens) — async | <1 ms trigger |
| 6 | *(cut first)* **Llama-3.2-1B-Instruct** | Spoken scene narration on double-tap | ~18 tok/sec |
| 7 | *(cut first)* **MediaPipe Hand Gesture** | Hands-free pause/mute control | ~38 µs |

Steps 1–2 run in parallel on the Hexagon NPU → core pipeline budget: ~35–40ms end-to-end.

### 13.2 Targeting Logic
- Center-crop targeting: only objects in the center ~30% of frame get prioritized during flow mode.
- Max 1–2 simultaneous audio cues at any time.
- Simple closest-in-center priority first; full Time-to-Collision math is a stretch goal only (see §11).
- **Static/dynamic priority split** (adapted from a prior project's separate tracked-object layer): a moving object (a person walking toward the user) gets priority over a static object at the same distance (a parked chair), even under the simple closest-in-center rule. This doesn't require full velocity tracking — a coarse frame-to-frame "is this bounding box moving" flag is enough to bump priority. Cheap to add on top of #15/#16, and it's a real safety improvement: a static obstacle can be routed around at leisure, an approaching person cannot.

### 13.3 Graceful Degradation Ladder (formalizes §5.3)
1. Full confidence: identity (auditory icon) + direction (pan) + distance (pulse rate).
2. Low visual confidence: drop identity, keep proximity-pulse + a deliberately degraded/grainy texture signaling uncertainty.
3. Total failure: haptic-only panic threshold (<0.5m).
Temporal smoothing: ~3 consecutive frames before a specific cue fires, to avoid flicker.

---

## 14. Hardware Hook — Use the iQOO, Explicitly

Continuous camera capture + multi-model NPU inference is a sustained thermal and battery load. The iQOO 15 carries a genuinely large **14,000 mm² vapor chamber** and a **7,000 mAh battery** — this is precisely the workload that hardware exists for. Most teams won't articulate *why* the sponsor's hardware matters to their specific build. Say it — it's true, and it flatters the platform honestly rather than as a throwaway line.

---

## 15. Onboarding & Verbosity

Sensory substitution has a real training arc (vOICe users train for weeks). Don't let the demo imply zero learning curve — that's the one claim a knowledgeable or blind judge will instantly distrust.

- A short guided onboarding mode.
- A **verbosity dial**: novice = fewer, louder, simpler cues; expert = denser information, more object classes distinguished.
- State this openly as a maturity signal, not a gap.

---

## 16. Fail-Safes (mostly unchanged, integrated with above)

- **Headphone choice**: resolve per §6 — bone conduction or haptics-primary redesign; source and pre-test before the event regardless of which.
- **Haptic channel**: now primary for proximity (§5.2), not just a `<0.5m` panic backstop.
- **Semantic-failure fallback**: proximity-pulse + uncertainty texture (§5.3, §13.3).
- **Temporal smoothing**: ~3 frames before a cue fires.
- **Airplane mode, proven live**: internet physically disabled in front of judges before the demo — direct proof of the offline claim.

---

## 17. Physical/UX Fixes (unchanged)

- Mount, don't hold: lanyard/chest strap so the camera stays forward and steady.
- One-tap calibration: hold level once, tap to set baseline "forward" — not full gyroscope auto-correction.
- Voice auto-ducking *(if time)*: reuse the always-on hazard mic to drop spatial-audio volume ~80% during conversation.

---

## 18. iQOO Hackathon Mechanics — Build the Bible Around the Actual Rules

### 18.1 The real scoring rubric (weight it, don't paraphrase it)

| Criterion | Weight | Measured by |
|---|---|---|
| End product quality | 30% | Jury |
| Novelty and impact | 20% | Jury |
| Creative phone use (camera, voice, on-device AI) | 15% | **HackTracker device data** |
| Technical depth | 15% | Jury |
| Office Kit usage | 10% | **HackTracker device data** |
| Demo and presentation | 10% | Jury |

**25% of the score is measured from actual device telemetry, not your pitch.** Plan Red Light/Green Light work (§18.2) accordingly — this isn't optional polish.

### 18.2 Red Light / Green Light plan

55% of the 30 hours is phone-only (Office Kit bridging to a laptop screen), 45% is dual-device.

| Phase | Approx. hours | What happens |
|---|---|---|
| **Red Light** (phone only) | ~16.5 hrs | Core app build on-device; live testing of sonification/haptic design (§5); Office Kit screen-mirror actively in use for visibility/debugging — this usage is scored |
| **Green Light** (both devices) | ~13.5 hrs | Model conversion/quantization, heavier debugging, dependency installs, dashboard polish |

Practice the Office Kit workflow (screen mirror, remote control, clipboard, file transfer) *before* the event — speed here is directly scored.

### 18.3 Two judged checkpoints before any final pitch

- **Eval Round 1 — Saturday 19:00** (~8 hours after hacking starts)
- **Eval Round 2 — Sunday 09:00** (~22 hours in)
- **Top 10 announced + final pitches — Sunday 13:30** (only top 10 get this)
- **Awards — Sunday 16:15**

This means most of your score is effectively locked in by Sunday morning. Build order must target **demo-ready by hour 8**, then iterate — not "build toward one ending." Mark in §21's feature matrix which items are must-have by Eval Round 1 vs. can slip to Eval Round 2.

### 18.4 Track

Seven tracks: FinTech/Commerce, Smart Education, HealthTech, Productivity, Smart Living, Developer Tools, Open Innovation. **HealthTech is defensible for visual impairment** — but have a one-sentence justification ready ("assistive tech for a disability is health & independence, not just convenience"), or fall back to **Open Innovation** if a judge reads assistive tech as outside "healthcare." Decide and commit before the event; don't leave it ambiguous in the pitch.

### 18.5 Team size

Registration caps teams at **3 people**. The current 4-role split (vision, audio, systems/bridge, demo/wow) needs to collapse to 3 — Systems/Bridge Owner and Demo/Wow-Feature Owner are the natural merge, since both already touch Office Kit and demo choreography.

---

## 19. The Live Demo — Redesigned Around Real Risk and Real Checkpoints

### 19.1 Replace the thrown-object test

A soft object tossed at the volunteer is the **highest-variance** demo choice: fast-moving objects are the hardest case for the pipeline's own temporal smoothing, it reads slightly gimmicky, and a live whiff undercuts the whole thesis. Replace with a **static obstacle course**:

1. Volunteer walks a short path, veers around a chair (obstacle avoidance).
2. Ducks/stops for a head-height obstacle — physically demonstrates the cane-blind-spot story (§3).
3. Stops at a taped "curb/drop-off" line (§8's negative-obstacle detection).
4. **Climax:** a judge says "find the door" out loud, volunteer is audibly/haptically vectored to it (§7).

More representative, far more controllable, and demonstrates avoidance + drop-off + semantic goal-seeking in one unscripted run.

### 19.2 Name the blindfolded-volunteer caveat out loud

A blindfolded sighted volunteer is **not** a representative blind user — they lack the trained spatial cognition and cane skills a real blind user has, which can make a demo both over- and under-sell the product. State this directly in the pitch rather than letting a judge catch it. If at all possible, get even one conversation's worth of feedback from an actual blind or low-vision person before the event, and say so on stage — co-design credibility is the thing most assistive-tech hackathon projects skip.

### 19.3 Match collateral to format

- **Eval Round 1 / Round 2** (likely jury walkarounds at your table): the hands-on obstacle-course demo above.
- **Top 10 stage pitch** (if you make it): a timed narrative presentation — different pacing, probably no live blindfolded run in a stage format. Prepare both.

---

## 20. Team Role Split (3-person cap)

- **Vision + Sonification owner** — YOLOv11 + Depth-Anything-V2 integration, center-crop targeting, **and** the audio/haptic design in §5 (kept together since they're two halves of one signal chain)
- **Systems / Office Kit / Demo owner** — dashboard, calibration mode, Office Kit workflow (scored), demo choreography and rehearsal
- **Voice + Hazard owner** — Whisper+OWL-ViT goal-seeking (§7, now co-headline), YamNet, app stability

*(Adjust to actual strengths — the point is 3 roles, not 4.)*

---

## 21. Feature Priority Matrix (build order, updated)

| # | Feature | Tier | Demo-ready by |
|---|---|---|---|
| 1 | Distance sensing (Depth-Anything-V2), incl. drop-off case | **Must-build** | Eval Round 1 |
| 2 | Object detection + spatial audio, redesigned per §5.1 | **Must-build** | Eval Round 1 |
| 3 | Haptic channel as primary proximity signal (§5.2) | **Must-build** | Eval Round 1 |
| 4 | Headphone/bone-conduction decision + testing | **Must-build (pre-event)** | Before doors open |
| 5 | Center-crop targeting | **Must-build** | Eval Round 1 |
| 6 | Voice goal-seeking ("find the door") | **Must-build — promoted from #9** | Eval Round 2 |
| 7 | Live debug dashboard (Office Kit) | **High priority (also scored directly)** | Eval Round 1 |
| 8 | Lanyard/chest mount + 1-tap calibration | **High priority** | Eval Round 1 |
| 9 | Uncertainty/degraded-audio layer (§5.3) | **High priority** | Eval Round 2 |
| 10 | Voice auto-ducking | **Medium** | If time |
| 11 | YamNet hazard listening | **Medium** | If time |
| 12 | Scene narration (Llama-3.2-1B) | **Cut first** | — |
| 13 | Full Time-to-Collision math | **Cut first** | — |
| 14 | Gesture controls (MediaPipe) | **Cut first** | — |
| 15 | Onboarding mode / verbosity dial | **Pitch-only if out of time** | Mention in narrative |
| 16 | Crowdsourced hazard mapping | **Pitch-only, never build** | Roadmap line |

---

## 22. Pre-Event Checklist

- [ ] Decide bone-conduction vs. redesigned occluding-headphone approach (§6); source and test with the actual demo phone
- [ ] Confirm Qualcomm AI Hub model downloads work via `qai-hub-models` CLI
- [ ] Lanyard/chest mount for the phone
- [ ] High-contrast demo object/course prepped (dark chair on light floor, taped drop-off line, marked "door")
- [ ] Rehearse the redesigned obstacle-course demo (§19.1) at least once, full run-through, with a backup volunteer
- [ ] Download and rehearse Office Kit (screen mirror, remote control, clipboard, file transfer) — speed here is scored
- [ ] Confirm track (§18.4) and student/professional bucket registration
- [ ] If at all possible, get one conversation's worth of feedback from a blind or low-vision person and note it for the pitch (§19.2)
- [ ] Finalize 3-person role split (§20)
- [ ] Confirm "SecondSense" name is finalized across submission form, deck, and app UI

---

## 23. Full Tech Stack (everything, in one place)

### 23.1 Hardware
- **iQOO smartphone** (event-provided flagship, e.g. iQOO 15) — primary and only compute device
- **Hexagon NPU** (Snapdragon) — runs all on-device inference
- **Wired headphones** — bone-conduction (preferred, per §6) or standard wired, chosen for latency; USB-C direct or USB-C→3.5mm DAC dongle, pre-tested
- **Phone haptic motor / LRA** — now a primary output channel (§5.2), not just a panic backstop
- **Lanyard or chest strap mount** — keeps the camera chest-forward and steady
- **Laptop (Green Light + Office Kit bridge only)** — never a compute dependency for the running app; used for dashboard, debugging, model conversion

### 23.2 On-device AI models (all NPU-native, all offline, via Qualcomm AI Hub)
| Model | Role | Status |
|---|---|---|
| **YOLOv11 (INT8)** | Real-time object detection & classification | Must-build |
| **Depth-Anything-V2 (W4A8 quantized)** | Monocular relative-distance estimation, incl. drop-off/negative-obstacle case (§8) | Must-build |
| **Whisper-Tiny** | On-device speech recognition for voice commands ("find the door") | Must-build — co-headline (§7) |
| **OWL-ViT** | Open-vocabulary zero-shot object grounding, paired with Whisper-Tiny for goal-seeking | Must-build — co-headline (§7) |
| **FastSam-S** | Pixel-level segmentation for precision, if time allows | Optional |
| **YamNet** | Background audio classification for hazard sounds (horns, sirens) — runs async | Medium priority |
| **Llama-3.2-1B-Instruct** | On-demand spoken scene narration | Cut first if short on time |
| **MediaPipe Hand Gesture** | Hands-free pause/mute control | Cut first if short on time |

### 23.3 Signal design layer (the actual product — see §5)
- **Stereo/HRTF panning** → encodes direction (azimuth)
- **Pulse repetition rate** → encodes distance/urgency (parking-sensor model, not pitch)
- **Auditory icons** (dog-yip, footstep-thud, wooden knock) and/or **spearcons** (sped-up single-word speech) → encode identity
- **Graded haptic intensity/rhythm** → primary proximity channel, freeing audio bandwidth (§5.2)
- **Degraded/grainy audio texture** → uncertainty signaling when detection confidence drops (§5.3)
- **Two operating modes**: flow mode (sparse, urgent-only) and scan/seek mode (rich, on-demand) — §4
- **Graceful-degradation ladder** (§13.3): full signal → proximity-only+uncertainty texture → haptic-only panic threshold

### 23.4 Software/tooling
- **Qualcomm AI Hub / `qai-hub-models` CLI** — model download, conversion, quantization
- **Office Kit** (pc.vivoglobal.com) — screen mirror, remote control, clipboard, file transfer between phone and laptop; directly scored (§18.1)
- **Live debug dashboard** — raw camera feed, detection boxes, depth heatmap, radar-style audio-cue view, streamed phone→laptop via Office Kit
- **QR-code multi-judge viewing** (adapted from a prior project's demo pattern): the same dashboard is also reachable via a QR code judges can scan on their own phones, so the room isn't bottlenecked on one shared laptop screen during the live obstacle-course run. Cheap to add if the dashboard is already a local web view — just expose it on the local network and print/display the QR code alongside the demo course.
- **HackTracker** — event-side device telemetry; not something we build, but something our Red Light/Green Light behavior (§18.2) needs to account for

### 23.5 Physical/UX layer
- One-tap calibration (hold level once, tap to set baseline "forward")
- Chest/lanyard mount (prevents arm-swing from breaking depth mapping)
- Onboarding mode + verbosity dial (novice/expert) — §15
- Voice auto-ducking during conversation, if time allows

---

## 24. Complete Differentiation Summary (every "why us," in one place)

1. **Zero dedicated hardware, zero recurring cost** — runs on a phone people already own, unlike biped's harness or .lumen's glasses.
2. **Describe vs. vector** — Seeing AI and Lookout narrate a scene on request; SecondSense continuously guides the body toward or around things in real time.
3. **Open-vocabulary goal-seeking** — "find the door" works for anything you can say, not a fixed object library; the single most differentiated capability in the build (§7).
4. **Cane-complement, not cane-replacement positioning** — sidesteps the "what if the phone dies" objection that a replacement framing invites (§3).
5. **Catches what the cane structurally cannot** — head-height hazards and negative obstacles/drop-offs (curbs, stairs, platform edges), not just ground-level objects (§2, §8).
6. **Orthogonal three-channel sonification** — direction, distance, and identity each get their own signal (pan / pulse-rate / auditory-icon), instead of overloading pitch for two jobs at once (§5.1). Most competitors and most hackathon-tier builds don't design this deliberately.
7. **Haptics as a primary channel, not a fallback** — resolves the ear-occlusion problem other audio-first tools either ignore or solve differently (biped: bone conduction; .lumen: haptics-led) (§5.2, §6).
8. **A self-trust / uncertainty layer** — the system sounds unsure when it is, rather than going silent or overclaiming (§5.3).
9. **Honest, non-overclaiming distance semantics** — relative-proximity ordering and rate-of-approach, not a false claim of calibrated metric range (§5.4).
10. **100% offline, proven live** — airplane mode toggled in front of judges; no cloud dependency at any point, unlike vision apps that lean on connectivity for richer features.
11. **Indoor-first as a stated use case, not an afterthought** — GPS-based tools (Soundscape) are useless indoors; this is also the demo's home turf (§9).
12. **Two operating modes matching real blind-user behavior** — stop-to-orient vs. move, rather than one undifferentiated stream of cues (§4).
13. **Sponsor-hardware-aware, honestly** — the iQOO's large vapor chamber and battery are a genuine fit for sustained camera+NPU load, and we can say why (§14).
14. **Co-design acknowledgment** — naming the blindfolded-volunteer-is-not-a-real-user caveat directly, and seeking even minimal feedback from an actual blind/low-vision person, which most assistive-tech hackathon projects skip (§19.2).
15. **Tiered confidence tagging with a real design lineage** — the WHITE/BLUE/RED uncertainty tiers (§5.3) aren't invented for this pitch; they're adapted from a prior shipped team project that used the same pattern for a different sensing problem. Citable, provable design maturity if a judge probes.
16. **Non-circular metrics discipline** — held-out vs. development-time numbers reported separately, never blended (§12), the same rigor the team has previously applied elsewhere.
17. **QR-code multi-judge dashboard access** — every judge can watch the live detection/depth/audio-cue state on their own phone during the demo, not just whoever's closest to the laptop (§23.4).

---

## 25. Sources Consulted

Deep research synthesis based on: Qualcomm AI Hub model catalog and latency benchmarks; official iQOO Hackathon 2026 rules, schedule, and scoring rubric (Reskilll); prior-art analysis of biped.ai, .lumen, Microsoft Seeing AI, Google Lookout, Microsoft Soundscape, IIT Delhi SmartCane, and The vOICe (Meijer); auditory-display HCI literature on auditory icons, earcons, and spearcons; India blindness/visual-impairment epidemiological data (PMC cost-of-illness study); iQOO 15 hardware specifications; design patterns adapted from the team's prior project PHANTOM-ECHO REVEAL (github.com/GargBhavya-tech/Phantom-Echo-Reveal) — specifically its tiered confidence-tag system, Prove→Measure→Imagine elimination philosophy, non-circular held-out validation discipline, and QR-code judge-viewing demo pattern.
