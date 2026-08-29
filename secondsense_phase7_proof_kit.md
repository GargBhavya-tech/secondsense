SecondSense — Phase 7 Proof & Pitch Kit (#38–#43 prep)
========================================================

This is prep material for `secondsense_build_map_v3.md`'s Phase 7 tickets — **none of these
tickets can be completed by writing more code.** Every one of them requires a real person, a
real phone, a real physical space, or a real conversation. What follows is the scaffolding I
CAN build: templates, drafts, and checklists, so when you do the physical parts, you're
filling in a ready structure instead of starting from a blank page.

---

## #38 — Target-Metrics Table (template, needs real measurements)

**What "done" looks like** (from the ticket): every row has a real, reproducible number, and
held-out runs (fresh obstacle layout, no prior walkthroughs) are explicit and separate from
development-time numbers.

**How to fill this in**: use the Debug Panel's live HUD (`infer:` ms, `dets:`, `p=` proximity)
and the laptop dashboard's confidence-tier history for anything requiring a count over time.
Re-run each metric at least twice to confirm reproducibility per the ticket's own test step.

| Metric | Development-time number | Held-out run 1 | Held-out run 2 | Notes |
|---|---|---|---|---|
| YOLO inference latency (ms/frame) | TBD — measure via Debug Panel `infer:` | TBD | TBD | Note which backend (CPU/NNAPI/QNN) was measured |
| Depth inference latency (ms/frame) | TBD | TBD | TBD | Runs every-2nd-frame; note fresh vs reused frame |
| End-to-end frame latency (ms) | TBD | TBD | TBD | |
| Object detection confidence (avg, real objects) | Known from offline testing: 0.34–0.57 on real photos with yolo26s | TBD (live) | TBD (live) | |
| Drop-off detection: true positives / real hazards presented | TBD | TBD | TBD | Test on a REAL curb/step, not a photo |
| Drop-off detection: false positives / flat-floor trials | TBD | TBD | TBD | Should be near-zero — safety-critical number |
| Barometer cross-check: confirmed / total real descents (iQOO only) | N/A on dev phone | TBD | TBD | |
| Hazard sound: true positives / real horn-siren-alarm trials | TBD | TBD | TBD | Math validated offline; keyword/threshold NOT yet validated live |
| Hazard sound false positives / quiet-environment trials | TBD | TBD | TBD | |
| Voice goal-seeking: command → correct target found | Blocked — needs QNN native bridge live | — | — | State as blocked honestly if bridge isn't done by demo time |
| Full obstacle-course completion (see #39) | — | Clean run 1: Y/N | Clean run 2: Y/N | The #39 "done when" condition IS this row |

**Test discipline**: keep development-time numbers (you, iterating, knowing the layout) and
held-out numbers (fresh layout, someone else sets it up) visually separate — mixing them
undermines the whole point of the metric.

---

## #39 — Obstacle-Course Demo (physical — I cannot do this)

**Course** (static only — no thrown/fast objects, cut for variance):
1. Veer around a chair
2. Duck/stop for a head-height obstacle
3. Stop at a taped drop-off line
4. Climax: judge says "find the door" → volunteer is voice-vectored to it

**Checklist**:
- [ ] Physical space set up with all 4 elements
- [ ] Phone mounted (chest), charged, screen-on flag active
- [ ] Camera + mic permissions confirmed still granted
- [ ] Sonification ON
- [ ] Voice goal-seeking is currently blocked on the QNN bridge — if still not live by
      rehearsal time, decide now whether to prioritize the bridge or adjust the course/#43
      narrative to match honestly
- [ ] "Done when": two consecutive clean live runs, no manual intervention

---

## #40 — Blindfold Caveat + Real-User Feedback (draft, needs a real conversation)

**Draft caveat line** (say it before a judge points it out):

> "Our live demo today uses a sighted, blindfolded volunteer — a controlled proxy for testing
> our sensing and cueing pipeline under pressure, not a substitute for how a blind or
> low-vision person actually navigates day to day. [If you get the real conversation:] We did
> talk to [name/context] about the sonification design, and [one concrete thing that changed
> or was validated]. The honest next step before deployment is real co-design with trained
> cane users, not a one-off demo."

I can't have that conversation for you — if you reach even one blind/low-vision person before
the event, note one specific thing they said and slot it into the bracketed part.

---

## #41 — Challenge-Response Answers (draft, rehearse yourselves)

Name the competitive landscape yourself (Biped, .lumen, Seeing AI, Lookout, Soundscape,
SmartCane, The vOICe) before a judge does:

- **"How is this different from [competitor]?"** — "Most are either cloud-dependent (Seeing
  AI, Lookout — need connectivity) or sensor-only with no semantic understanding (SmartCane's
  ultrasonic). We're fully offline, on-device NPU inference, with three orthogonal
  sonification channels instead of one alert tone."
- **"What happens if the model gets it wrong?"** — "It says so. Our WHITE/BLUE/RED confidence
  tier degrades the audio texture instead of faking confidence — RED explicitly means
  'something's there, I can't identify it,' never a guessed label."
- **"Biggest known limitation right now?"** — the depth model's sign inversion on a specific
  descending-staircase framing, found and documented this session, cross-checked (not fully
  fixed) via barometer, now additionally covered by a second independent detector
  (V-disparity+RANSAC). Honest, well-documented — better than pretending nothing's imperfect.
- **"How accurate is it really?"** — point at #38's filled table. Never answer from memory.
- **"Why not use [existing app]?"** — "[X] does [narrower thing]; we target continuous,
  always-on obstacle+hazard+voice-goal navigation in one offline pipeline, not a single-purpose tool."

**Done when**: every team member can say each in one clean sentence, unread — rehearsal, not
something I can do for you.

---

## #42 — Backup Recording (physical — I cannot do this)

- [ ] Record AFTER #39's two clean runs are consistent, not before
- [ ] Show the airplane-mode toggle explicitly on camera (proves the offline claim)
- [ ] Capture the full course: avoidance → drop-off → "find the door" → arrival
- [ ] Keep it real/unedited — a staged-looking backup undermines trust more than a rough live demo
- [ ] Store it locally, reachable without internet at the venue

---

## #43 — Pitch Narrative, Both Formats (skeleton, needs your voice)

**"Prove → Measure → Imagine"** — say "Imagine" as what the system *refuses* to fake, not a
missing feature.

### Short stage format (~90s — confirm actual time budget)
1. **Hook** (10s): the specific problem — [X]M blind/low-vision people, existing tools are
   cloud-dependent or dumb-sensor-only.
2. **Prove** (30s): live demo moment — confidence tier, drop-off warning, "find the door" climax.
3. **Measure** (25s): 2–3 numbers from #38's table, spoken exactly as measured, not rounded up.
4. **Imagine, as discipline not feature** (25s): "We could have shipped a version claiming
   precise metre distances. We didn't — monocular depth can't honestly claim that precision.
   That refusal to overclaim is the actual product."

### Longer format (Q&A round / written submission)
Same four beats, expanded with: the #40 caveat, one #41 answer addressed proactively, and an
honest open-limitations list (QNN bridge status / depth staircase edge case / whatever's still
true by demo day).

**Test step**: dry-run against someone who hasn't seen the project. Confirm every spoken
number matches #38's table exactly, and the closing line lands as "our honesty discipline,"
not "what we didn't finish."

---

## What I can still help with from here
- Filling in #38's table as you report real numbers back
- Refining #40/#41/#43 drafts once you tell me what actually happens in rehearsal
- Debugging anything that breaks during course setup (camera, permissions, sonification)

**What only you can do**: walk the course, have the real conversation, record the video,
rehearse out loud.
