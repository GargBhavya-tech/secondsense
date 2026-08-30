# SecondSense — The Complete Technical Bible

*Written so someone with zero prior context can read this end to end and understand exactly
what the system does, why every design decision was made, the mechanism and the numbers behind
every module, worked examples with real values, every edge case and how it's handled, and how
to independently test each piece. No code is shown — but nothing is hand-waved either.*

---

## Part 0 — What Problem Is This Solving, In Plain English

A blind or low-vision person walking through the world has a white cane. The cane is perfect at
one thing: telling you what is touching the ground in the ~1 metre arc in front of your feet.
It tells you nothing about:

- a person about to cross your path from the side,
- a step going *down* before your foot reaches the edge,
- a low branch or an open cabinet door at head height,
- which door is the exit,
- what a sign says,
- where you put your phone down five minutes ago.

SecondSense is a phone worn on the chest, camera facing forward, that fills those gaps. It
watches the world, decides what matters, and tells you through **sound and vibration** — never
the screen. You can also talk to it.

The entire design answers four questions, in order, before it ever makes a sound:

1. **What is actually in front of me?** (raw perception — objects and distance)
2. **Which one thing matters most right now, and is it a genuine hazard or just something in
   the room?** (fusion and hazard classification)
3. **Has this been true long enough to act on, or is it a one-frame flicker?** (temporal
   confirmation)
4. **Given what the user is doing — walking, sitting, on a bus — how loudly, if at all, should
   I say this?** (context routing)

And one question it treats as sacred and answers completely differently from everything else:

5. **"Is it safe to cross?"** — this must *never* be answered by a language model. Section 13
   is entirely about why, and how it's prevented.

Everything below is organised as a stack of layers. Each layer answers a narrower slice of
those questions and passes an explicit result to the next. No layer shares hidden state with
another — which is what makes each one testable on its own.

---

## Part 1 — The Full Pipeline, At a Glance

```
   Camera frame arrives  (~5–15 per second, depending on heat + context)
        │
        ▼
   LAYER 0 — Perception: two AI models
        ├─ object detector (YOLO)  → boxes + labels + scores
        └─ depth model             → a "how near/far" value per pixel
        │
        ▼
   LAYER 1 — Per-object fusion + stabilisation + confidence tier
        │     "chair, 2 m, slightly left, not approaching, WHITE tier"
        │
        ▼
   LAYER 2 — Ego-motion: how the camera itself moved this frame
        │     (needed by the hazard checks below)
        │
        ├───────────────┬──────────────────┬─────────────────────┐
        ▼               ▼                  ▼                     ▼
   LAYER 3          LAYER 3b           LAYER 3c              LAYER 3d
   Drop-off         Specular-Trap      Overhead /            Camera-health
   state machine    vetoes (cancel     head-height           (blocked / dim /
   (5 evidences)    false drops)       hazard                knocked off angle)
        │               │                  │                     │
        └───────────────┴──────────────────┘                     │
                        │  (drop-off confirmed / possible / safe) │
                        ▼                                          │
   LAYER 4 — Target selection: of everything in view, the ONE      │
            thing to cue right now (hazard > voice goal >          │
            memory bearing > nearest obstacle), habituation-gated  │
                        │                                          │
        ┌───────────────┼──────────────────────────────────────────┘
        ▼               ▼
   LAYER 5 — Output channels
        ├─ Sonification: pan = direction, pulse-rate = distance, timbre = identity
        ├─ Haptics: graded buzz patterns for each hazard class
        ├─ Earcons: a short tune on every context switch; the panic pattern
        └─ Speech: answers, warnings, sign read-outs (English or Hindi)

   Running continuously, alongside all of the above:
        • Motion sensors  → footsteps, vehicle vibration, tilt, turning
        • Barometer       → am I descending? (independent drop-off check)
        • Microphone      → siren / horn detection; voice commands; conversation ducking
        • LAYER 6  Activity context → reconfigures every layer above (6 modes)
        • LAYER 7  Thermal governor → sheds work as the phone overheats
        • LAYER 8  Voice assistant → wake, intent grammar, on-device language model
        • LAYER 9  Safety gate     → intercepts "is it safe?" before any model sees it
```

---

## Part 2 — Layer 0: Perception (the two models that look at every frame)

**One-line goal:** turn a camera frame into (a) a list of named boxes and (b) a rough distance
for every pixel.

### 2.1 The object detector — YOLO

**What it is.** YOLO ("You Only Look Once") is a neural network that takes an image and, in one
pass, outputs a set of boxes. Each box has: four numbers for its position (left, top, right,
bottom, all expressed as fractions 0–1 of the frame so they're resolution-independent), a class
label, and a confidence score 0–1.

**The class list.** It recognises the **80 "COCO" classes** — a standard set the model was
trained on: person, bicycle, car, bus, chair, couch, bench, bottle, cup, backpack, handbag,
suitcase, laptop, tv, cell phone, book, dog, potted plant, traffic light, and so on. It does
**not** know "my keys", "the light switch", or "a door" — those aren't COCO classes. The app is
explicit about this limit everywhere it matters (Section 14).

**Why YOLO and not something richer.** A captioning model could describe the scene in a
sentence, but (a) it's far too slow to run 10× per second on a phone, and (b) a blind traveller
doesn't need a paragraph — they need *where* and *what*, as fast as possible. YOLO gives exactly
that and nothing else.

**Label collapsing.** The 80 fine classes are collapsed into a smaller "icon vocabulary" that
the sound layer has bespoke sounds for:

| Collapsed label | COCO classes that map to it |
|---|---|
| `person` | person |
| `dog` | dog |
| `vehicle` | car, truck, bus, motorcycle, bicycle, train |
| `chair` | chair, couch, bench, bed, toilet |
| `furniture` | dining table, tv, laptop, refrigerator, oven, microwave |
| (raw name) | everything else — passes through as e.g. "backpack", "bottle" |

### 2.2 The depth model — Depth-Anything

**What it produces.** A **depth map**: the same size as the frame, where each pixel holds a
number meaning "how near or far this point is". Bright = one end of the range, dark = the other.

**The honest truth about it.** *You cannot measure distance from a single photo.* A photo has
no ruler. What the model does is exactly what a person does looking at a photo with one eye
closed — it reads the *cues*: things lower in the frame are usually nearer, blur means far,
converging lines mean depth, a large face is close, a small one is far. Trained on tens of
millions of images, it's very good at this. But its raw output is **relative** — "A is nearer
than B" — not "A is 2.3 metres away".

**Two versions are used:**

- The **relative** model (Depth-Anything-V2-Small) — fast, used for the live cue pipeline where
  "nearer/farther" is all that's needed, and for the laptop demo.
- A **metric indoor** model (Depth-Anything-V2-Metric-Indoor-Small) — a variant fine-tuned to
  output real metres directly, used where an actual number matters (the heavier reconstruction
  path, and the object-memory feature).

**Turning relative into a usable number.** Where a rough metre value is needed from the
relative model, the app makes an explicit, labelled assumption: normalise the relative output
so its values span 0 to 1, then map that onto **0.5 m (nearest) to 4.0 m (farthest)**. This is
a working guess, never presented as a measurement. There is also a smarter path — the
**MetricDepthScaler** — which fits a flat "this is the floor" plane to the lower part of the
frame using the phone's tilt angle, and uses the geometry of that plane to convert the relative
depth into approximate metres.

### 2.3 How often each model runs

Running both models on every frame at full resolution would overheat the phone in minutes and
drain the battery. So:

- The **object detector** runs every *N*th frame, where *N* is set by the activity context
  (every frame while walking, every 3rd–6th while sitting) and raised further by the thermal
  governor as the phone heats up.
- The **depth model** runs on its own separate cadence (every 2nd frame while walking, every
  6th–8th while sitting), and the most recent depth map is reused on the frames in between.

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| Object detector emits a one-frame ghost (a door frame briefly read as "person") | A spurious cue / beep | Layer 1's stabiliser requires a detection to persist ~3 frames before it can trigger anything |
| Depth model output is all near-identical (staring at a blank wall) | "Everything is 2 m" — no useful gradient | The drop-off checks need texture/edges to work and will report "sensors can't tell" rather than a fake reading; the camera-health monitor also flags a featureless frame |
| A COCO class the sound layer has no icon for (e.g. "toaster") | No identity sound | Falls back to a sped-up spoken word ("spearcon") — degraded but not silent |
| User asks to find something not in the 80 classes ("keys") | Endless silent searching | Grounding knows its own vocabulary — it fails fast with an honest "I can only find common things like…" (Section 14) |

### How to test this layer

Switch the engine to **MOCK** (Section 17) and confirm the whole app runs with synthetic
detections. Then on a real device: point the camera at a person and a chair, open the spectator
dashboard (Section 16), and confirm both appear in the detections list with sensible boxes and
scores, and that the depth-derived "proximity" for the nearer object is higher.

---

## Part 3 — Layer 1: Per-Object Fusion, Stabilisation, Confidence Tiers

**One-line goal:** turn "box + label + score" plus "a patch of the depth map" into a single
human-meaningful description per object, and decide how much to trust it.

### 3.1 The four fused properties

For each detected object, the app computes:

- **Direction** — from the box's horizontal centre. Centre-x near 0 → hard left; near 0.5 →
  dead ahead; near 1.0 → hard right.
- **Proximity** (0 = far, 1 = right in your face) — from the depth map sampled inside the box,
  combined with a "how much of the frame does this box fill" prior (a big box is usually
  close).
- **Rate of approach** — the box compared to recent frames: if it's growing *and* its depth is
  shrinking, it's coming toward you. A positive number means closing; negative means receding.
  This is what lets the app say "getting closer" versus "just sitting there".
- **Moving vs static** — a coarse yes/no flag from frame-to-frame motion *after* subtracting
  the camera's own motion (Layer 2). A walking person is "moving"; a parked chair is "static".

### 3.2 Stabilisation — never trust one frame

Camera frames flicker; detections wink in and out. If every flicker produced a cue, the user
would hear a stutter of false beeps. So a detection's confidence is **boosted when it's
re-seen in the same place across consecutive frames** and **held down when it appears for only
one frame**. Practically: a detection must be consistently present for about **three frames**
before it's allowed to drive a cue.

### 3.3 Confidence tiers — WHITE / BLUE / RED

Every cue that reaches the user carries a tier, driven by how strong and steady the evidence
is:

- **WHITE** — solid. Strong detection score, steady across frames, depth agrees.
- **BLUE** — plausible but shakier (weak score, or the depth and the box disagree a bit).
- **RED** — a confirmed hazard (a confirmed drop-off, an overhead object).

The tier is shown to a sighted helper as a colour on the dashboard, and the app also tracks the
*recent history* of the tier — a steady run of WHITE reads very differently from a WHITE/BLUE/
RED flicker even at the same instant.

### Worked example

You walk toward a sofa. Frame by frame:

- Frames 1–2: the detector reports "couch, score 0.55" — appears, but only briefly, box jitters
  → held down, no cue.
- Frames 3–5: "couch, score 0.72", box stable in the same spot, depth patch says proximity
  0.4 → stabiliser boosts it, tier = WHITE, direction = slightly right (box centre-x ≈ 0.62),
  approach = +0.05 (slowly closing).
- Result handed up: *"couch — proximity 0.4 — slightly right — approaching slowly — WHITE"*.

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| Detection score high but depth patch is garbage (reflective surface) | Confident but wrong distance | Tier drops to BLUE when box size and depth disagree |
| Two objects overlapping in the frame | Depth patch mixes both | Proximity is sampled from the box centre and lower region, which usually belongs to the nearer/front object |
| Object at the very edge of the frame, half-visible | Direction and size unreliable | Edge boxes are de-weighted in target selection (Layer 4) |

### How to test

On the dashboard, watch the tier bar while walking toward and away from a fixed object — it
should be a stable WHITE when the object is clearly in view, and flicker or drop to BLUE when
you point the camera away so the object is only half-seen.

---

## Part 4 — Layer 2: Ego-Motion (how the camera itself moved)

**One-line goal:** measure how the whole scene shifted between this frame and the last, so the
hazard checks can tell "the world moved because *I* turned" apart from "that thing moved".

### The mechanism

The app picks a set of trackable spots on the ground and near the frame edges (corners,
textured points) and follows them from the previous frame to the current one using **optical
flow** (Lucas–Kanade — a standard method that, for each tracked point, solves for the small
shift that best lines up the little patch of pixels around it). It then fits a single
translation + rotation that explains most of those shifts. That's the camera's own motion for
this frame.

**Why it matters:** the drop-off checks and the Specular-Trap "is this floor moving like
floor?" veto both need to know the camera's motion to work. If you're standing still, several
of those checks disable themselves (no motion → no signal to analyse).

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| Whip-pan (fast head turn) | Optical flow loses tracking, garbage motion estimate | Points that don't agree with the majority shift are thrown out; if too few agree, motion is marked "unknown" and motion-dependent checks skip this frame |
| Textureless floor (polished concrete) | Nothing to track on the ground | Falls back to edge-region points; if still too few, the coplanarity veto (3b) reports "can't tell" rather than a fabricated result |
| Moving crowd fills the frame | The "majority shift" could follow the crowd, not the camera | Ground-region points are weighted more heavily since the floor is rigid relative to the camera |

### How to test

Stand still and confirm the estimated ego-motion is near zero. Pan the phone slowly left and
confirm the estimate reports a consistent rightward scene shift of roughly the right size.

---

## Part 5 — Layer 3: Drop-Off Detection (the scary one)

**One-line goal:** detect a step *down* — a curb, stairs, a platform edge, a missing floor —
*before* your foot reaches the edge, without crying wolf.

### Why this is hard

In a photo, "floor" and "a hole where the floor should be" can look almost identical — a
horizontal line with different-looking stuff below it. Shadows, puddles, and shiny marble
produce that exact signature (Section 6 handles those). And a false "STOP — cliff!" that turns
out to be a shadow will make the user stop trusting the device.

### The five independent lines of evidence

Every frame (when the hazard pipeline is on), the app gathers:

1. **Edge-lattice analysis.** A real step-down creates a specific kind of strong horizontal
   edge, with a texture change above versus below it. The app scores how "step-edge-like" the
   strongest horizontal edge in the lower frame is.
2. **Depth evidence.** Sampling the depth map across that candidate edge: the ground should
   jump from "near" (the floor you're on) to "far" (the lower level) right at the edge. The
   depth verdict is one of **SUPPORTS**, **CONTRADICTS**, or **UNRELIABLE**.
3. **Ground-plane analysis.** The app fits a flat plane to the lower-centre of the frame ("this
   is the floor I'm standing on") and checks whether the region beyond the candidate edge
   breaks out of that plane downward.
4. **Object suppression.** If the detector sees an object sitting exactly on that edge (a rug
   edge, a threshold strip, a doormat), the edge is probably a harmless material boundary, not
   a cliff — this *reduces* the drop evidence.
5. **The barometer.** Air pressure rises as you descend. This sensor **cannot** be fooled by
   light or texture. It's slow and noisy, so it never fires the warning *by itself*, but a
   confirmed pressure descent **upgrades** a "possible" to a "confirmed".

### The state machine

The evidence feeds a state machine with five states: **SAFE**, **POSSIBLE DROP**, **CONFIRMED
DROP**, **SENSOR BLOCKED**, **PATH NOT TRAVERSABLE**. The rules that move between them:

- **→ POSSIBLE DROP** after drop evidence appears in **2 of the last 3 frames**.
- **→ CONFIRMED DROP** after *strong* evidence (a clear step edge **and** a SUPPORTS depth
  verdict) in **3 of the last 5 frames**.
- **A weak-but-present signal keeps it at POSSIBLE** rather than clearing it — it will never
  silently drop from CONFIRMED back to SAFE without saying so.
- **PATH NOT TRAVERSABLE** if the whole forward view is blocked/untextured (a wall right in
  front, a covered camera).
- **SENSOR BLOCKED** if the camera-health monitor (3d) says the lens is covered.

### What fires

- **CONFIRMED DROP** (rising edge only — the moment it's first confirmed) → the escalating,
  impossible-to-ignore haptic pattern, at **maximum intensity if the barometer also
  confirmed**. RED tier.
- **POSSIBLE DROP** → one subdued haptic pulse, rate-limited to at most once every 1.5 seconds
  so a flickering "maybe" can't machine-gun the motor.
- These fire **regardless of the sound settings** — a drop-off warning is not something you can
  accidentally mute.

### Worked example

You approach a single step down into a sunken living room.

- Frame 10: the strong horizontal edge at the floor line scores high on "step-edge-like". Depth
  across it is UNRELIABLE (low light). Ground-plane says the far region dips slightly. Evidence
  = weak-present. State: SAFE → (1 of 3) still SAFE.
- Frame 11–12: same edge, ground-plane dip clearer. Evidence present 2 of 3 → **POSSIBLE
  DROP**. One subdued buzz.
- Frame 13: you've taken half a step; the barometer registers a small pressure rise →
  `descendingConfirmed`. This upgrades POSSIBLE → **CONFIRMED DROP**. The full escalating buzz
  fires, before your weight shifts onto the front foot.

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| A truck passes, briefly creating a strong horizontal edge | One-frame false "drop" | Needs the evidence in 2 of 3 (possible) / 3 of 5 (confirmed) frames — a one-frame spike never confirms |
| Descending a ramp (no edge, but you ARE going down) | Vision sees no step; user is still descending | The barometer path catches the sustained descent and raises at least POSSIBLE |
| Standing at the top of stairs, not moving | Ground-plane / flow checks need motion | Vision edge + depth still work stationary; the flow-based Specular veto is simply skipped |
| A dark doormat at a threshold | Looks like "different stuff below the edge" | Object-suppression (evidence line 4) reduces the drop score when a detection sits on the edge; the Specular colour veto (6.1) also checks for a real material change |

### How to test

Force the states with the dashboard's evidence readout. Physically: walk slowly toward a single
step down in good light and confirm you feel the POSSIBLE pulse a stride before the edge and the
CONFIRMED pattern as you reach it. Then repeat toward a painted line on flat ground and confirm
nothing fires.

---

## Part 6 — Layer 3b: The Specular-Trap Vetoes (cancelling false drop-offs)

**One-line goal:** when a **puddle**, a hard-edged **shadow**, or **polished marble** produces
the exact visual signature of a drop-off, cancel the warning — using physics the trick of light
can't fake.

These are **vetoes**: their only job is to *downgrade* a drop-off warning, never to create one.

### 6.1 Colour analysis — the shadow veto

**The physics.** A cast shadow makes the ground *darker* but keeps its *colour* (hue) the same
— shadowed beige carpet is still beige. A real edge between two materials (carpet → concrete →
empty air) almost always changes the colour too.

**The mechanism.** The app samples a band of pixels just above and just below the candidate
edge, inside the walkable corridor. For each pixel it computes three "which colour channel
dominates" angles (roughly: `angle1 = arctan(red ÷ max(green, blue))`, and similarly for green
and blue). It then compares:

- the **brightness step** across the edge (is it much darker below?), and
- the **hue step** across the edge (did the dominant-colour angles change?).

A **shadow likelihood** is produced from *(large brightness step) AND (small hue step)*. It
starts counting at a brightness drop of about 12% and saturates by about 40%, and it needs the
hue change to be below a small threshold (about 0.12 radians) to count as "no colour change".

**What it does:** a shadow likelihood at or above **0.6** downgrades a **CONFIRMED DROP** to
**POSSIBLE DROP**. It is deliberately hue-based so that a genuinely dark-but-real material (dark
slate tiles) still shows a hue edge and is *not* wrongly vetoed.

### 6.2 Motion analysis — the flat-surface veto

**The physics.** As you walk, a flat floor — wet, glossy, mirror-polished, whatever — moves in
the camera in a smooth, predictable, *coplanar* way. A real drop has a break in that motion at
the edge.

**The mechanism.** The app tracks known-good floor points (from the region it's confident is
floor) from the previous frame to the current one, and fits a simple 2-D transform (an affine
fit, solved with basic linear algebra) that describes how the floor moved. It then checks the
points in the band *beyond* the candidate edge: do their actual motions match what that
floor-transform predicts (within about 1.1 pixels)? The fraction that match is the **inlier
fraction**. A coplanarity confidence is computed as `(inlier_fraction − 0.55) ÷ 0.35`, clamped
to 0–1 — so you need better than 55% of the far-side points moving like the floor before it
starts vetoing, and 90%+ to veto fully.

**What it does:** coplanarity at or above **0.6** *clears* a **POSSIBLE DROP** back to **SAFE**
and downgrades a **CONFIRMED DROP**. It only runs when the camera is actually moving (it needs a
minimum ego-motion of about 0.004 in normalised units) — standing still, it's skipped.

### 6.3 The barometer backstop (re-escalation)

A veto could, in principle, wrongly cancel a *real* drop (imagine a real staircase down that
happens to be in shadow). So: if a warning has been held at POSSIBLE by a veto **and** the
barometer independently confirms you're descending, that overrides the veto and treats it as
CONFIRMED. Physics wins over photometry.

### Worked example

Sunlight through a window throws a crisp-edged shadow across a corridor floor.

1. Layer 3 flags a POSSIBLE drop (strong horizontal edge, darker below).
2. Colour veto: brightness step across the edge ≈ 35% (large), hue angles change by ≈ 0.03
   radians (tiny) → shadow likelihood ≈ 0.8 → ≥ 0.6.
3. Motion veto: you're walking; 88% of the points beyond the "edge" move exactly like the
   floor transform → coplanarity ≈ `(0.88 − 0.55) / 0.35` ≈ 0.94 → ≥ 0.6.
4. Barometer: no pressure change.
5. Both vetoes fire → POSSIBLE is cleared to **SAFE**. You walk through the shadow with no
   false alarm.

Ten metres on there's a real step down, also partly shadowed. Colour veto: brightness step
large, but the hue angles *also* shift (carpet → concrete) → shadow likelihood ≈ 0.3, below
0.6, no veto. Motion veto: the far-side points do *not* track the floor transform → coplanarity
≈ 0.1, no veto. The warning stands and escalates.

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| A real drop that is entirely in shadow | Shadow veto could suppress it | The motion veto still sees the non-coplanar edge; and the barometer re-escalation backstop catches the descent |
| Standing still at a puddle | Motion veto can't run | Colour veto still works stationary; if both vetoes are unavailable the warning is simply *not* downgraded (fail toward caution) |
| Very dark scene | Colour angles are noisy | Low-light frames are flagged by camera-health; the vetoes report low confidence and don't downgrade |

### How to test

Needs real footage: walk toward (a) a hard-edged shadow on flat ground — confirm no lasting
drop warning; (b) a shallow puddle on flat ground — confirm no lasting drop warning; (c) a real
step down that happens to be shadowed — confirm the warning still fires. This is the one part of
the system with zero field validation so far.

---

## Part 7 — Layer 3c/3d: Overhead Hazard, Habituation, Camera Health

### 7.1 Overhead / head-height hazard

**Why:** the white cane sweeps the ground and misses anything at head height — a branch, an
open cabinet door, a scaffold pole, a shelf corner.

**The rule:** any detected object whose **box centre is in the upper 40% of the frame** *and*
whose **proximity is ≥ 0.45** counts as an overhead hazard. It fires a **distinct** haptic
pattern (different from the drop pattern), **only on the rising edge** (the moment it first
appears), so it doesn't hammer the motor every frame while the object stays in view. Fires
regardless of sound settings.

### 7.2 Habituation — don't nag about static things you're near

**Why:** if you stop next to a wall, the app can see the wall is close. It should say so once,
then be quiet — you know it's there. Endless "wall! wall! wall!" gets the headphones ripped
out.

**The rule (the ObstacleHabituation filter):** once a static obstacle has been cued, it's
**muted** unless something changes:

- you start **approaching** it (rate of approach rises past ~0.06), or
- it gets **noticeably closer** (proximity rises past a ~0.06 delta), or
- you turn so it's at a **different bearing** (past a ~0.16 azimuth gate), or
- it becomes **imminent** (proximity ≥ 0.85 — very close), which always un-mutes.

A newly-announced obstacle is held (not re-cued) for about 2.5 seconds regardless. Safety
hazards — drop-offs, overhead — **bypass this entirely**; those always fire.

**Worked example:** you pause at a bus stop, shoulder ~0.5 m from the shelter pole. One cue:
"pole, close, left." Then silence. You shift your weight and drift toward it — rate of approach
crosses 0.06 → the cue returns.

### 7.3 Camera health — is the lens usable?

**Why:** the whole vision stack is blind if the lens is covered, pointed at the ceiling, or
knocked sideways in the harness.

**The checks (CameraHealthMonitor), each requiring the condition to hold for ~1.5 seconds
before it's announced, and ~0.9 seconds of good frames before "recovered" is announced:**

- **BLOCKED** — frame mean brightness below ~20 (near-black) **and** standard deviation below
  ~7 (featureless). → "Camera is blocked. Please clear it."
- **DIM** — mean brightness below ~42. Not announced as an alarm; perception is eased off.
- **MISALIGNED** — the phone's tilt is more than ~14° off vertical in pitch or more than ~20°
  in roll — **but only judged once the wearer has done a one-time "hold it vertical and tap
  Calibrate" step**, so it's measured against how *they* set it up, not a guess.

While the camera is BLOCKED or MISALIGNED, every vision-derived cue goes silent and the app
leans on the barometer and microphone.

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| Walking from bright outdoors into a dark room | Momentary "BLOCKED" | The 1.5-second sustain requirement rides through the transition |
| Wearer bends down to tie a shoe | Big transient tilt → "MISALIGNED" | Sustain requirement; and it only judges tilt at all after the calibration step |
| Lens smudged but not covered | Not dark, not featureless → not "BLOCKED" | This is the Specular/defocus case — the drop-off checks report low-confidence rather than a fake reading; a dedicated blur check is a known gap |

### How to test

Cover the lens with a hand for 2 seconds → expect the spoken "camera is blocked"; uncover →
expect "camera is okay now" and a buzz. Tap Calibrate while holding the phone vertical, then
tilt it 30° down → expect "camera has moved".

---

## Part 8 — Layer 4: Target Selection (the one thing to cue)

**One-line goal:** of everything the app perceives this frame, pick the **single** thing to
turn into a sound right now. One clear cue beats three competing ones.

### The priority order

1. **An active safety hazard** — a confirmed/possible drop-off or an overhead object always
   wins. (These also fire their own haptics independently.)
2. **A live voice goal** — if you asked it to "find a chair" and a chair is visible, steering
   you to that chair outranks cueing a random obstacle.
3. **A memory bearing** — if you asked "where's my phone" and it's not in view yet, the coarse
   remembered direction is cued until the phone re-enters the camera (then priority 2 takes
   over — the "visual handoff").
4. **The nearest relevant obstacle** — the closest thing you might walk into, after the
   habituation filter (7.2) has had its say.

Plus a **context-independent safety floor**: no matter the mode, no matter if cues are muted or
paused, if any object is **proximity ≥ 0.90 and rate-of-approach ≥ 0.12** (very close and
closing fast), you get one haptic, at most once every 3 seconds. Nothing turns this off.

### Worked example

You're being guided to a "chair" (priority 2). A person walks quickly across your path,
reaching proximity 0.92, approach 0.2. The safety floor fires one haptic immediately. The chair
cue continues (the person isn't a *hazard state*, just close) — but if the person triggered a
confirmed overhead or drop hazard, priority 1 would take the cue channel.

### How to test

Set a find goal, then walk toward an unrelated obstacle, and confirm the cue stays on the goal
(not the obstacle) until you either reach the goal or a real hazard interrupts.

---

## Part 9 — Layer 5: The Output Channels

**One-line goal:** deliver the chosen cue as sound and vibration a person can act on in under a
second.

### 9.1 Sonification — space as sound

Speech is precise but slow ("a person is approaching from your two-o'clock at three metres"
takes 4 seconds). So the continuous cue is **abstract sound** with three independent
dimensions:

- **Direction = stereo pan.** Left object → left ear, ahead → centred, right → right ear. An
  **equal-power** pan law is used (left gain = cos, right gain = sin of the pan angle) so it
  feels like a position in space, not a lopsided volume.
- **Distance = pulse rate.** Far: a slow *blip … … blip*. Close: fast pulses, like a reversing
  truck. Distance is **never** carried by pitch.
- **Identity = timbre.** Person, wall, a named goal — each has its own short signature sound
  ("auditory icon" / "spearcon"). Pitch is reserved for identity, never distance.

Keeping the three on separate channels means a close object on your left is *fast pulses, in
the left ear, with the wall timbre* — three facts, one sound, no interference.

### 9.2 Haptics — graded buzz patterns

| Event | Pattern |
|---|---|
| Possible drop | one subdued pulse, ≤ once per 1.5 s |
| Confirmed drop | escalating, unmissable, on the rising edge; max intensity if barometer-confirmed |
| Overhead | a distinct short pattern, different from the drop pattern, rising edge only |
| Path blocked / sensors can't tell | its own pattern — "I don't know" is never silent |
| Safety floor (very close + closing) | one buzz, ≤ once per 3 s, cannot be disabled |

### 9.3 Earcons — a tune per context switch

On every activity-mode change (Section 10) a short, distinct melody plays so you know the mode
instantly without waiting for the spoken line:

| Mode | Tone shape | Meaning |
|---|---|---|
| Walking | two notes **rising** (~520 → ~720 Hz) | "go" |
| Standing | one firm mid note (~600 Hz) | "stopped, still watching" |
| Home | two notes **falling** (~560 → ~460 Hz) | "relaxed, room memory" |
| Sitting | one low note (~330 Hz) | "settled" |
| Transit | a soft three-note chime (~494 / 622 / 740 Hz) | "in a vehicle" |
| Conversation | one very short low tick (~300 Hz) | "quiet" |

The **panic** earcon is a fast ~990 Hz triple-beep repeated twice — unmistakable.

### 9.4 Speech — and conversation ducking

Speech is used for: answers to voice questions, sign read-outs, hazard *escalations* the user
asked about, and the honest "I can't tell" lines. When the microphone hears nearby human
speech, the app **quiets its cues** so you can hear the person you're talking to, and comes
back up when the talking stops.

### How to test

Put on headphones. Have a helper move a phone-detected object left→right in front of the camera
and confirm the cue pans left→right in your ears, and that the pulse speeds up as they walk it
toward the camera. Switch modes and confirm each earcon is recognisably different.

---

## Part 10 — Layer 6: The Activity-Context System

**One-line goal:** a blind person's needs while **walking a street**, **sitting on a sofa**,
and **riding a bus** are completely different — reconfigure the whole app for each.

### The six modes and exactly what each sets

| Mode | Continuous cue loop | Hazard pipeline | Object-detector cadence | Depth cadence | Sign-reading + faces | Spoken verbosity |
|---|---|---|---|---|---|---|
| **Walking** | ON | ON | every frame | every 2nd | off | normal |
| **Standing** | off | ON | every 2nd | every 4th | off | minimal |
| **Home** | off | ON | every 3rd | every 4th | ON | minimal |
| **Sitting** | off | **OFF** | every 4th | every 6th | ON | silent |
| **Transit** | off | **OFF** | slow | every 6th | ON | silent |
| **Conversation** | off | **OFF** | slow | every 8th | ON | silent |

**Why "hazard OFF" in Transit is not just a battery choice:** on a moving bus, the vehicle's
motion and vibration make the drop-off checks see phantom "drops" constantly. Running them
there is *harmful*, not just wasteful.

**The safety floor (Layer 4) still runs in every mode**, including the "OFF" ones. A mode tunes
sensitivity and verbosity; it never removes the last line of defence.

### How it detects the mode by itself (ContextAutoDetector)

Polls every ~2 seconds:

- **Walking** — a footstep was detected in the last **1.4 seconds**. Footsteps come from the
  accelerometer: the up-down bounce is low-pass filtered to track gravity, the residual is
  taken, and a step is counted on each upward swing past **1.8 m/s²** that clears a **280 ms**
  refractory gap. No special permission needed. Stride is assumed ~0.72 m.
- **Transit** — no footsteps, but a **sustained vibration energy**. The app keeps a slow
  running average of the *squared* residual acceleration (`vibration += 0.05 × (residual² −
  vibration)`). Standing still this sits near ~0.02; a moving bus/car floor parks it around
  0.2–1.5. If, with **no steps for 4 seconds** (so footstep energy has decayed), the vibration
  is **≥ 0.22**, it guesses Transit; otherwise Standing.
- **Sitting / Home / Conversation** — the motion sensors genuinely cannot tell a chair from
  your feet, a known room from an unknown one, or a face-to-face chat. These are **only ever
  set by you** (voice or gesture).

### Why it won't flip-flop, and why your word wins

- **The 15-second grace.** An automatic guess only takes effect after the *same* guess has held
  continuously for **15 seconds**. A 3-second blip (you paused at a curb) changes nothing.
- **The 5-minute sticky window.** The moment *you* set a mode (say "I'm sitting", or swipe),
  the app obeys instantly and then **ignores its own sensors for 5 minutes**. Fidget all you
  like; it won't argue.

### Merging with the thermal governor

Both the context and the thermal governor (Section 11) want to set the same knobs (detector
cadence, depth cadence, whether aux perception runs). They're merged by taking the **more
conservative** value for each knob — whichever wants to do *less* work wins.

### Worked example

You board a bus. The engine idles, the floor buzzes, no footsteps. For 15 seconds nothing
changes (the grace window). Then: the transit earcon chimes, and it says *"Feels like a
vehicle. Hazard alerts off, sign reading on."* Now it's quietly reading the route number over
your shoulder instead of hallucinating a drop-off every time the bus lurches. You get off and
start walking; after 15 seconds of steps it chimes the walking earcon and full guidance
resumes.

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| Walking on a vibrating surface (a moving train platform) | Vehicle vibration + steps at once | Steps win — `walking` is checked first; you're still walking even if the ground shakes |
| You stop at a red light for 20 s | Auto-detect flips to Standing, cues go quiet | Correct behaviour; the grace makes it a deliberate 15 s, not instant, and one step re-arms walking |
| You say "I'm sitting" then get up and walk | Sensors would say "walking" | The 5-minute sticky window ignores that; if you actually want to walk, say "let's go" |

### How to test

Walk for 20 s → expect the walking earcon and full guidance. Sit still on a chair for 20 s →
expect it to *not* auto-switch (Sitting is user-only) but Standing after 15 s of stillness. Say
"I'm on the bus" → expect the transit earcon immediately and hazard cues to stop.

---

## Part 11 — Layer 7: The Thermal Governor

**One-line goal:** a phone strapped to a 37 °C body with no airflow overheats in 15–20 minutes;
a throttled phone processes frames ~5× slower; a slow drop-off warning is an extra 20–30 cm of
walking before you hear it. Keep the safety loop fast by shedding everything else first.

### The four signals it fuses

Each is mapped to a tier (nominal / warm / hot / critical) and the **worst** of the four wins:

1. **Android's thermal status** — the OS's own verdict (none → light → moderate → severe).
   Most authoritative.
2. **Thermal headroom** — the OS's forecast, 0 to 1+, where >0.9 means throttling is imminent.
   Mapped: ≥0.98 → critical, ≥0.93 → hot, ≥0.85 → warm.
3. **Battery temperature** — the "skin" proxy: ≥46 °C → critical, ≥44 → hot, ≥42 → warm.
4. **The app's own measured speed** — see the two tricks below.

### Trick 1 — ignore the warm-up

The AI chip is genuinely slow for the first ~90 frames after a cold start (~200 ms/frame) —
that's warm-up, not overheating. So the speed signal is **ignored entirely** until three
90-frame windows have been measured.

### Trick 2 — the speed check is *relative*, not a fixed number

Our AI path genuinely runs at ~5 frames per second on a good day. An absolute "slower than
X ms → panic" threshold would be meaningless. Instead the governor **learns this specific
phone's steady-state speed**: during a moment when the other three signals all say "cool", it
records the current 90th-percentile frame time as the **baseline**, and only ever ratchets that
baseline *down* (never up — so a throttled reading can't poison it). Then:

- current p90 > **2× baseline** → hot
- current p90 > **1.5× baseline** → warm
- otherwise → nominal

### What gets shed, in order

As the tier rises, work is dropped least-critical-first:

1. Extra perception (sign reading, face detection) off.
2. The microphone hazard-sound listener off.
3. Camera resolution dropped (e.g. to 320×240) — the models want small inputs anyway.
4. Fewer frames processed (every 2nd, 3rd…).

**But a safety floor is always kept:** even at *critical* while walking, it still processes a
frame every 3rd and runs depth every 6th — the core "is there a drop-off" loop never fully
stops.

### Behaviour

- **Escalation is immediate** (get cautious fast).
- **De-escalation waits 20 seconds** of sustained calm (don't oscillate).
- It tells the user honestly, once: *"Device is warm. Running in power save — updates a little
  slower,"* and later *"Cooled down. Back to full speed."* (These lines are translated to Hindi
  if that's the preference.)

### Worked example

Minute 0: cold. p90 frame time settles at 205 ms after warm-up → recorded as the baseline.
Minute 18: the phone is hot from body heat. Android thermal status = light, headroom = 0.90 →
warm. Battery 43 °C → warm. p90 has crept to 340 ms ≈ 1.66× baseline → warm. Worst-of = warm →
policy: aux perception off, hazard-sound listener off. Minute 25: headroom 0.94, p90 430 ms ≈
2.1× baseline → hot → drop to 320×240 and every-2nd-frame, but depth still every 4th, hazard
loop intact. Minute 40: back in the shade, all signals cool for 20 s → step back to warm, then
nominal; "back to full speed".

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| Cold start reads as "throttled" | False downshift in the first seconds | The 90-frame ×3 warm-up guard ignores the speed signal until then |
| Baseline recorded during a brief throttle | Permanently pessimistic | Baseline only ratchets *down*, and is only recorded when the other three signals all say cool |
| Rapid warm/cool cycling near a threshold | Policy oscillation | Immediate up, 20-second-hysteresis down |

### How to test

Force each tier with the OS thermal override and confirm the policy changes as described and
the spoken notice fires once. The real proof — a 15–20 minute harness soak showing frame time
stays roughly flat instead of doubling — is not yet done.

---

## Part 12 — Layer 8: The Voice Assistant

**One-line goal:** let the user just *talk* — "what's ahead", "find a chair", "read that",
"I'm sitting" — with a fast deterministic path for the common cases and a small on-device
language model for the rest.

### 12.1 Waking it

- **One tap anywhere** → a beep + a buzz confirm listening → speak.
- **Single tap in the top third** → "what's around me" (offline, no mic).
- **Double tap anywhere** → same "what's around me".
- **Long-press** → spoken settings menu (long-press the bottom-right corner → the sighted-
  helper Admin panel).
- **Two-finger tap** → pause / resume.
- **Three-finger tap** → read the gesture help.
- **Volume up** → repeat the last spoken line. **Volume down** → cycle loudness.
- **Both volume keys held 2 s** → panic (Section 12.4).
- If the phone's own screen-reader is on, the custom gestures are suppressed and three big
  labelled buttons are shown instead.

### 12.2 The fast path — the intent grammar

A hand-written rule set recognises ~15 intents instantly, with **no AI model**:

`find X` · `where did I leave my X` · `what's ahead` · `read this` · `status / battery / where
am I` · `repeat` · `cues on/off` (incl. "stop the beeping", "mute", "be quiet") · `pause /
resume` · `stop looking` · `switch mode` ("I'm sitting", "getting on the bus", "let's walk") ·
`speak Hindi / English` · `call <name>` · `set a timer for <time>` · `help`.

It's **context-aware**: a bare word like "keys" is taken as "find keys" only when you're
actively navigating (Walking / Standing / Home) and the phrase is ≤ 4 words; while sitting or
in conversation, a stray word is treated as chatter and passed on. A stop-list prevents verbs
("walk", "go", "stop") from ever being taken as a find *target*.

Rule order matters — it checks help, repeat, language, context-switch, pause/resume, cues,
cancel, read, recall, describe, status, phone tasks, safety questions (Section 13), explicit
"find X", then bare noun. Anything unmatched becomes `Unknown` and goes to the language model.

### 12.3 The reasoning path — Gemma on the phone

`Unknown` transcripts go to **Gemma-3-1B-IT** (a 1-billion-parameter language model, 4-bit
quantised, ~530 MB) running **entirely on the phone** via Google's MediaPipe LLM runtime. It's
handed the transcript **plus a snapshot of what the app perceives** — *"you're walking; ahead:
door on the left, person ahead; hazard: none; battery 82%; camera ok"* — and asked to either
map the request onto one of the known actions (returning a single line of JSON like
`{"action":"find","target":"elevator"}`) or answer the question directly
(`{"action":"say","text":"..."}`).

The prompt is wrapped in Gemma's required chat format (without it, the 1B model replies in the
third person or just echoes the prompt). The reply parser is lenient — it finds the first
`{...}`, tolerates surrounding prose, and if there's no JSON at all it just speaks the raw
text.

**Why a tiny model on the phone, not a big cloud one:** offline, private, instant, free.
**Its honest limits:** a 1B model is not clever. It sometimes echoes the question or gives a
vague answer. It's a helper for the long tail; the grammar does the real work. On device, we
confirmed it's unreliable enough that it must **never** touch safety — Section 13.

### 12.4 The panic gesture

Both volume keys pressed together (within 500 ms of each other) and held → after 2 seconds:
the panic earcon fires, a buzz, and it speaks *"Emergency. Battery X percent. \<mode\> mode.
Say who to call, or stay quiet to cancel."* Then it opens the microphone; "call mom" routes
through the normal contact-dial path. Nothing dials automatically; silence cancels. Releasing
either key before the 2 seconds cancels.

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| "find the hindi newspaper" | The word "hindi" could flip the app to Hindi | Language-switch needs a verb cue ("speak", "switch to"); a bare mention is treated as a find target |
| "can I walk" | Bare-noun path could make it `find "walk"` | Verbs are on the never-a-target stop-list; and it's caught as a safety question first (Section 13) |
| Speech recogniser mishears badly | Wrong action | Most mishears become `Unknown` → the LLM sees the scene context and usually asks for a repeat or gives a safe generic; a blank transcript → "I didn't catch that" |
| LLM returns `{"action":"call","name":"<contact>"}` (parroting the prompt) | Would try to call a contact literally named "<contact>" | The parser rejects any value that looks like a placeholder (`<...>`, "the thing", "name") |

### How to test

Run the 78-phrase demo sweep test (Section 19) — it maps realistic phrasings to expected
actions and fails loudly on any misroute. On device: tap, say each of the 15 grammar intents,
confirm the right thing happens. Say something open ("what room is this?") and confirm it
reaches Gemma and gets a scene-grounded answer.

---

## Part 13 — Layer 9: The Safety Gate ("is it safe to cross?")

**This is the most carefully engineered part of the system.** Read the whole section.

### The problem

If a blind person asks a language model "can I walk?" or "is the path clear?", the model —
trained to be helpful and agreeable — will very often say **"yes, you can go"**, *even when the
app's own sensors are reporting a drop-off*. Small models are especially prone to this
("sycophancy"). An assistive device that confidently green-lights someone into traffic is a
catastrophe. And keyword filtering doesn't work — there are infinite ways to ask, in every
language.

### The solution: the language model is never allowed to answer

A safety question is intercepted **before** it reaches the grammar or Gemma, and answered by a
**deterministic, pre-written response built from the actual sensor state**. Six layers make
sure nothing slips through:

#### Layer A — a learned classifier catches the question in any wording or language

A tiny model — trained on a laptop, then baked into the app as a **37 KB table of numbers**
(no download, no separate file) — reads the **shape of the characters** in what you said. The
mechanism, precisely:

1. Lower-case the text, pad with a space each side.
2. Take every **3-character and 4-character substring** ("n-gram"). For "is it safe" that's
   " is", "is ", "s i", … and " is ", "is i", "s it", ….
3. Hash each n-gram (MD5, first 4 bytes, taken modulo **4096**) to pick one of 4096 buckets;
   set that bucket to 1.
4. Scale the 4096-long vector to unit length.
5. Multiply by 4096 learned weights, add a learned bias, squash through a logistic curve to get
   a probability 0–1.
6. **≥ 0.50 → this is a safety question.**

It was **trained** on ~100 safety questions in **English, Hindi (both scripts), and Kannada**
versus ~90 non-safety phrases (find/read/status/call/chatter), with filler-word augmentation.
Because the features are character patterns, not a dictionary, it spans scripts. On held-out
phrases it scored **100% — zero false negatives, zero false positives** — and the threshold is
deliberately set low so it errs toward *catching too much*: a false "that's a safety question"
just gives the safe deflection, which is a fail-safe outcome.

#### Layer B — an obvious-phrasing fast path

A short list ("is it safe to cross", "can I walk", "is the path clear", "anything in my way")
takes an even faster route, and three regex families catch the common shapes
(`<safe/ok/clear> to <cross/walk/go>`, `<can/should/may> I <cross/walk/go>`,
`<anything/obstacle/hazard> ... <ahead/in my way>`). This is an *optimisation*, not the
guarantee.

#### Layer C — the deterministic answer never says "safe" as a yes

It fills one fixed template:

> *"I cannot decide whether it is safe to move. The sensors show \[current state\], but you
> must rely on your white cane, traffic sounds, and your own judgement to proceed."*

`[current state]` comes from the real hazard pipeline: *"there is a drop-off directly ahead"* /
*"there may be a step or drop ahead"* / *"the path ahead looks blocked"* / *"the camera cannot
see clearly"* / *"the sensors detect: person on your right, book ahead"* / *"no objects are
detected ahead"*. There is a fully-translated Hindi version. This wording is deliberate — it
hands responsibility back to the person and their cane, which is exactly what mobility
instructors teach and what keeps the product defensible.

#### Layer D — a 1.5-second memory (hysteresis)

If a drop-off flickered into view *during* the moment you asked, the answer still reports it.
The app takes the **worst hazard state seen in the last 1.5 seconds**, not just the
instantaneous one. Better over-cautious.

#### Layer E — even if something reaches Gemma, its reply is scanned

If Gemma's answer contains a movement green-light — "you can go", "it's clear", "path is
clear", a bare "yes" near "walk/cross" — that answer is **thrown away** and replaced with the
deterministic deflection. This check runs on the model's **own output** (a small, fixed set of
phrases), not on your unbounded input, so it's reliable.

#### Layer F — Gemma is also told to refuse

Its prompt says: *"you are an objective environment describer, you have no hazard sensors, you
must never give safety verdicts; if asked about safety, reply with a deflection flag."* Not
trusted alone (1B models don't reliably follow instructions), but one more layer.

### Worked example

You're at a kerb. You ask *"is it okay to walk?"*

1. Layer A: the n-gram classifier scores 0.94 → safety question.
2. The grammar and Gemma **never see it**.
3. Layer D: the 1.5-second hazard memory holds "possible drop ahead" (it flickered as you
   stepped up to the kerb).
4. Layer C: *"I cannot decide whether it is safe to move. The sensors show there may be a step
   or drop ahead, but you must rely on your white cane, traffic sounds, and your own judgement
   to proceed."*

It will never, under any circumstances, just say "yes".

### A confirmed real-world finding

On the actual device, Gemma-1B answered *"You see nothing there"* to *"is anything in my way?"*
while a book, a person, and furniture were all detected. This is exactly why Layers A–F exist
and why the language model is fenced off from this question entirely.

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| Novel phrasing the classifier hasn't seen | Could slip to the LLM | The LLM prompt (F) + the output green-light veto (E) catch it downstream |
| A transient "sensors blocked" during the 1.5 s window | Answer says "camera can't see" when it's actually fine | Accepted — erring conservative is the design |
| Non-safety question with the word "safe" ("is my data safe?") | Wrongly deflected | Rare, and a false deflection is harmless (fail-safe); the classifier's negatives include such cases |

### How to test

`NgramSafetyGate` has unit tests: safety phrases in three scripts must be flagged, `call mom`
/ `find keys` / blank must not. On device: ask "can I walk", "is anything in my way", "should
I move", "are there obstacles ahead" — every one must give the deterministic deflection with
the cane caveat, and none may get a yes/no from Gemma.

---

## Part 14 — Layer 10: Finding Things

### 14.1 Live find — "find a chair", "take me to the door"

You say a target. Each frame, the app matches that word against the current detections. If it
matches, the object becomes a **steering cue**: pan toward it, pulses speed up as you close in,
and when it's both **centred (within ~0.12 of frame centre) and close (proximity ≥ 0.75)** you
hear *"you've reached the chair"* and feel the "arrived" haptic.

**Synonym bridging.** The detector emits labels like "backpack", "chair", "cell phone". People
say "bag", "sofa", "phone". A synonym table maps the spoken word to the set of labels that
should satisfy it:

| You say | Matches detector label(s) |
|---|---|
| bag / purse / luggage | backpack, handbag, suitcase |
| sofa / couch / seat / stool | chair, couch |
| phone / mobile | cell phone |
| tv / laptop / computer / table / desk / fridge | furniture (+ the raw name) |
| bike / cycle / car / bus / truck | vehicle |
| water | bottle |
| person / people / man / woman / someone | person |

**The "how long does it search" rule** (this was tuned after user feedback that it gave up too
soon):

- If the word is one it **cannot ground at all** ("keys", "door", "remote control") → after
  **6 seconds** it says so honestly: *"I can't look for a \<X\>. I can only find common things
  like people, chairs, bags, bottles, laptops and phones. Say never mind to stop."*
- If the word **is groundable** but hasn't been seen yet → at **14 seconds**, one nudge:
  *"Still looking for the \<X\>. Turn around slowly so I can see more."* (The goal is **not**
  dropped.)
- Only at **40 seconds** with still no sighting does it give up.

### 14.2 Memory recall — "where did I leave my phone?"

The app keeps a short-term memory of objects it saw you **set down** — detected by a "settled
object" check that waits for both the object and the camera to go still. It tracks your rough
position by **dead reckoning**: counting steps (× stride ~0.72 m) along the heading at each
step, from the phone's motion sensors. There is **no GPS and no map**, so this **drifts** ~5–10%
of the distance travelled — good for about **one room over about a minute**. On "where's my
phone", it speaks a coarse bearing (*"about three steps behind you, on your left"*) and steers
you there until the phone re-enters the camera, at which point the live find (14.1) takes over
— the "visual handoff". Ask about something from an hour ago and it says it has no memory of
it.

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| "find a bag" with a backpack in view | Old code said "can't find a bag" while listing "bags" as findable | Synonym table now matches backpack/handbag/suitcase; the contradictory message is gone |
| You turn away from the object during the search | It leaves the frame | Groundable targets search for 40 s with a "turn slowly" nudge, so a brief look-away doesn't end it |
| Dead-reckoning drift over a long walk | "Your phone is here" points at a wall | The feature is honest about its ~1-room / ~1-minute horizon and hands to visual as soon as it can |

### How to test

Put a backpack on a chair, walk away, say "find a bag" → expect it to steer you back and say
"arrived". Say "find my keys" → expect the 6-second honest "I can only find common things…".
Set a phone down, walk two rooms away, say "where's my phone" → expect a coarse bearing (and
expect it to be roughly, not exactly, right).

---

## Part 15 — Layer 11: Sign Reading + Translation

### The mechanism

- **OCR** (optical character recognition) reads text from the frame. Two recognisers —
  **Latin** (English) and **Devanagari** (Hindi) — run on **alternating frames**. Only text
  whose box centre is in the central region (roughly 22–78% across, 12–88% down) is read, and
  the same text isn't re-read within ~4 seconds.
- **On-device translation.** If a sign's script isn't your listening language, it's translated
  locally before being spoken. The Hindi↔English model pair downloads **once over Wi-Fi**
  (~30 MB each) and is fully offline after. If it isn't downloaded, the sign is read in its
  printed script.
- **Language preference** (English or Hindi) is set by voice ("speak Hindi") or the menu. All
  spoken feedback then comes in that language — the app routes its own English status lines
  through the translator too, so you don't get English words in a Hindi voice.
- **"Read this"** grabs the current frame and reads whatever text is on it right then, even in
  a mode where continuous sign-reading is off.

### Edge cases

| Edge case | What would go wrong | How it's handled |
|---|---|---|
| A wall of small text | Every line read aloud, endlessly | Only central text; ~4 s cooldown per distinct string |
| Translation model not downloaded | Silence, or a crash | Falls back to reading the native script |
| Mixed-script sign | One recogniser misses half | The two alternate, so over a couple of seconds both scripts are covered |

### How to test

Point at an English EXIT sign with Hindi selected → expect it spoken in Hindi (if the model is
downloaded) or read in English (if not). Say "read this" at a room-number plate → expect the
number spoken once.

---

## Part 16 — Side Systems

### 16.1 The persistent room map (experimental, parked)

**Idea:** walk your room once, slowly; the app builds a memory of the layout — furniture
positions, clear-floor area, where you keep things — anchored to a reference spot, saved to a
file. Next time home, it re-aligns and can route you around the sofa to the kitchen.

Uses the phone's **augmented-reality tracking** (a proper position-in-metres, unlike drifty
step-counting) plus the depth sensor to drop detected objects and floor/wall patches into a
grid. **Four of five steps are built** (position tracking, world-placed objects, the floor/
obstacle grid, save + reload + re-anchor). **Step five — actually using the map to route you —
is not built.** Known rough edges: it double-counts the same object as tracking drifts; there's
no visual re-localisation, so re-alignment is only as good as you re-standing on the same spot.
**Parked** — a direction, not a feature.

### 16.2 The laptop 3D demo (for judges to watch)

A separate laptop program. The phone publishes its camera frames as small images on a tiny
built-in web address; the laptop pulls them and builds a **live, growing, full-colour 3-D point
cloud of the room** as you pan.

The trick: for each frame, run a depth model → turn it into a cloud of coloured 3-D dots (each
pixel → a dot at its estimated position, painted its real colour) → **line up each new frame's
dots with the ones already on screen** (a method called ICP — it slides and rotates one cloud
until it best overlaps the other) → merge. Because the depth is relative, a small per-frame
correction keeps every cloud the same size so they stitch instead of piling up. A heavier path
(metric depth + proper visual odometry + surface fusion into a solid mesh) exists for when a
graphics card is available.

*Status:* works end to end against the live phone — a two-minute sweep produced a coherent,
recognisably room-shaped coloured cloud (~5 × 4 × 3.5 m). On a laptop with no graphics card the
depth model is the bottleneck (~1 frame/second) and the result is soft rather than crisp.

### 16.3 The spectator dashboard

A plain web page the phone serves on the local network. A helper or judge opens it in a
browser and sees, live: the current cue, the confidence tier (as a big room-visible colour
bar), whether a drop-off is flagged, the detections list, the inference speed, the thermal
state. Read-only; no effect on the phone; no internet.

---

## Part 17 — How the Software Is Put Together (so it can be built and tested)

### 17.1 Swappable "engines"

The part that runs the AI models is behind one boundary with three interchangeable
implementations:

- **MOCK** — invents plausible detections. The whole app (sound, haptics, UI, logic) runs with
  no models and no phone.
- **TFLITE** — runs the models on the phone's regular processor. Works on any Android device
  and a plain emulator. The reliable test path.
- **QNN** — runs the models on the phone's dedicated **AI chip** (Hexagon NPU). Much faster
  and more power-efficient — the real deployment path. Getting it working meant fixing five
  separate low-level bugs in how the app talks to the chip (missing library declarations, a
  changed way the chip exposes its interfaces, a struct-layout version mismatch, tensor
  binding by numeric ID, and a data-type widening). It falls back to TFLITE/MOCK automatically
  if anything fails, so a normal build never crashes. (One subtlety: the chip's models expect
  the image data in a specific channel order — feeding the wrong one silently tanks accuracy.)

Everything *above* the engine boundary — all the fusion, the hazard state machine, the cue
generation — is **model-agnostic**. Swapping engines changes speed and power, not behaviour.

### 17.2 Tested by pure logic

The tricky decision-making — target selection, the drop-off state machine, the safety-question
classifier, the intent grammar, the context manager, the earcon shapes, the goal-grounding
synonyms — is written as **pure functions with no Android dependencies**, unit-tested on a
laptop in milliseconds. There are ~**118** such tests. One runs **78 realistic phrases a judge
might say** through the intent grammar and fails loudly on any misroute.

### 17.3 Build flags

- default build: no on-device LLM (a stub), no MediaPipe — smaller.
- `-PenableLlm=true`: compiles the Gemma path + bundles the MediaPipe runtime (~120 MB). The
  ~530 MB model file is **side-loaded** to the phone, never bundled.
- `-PenableQnnNative=true`: compiles the AI-chip bridge against the Qualcomm SDK.

---

## Part 18 — Full Edge-Case Reference Table

| Failure condition | Why it's hard | How SecondSense handles it |
|---|---|---|
| One-frame ghost detection | A single bad frame would beep | Stabiliser requires ~3 consecutive frames (Part 3.2) |
| Partial info: object half off-frame | Direction/size unreliable | Edge boxes de-weighted in target selection (Part 8) |
| Camera turned by the user vs. object moved | Both shift the pixels | Ego-motion subtracted first (Part 4) |
| Shadow / puddle / marble looks like a cliff | Identical visual signature to a real drop | Three physics vetoes — colour, coplanar motion, barometer (Part 6) |
| A veto wrongly cancels a *real* drop | Vetoes could over-suppress | Barometer re-escalation overrides any veto on a confirmed descent (Part 6.3) |
| One-frame spike (truck, light flicker) | Constant false hazards | 2-of-3 / 3-of-5 frame confirmation in the state machine (Part 5) |
| Descending a ramp (no visible edge) | Vision sees nothing | Barometer catches the sustained descent (Part 5) |
| Standing near a wall | Endless "wall! wall!" | Habituation filter mutes static, non-approaching obstacles (Part 7.2) |
| Lens covered / knocked off angle | Whole vision stack is blind but silent | Camera-health monitor announces it; vision cues go silent (Part 7.3) |
| Sitting still, full pipeline running | Wasted battery + noise | Context modes cut cadence, cues, verbosity (Part 10) |
| Drop-off detection on a moving bus | Vehicle motion = phantom drops | Transit mode turns the hazard pipeline OFF (Part 10) |
| App guesses your context wrong | It goes quiet when you want guidance (or vice versa) | One word overrides for 5 minutes; auto-guesses need 15 s of agreement (Part 10) |
| Phone overheats in the harness | Everything slows, warnings lag | Thermal governor sheds least-critical work first, keeps a safety floor (Part 11) |
| Governor false-triggers on cold-start slowness | Unnecessary downshift | 90-frame ×3 warm-up guard; relative (not absolute) speed baseline (Part 11) |
| Language model asked "is it safe to cross?" | It will say "yes" over a real drop-off | The question is intercepted before any model sees it; 6 layers of defence (Part 13) |
| LLM parrots a prompt placeholder | Would act on "<contact>" literally | Parser rejects placeholder-looking values (Part 12) |
| "find my keys" (not a known class) | Endless silent search | Grounding fails fast at 6 s with an honest capability list (Part 14.1) |
| "find a bag" with a backpack visible | Label mismatch → false "can't find" | Synonym table bridges spoken words to detector labels (Part 14.1) |
| Find gives up too soon | User loses a findable object over a brief look-away | Groundable targets search 40 s with a 14 s "turn slowly" nudge (Part 14.1) |
| Dead-reckoning drift on "where's my X" | Points at the wrong spot after a long walk | Honest ~1-room / ~1-minute horizon; hands to visual ASAP (Part 14.2) |
| Sign wall of text | Reads every line forever | Central text only, ~4 s per-string cooldown (Part 15) |
| Translation model not downloaded | Silence | Falls back to the printed script (Part 15) |
| English status line in the Hindi voice | Sounds broken | Status lines routed through the translator when Hindi is selected (Part 15) |
| Paused, but hazard talk / sign reading continues | "Pause" doesn't actually pause | The frame pipeline returns early when paused — only the un-mutable safety-floor haptic survives |
| Multi-finger tap not registering | 3 fingers land/lift slower than 1 | Looser time (750 ms) + drift (130 px) windows for multi-finger gestures |
| AI-chip bridge fails on a device | App would crash | Automatic fallback to the plain-processor path |

---

## Part 19 — How to Test the Whole System

### Unit tests (laptop, milliseconds)

Run the app's test suite. It covers, as pure logic: the drop-off state machine, the Specular
vetoes, the target selector, the context profiles + manager (sticky/grace), the auto-detector's
vehicle-vibration classifier, the intent grammar (including the 78-phrase judge sweep), the
n-gram safety classifier (three scripts), the LLM prompt/parse contract + green-light veto, the
goal-grounding synonyms, the earcon shapes, the perf policy, the camera-health thresholds, the
dead-reckoner, the object-memory geometry, the obstacle-habituation filter.

### On-device manual pass

1. **Perception** — point at a person + a chair; both appear on the dashboard with sane boxes;
   the nearer one has higher proximity.
2. **Drop-off** — walk slowly toward one step down in good light → POSSIBLE pulse a stride out,
   CONFIRMED pattern at the edge. Toward a painted line → nothing.
3. **Specular** (needs real scenes) — toward a hard-edged shadow / a puddle on flat ground → no
   lasting warning; toward a shadowed real step → warning still fires.
4. **Overhead** — hold a box at head height, close → the distinct overhead buzz, once.
5. **Habituation** — stand 0.5 m from a wall → one cue, then silence; drift toward it → cue
   returns.
6. **Camera health** — cover the lens 2 s → "camera is blocked"; uncover → "okay now".
7. **Context** — walk 20 s → walking earcon + full guidance; "I'm on the bus" → transit earcon
   + hazard cues stop; stand still 20 s → Standing.
8. **Thermal** — force each OS thermal tier → policy changes as in Part 11; spoken notice once.
9. **Voice grammar** — say each of the 15 intents → the right thing happens.
10. **LLM** — "what room is this?" → a scene-grounded spoken answer.
11. **Safety gate** — "can I walk", "is anything in my way", "should I move" → the deterministic
    deflection every time; never a yes/no.
12. **Find** — "find a bag" (backpack in view) → steered to it, "arrived"; "find my keys" → the
    6-second honest capability line.
13. **Recall** — set a phone down, walk two rooms, "where's my phone" → a coarse (roughly
    right) bearing.
14. **Sign reading** — English sign, Hindi selected → spoken in Hindi (model downloaded) or read
    in English (not).
15. **Gestures** — top-third tap → "what's around"; double-tap → same; two-finger tap → pause
    (and confirm hazard talk actually stops); three-finger tap → help; both volume keys 2 s →
    panic flow.

### Still-needed field validation (not yet done)

Real puddle/shadow/marble footage; a 15–20 minute thermal soak; a real walk + bus/car ride to
tune the vehicle-vibration threshold; a hands-on session with an actual blind user for the
whole UI.

---

## Part 20 — Glossary (plain-English definitions of every term used above)

- **YOLO** — a neural network that finds and names objects in an image in one fast pass,
  outputting a box + label + confidence for each.
- **COCO classes** — the standard list of 80 everyday object types YOLO was trained on.
- **Depth map** — an image where each pixel's value means "how near or far this point is".
- **Relative depth** — "A is nearer than B", with no real-world unit. **Metric depth** — an
  actual distance in metres.
- **Proximity (in this app)** — a 0-to-1 closeness score (1 = right in your face), from the
  depth map plus box size.
- **Rate of approach** — how fast an object is closing the distance to you; positive = closing.
- **Ego-motion** — how the camera itself moved between two frames.
- **Optical flow** — for each tracked point, the small pixel shift that best lines up its
  surroundings between two frames.
- **State machine** — a set of named states (SAFE, POSSIBLE, CONFIRMED…) with explicit rules
  for moving between them, so behaviour is predictable.
- **Hysteresis / dwell** — requiring a condition to hold for a minimum time (or looking back
  over a recent window) before acting, so a flicker doesn't trigger anything.
- **Veto** — a check whose only job is to *cancel* a warning, never to raise one.
- **Coplanar** — lying in the same flat plane; here, "the far side of the edge is moving like
  the floor, so it *is* the floor".
- **Hue** — the colour itself (red-ness, blue-ness), separate from how bright it is. Shadows
  change brightness but not hue.
- **Sonification** — representing data (here, space) as non-speech sound.
- **Earcon** — a short, distinctive musical motif standing for an event ("you switched to
  walking mode").
- **Equal-power pan** — a stereo left/right balance law that keeps the *total* loudness
  constant as a sound moves across, so it reads as a position, not a volume dip.
- **Habituation** — deliberately stopping repeated alerts about something the user has already
  acknowledged and isn't approaching.
- **Dead reckoning** — estimating your position by adding up your steps and turns from a known
  start, with no external reference (so it drifts).
- **Thermal headroom** — the phone OS's own forecast of how close it is to throttling, 0 to 1.
- **p90 (90th-percentile) frame time** — the value that 90% of recent frames were faster than;
  a robust "typical worst case" speed measure.
- **Intent grammar** — a hand-written rule set that maps spoken phrases to a fixed set of
  actions, with no AI.
- **Grounding** — matching a spoken word ("chair") to an actual detected object in the frame.
- **Sycophancy (of language models)** — the tendency to agree with the user's implied premise
  ("it looks clear, right?") instead of being accurate.
- **n-gram** — a run of *n* consecutive characters (or words). Character n-grams let a
  classifier work across scripts without a dictionary.
- **Feature hashing** — turning text into a fixed-length vector by hashing each n-gram to one
  of a fixed number of slots. Compact, no vocabulary file.
- **Logistic regression** — the simplest "weigh the evidence and squash to a probability"
  classifier: multiply features by learned weights, add a bias, apply an S-curve.
- **ICP (Iterative Closest Point)** — a method that aligns two clouds of 3-D points by
  repeatedly nudging one to best overlap the other.
- **MediaPipe LLM runtime** — Google's on-device framework for running a quantised language
  model on a phone.
- **Quantised (4-bit) model** — a model whose numbers are stored at low precision to shrink it
  and speed it up, at a small accuracy cost.
- **TFLite / QNN / MOCK** — the three interchangeable back-ends for running the vision models:
  plain processor / dedicated AI chip / fake-for-testing.

---

## Part 21 — Status: What Is Proven, What Still Needs a Real Test

| Area | State |
|---|---|
| Object detection + depth on the AI chip | Working on the reference phone |
| Per-object fusion, stabilisation, tiers | Working; unit-tested |
| Ego-motion | Working |
| Drop-off state machine | Built + unit-tested |
| Specular-Trap vetoes | Built + unit-tested; **no real puddle/shadow/marble footage tested** |
| Overhead hazard, habituation | Built + unit-tested |
| Camera health + calibrate workflow | Built; needs field use |
| Target selection + safety floor | Built + unit-tested; verified on device |
| Sonification / haptics / earcons | Built; needs hands-on feel-testing |
| Activity context (6 modes) | Built + unit-tested |
| Context auto-detection | Built; **needs a real walk + bus/car ride to tune the threshold** |
| Thermal governor | Downshift verified live; **full 15–20 min soak proof not done** |
| Voice intent grammar (~15 intents) | Built, unit-tested, verified on device |
| On-device language model (Gemma) | Runs on device; confirmed weak — long-tail helper only |
| Safety gate (Part 13) | Built + tested; the learned classifier scores 100% on held-out phrases in 3 languages |
| Panic gesture | Built; needs feel-testing |
| Find (grounding + synonyms + timeouts) | Built + unit-tested |
| Object memory ("where's my X") | Built; in-memory only, ~1-room / ~1-min horizon by design |
| Sign reading + Hindi/English translation | Built; translation needs its one-time model download |
| Persistent room map | Steps 1–4 built; step 5 (routing) not built; **parked** |
| Laptop 3D demo | Works end to end; soft on a CPU-only laptop |
| Blind-first UI (gestures, spoken menu, pause) | Built; the whole thing needs a session with an actual blind user |

**Not attempted (out of scope):** a native on-chip speech-recognition + open-vocabulary
object-finding bridge (weeks of low-level work — the app currently uses the phone's built-in
speech recogniser and the fixed 80-class list); a trained neural classifier for "is that
specifically a staircase" (needs a dataset and training pipeline — the geometric + depth +
motion + barometer fusion covers drop-offs without it).

---

## Part 22 — The Philosophy, One More Time

Every unusual choice traces to the same commitments:

- **It runs offline** — a navigation aid that needs a signal will fail someone at the worst
  moment.
- **Many small independent checks that vote**, not one big model — because a single model
  fooled by a shadow is dangerous, and five checks that disagree are safe.
- **It degrades honestly** — "I can't tell" is a valid, frequent answer.
- **The AI is never the last word on safety** — the most dangerous question gets the most
  deterministic, most conservative, most human-agency-preserving answer, and the language model
  is physically fenced off from it.
- **The blind person is in control** — the app watches, suggests, and gets out of the way. It
  guesses your context but obeys your correction instantly and then trusts you. It never takes
  the white cane out of the loop — it just tells you what the cane can't reach.
