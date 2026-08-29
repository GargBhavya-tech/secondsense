# Ticket #1 — Ear-Occlusion / Headphone Decision (RESOLVED)

**Phase 0 · Pre-event · Discuss · Must-build before doors open**
Companion to `secondsense_build_map_v3.md` #1 and `secondsense_bible_v4.md` §6.

---

## Decision

**Option B — haptics-primary.** Execute with **wired USB-C stereo earbuds** for the event demo; name **bone conduction** as the production-version upgrade in the pitch.

### Why (one paragraph)
The safety-critical proximity signal runs on the phone's haptic motor (§5.2), so it never touches the ears — ear occlusion therefore only affects the *non-safety* audio channels (direction pan, identity icons). Wired keeps latency low for real-time cues; USB-C stereo preserves the two channels the direction-pan design (#18) needs; and wired USB-C buds are trivially sourceable in Bengaluru, unlike wired bone conduction (a rare category — almost all bone-conduction headsets are Bluetooth, which reintroduces the latency we went wired to avoid). Crucially, this answer is *demonstrable*: mute the audio and the volunteer still navigates on haptics alone.

---

## Sourcing spec (buy / confirm before the event)

| Item | Spec | Notes |
|---|---|---|
| **Primary: wired USB-C stereo earbuds** | Native USB-C plug, stereo, ideally **open-fit / hard-shell** (not sealed silicone in-ear) | Open-fit lets ambient sound in → less occlusion, still wired + stereo + cheap. Test the loaner iQOO's bundled vivo/iQOO USB-C earphones first — may need no purchase. |
| **Adapter (only if using 3.5mm buds)** | **Active DAC** USB-C→3.5mm dongle | The iQOO 15 has **no 3.5mm jack** and USB-C v3.2. A *passive* adapter may not work (no guaranteed analog-over-USB-C). Buy an **active DAC** dongle, or skip and use native USB-C buds. |
| **Optional stage prop: bone-conduction / open-ear set** | Bluetooth is fine here (Shokz OpenRun, or budget boAt/Wings open-ear) | Only to *show* the ears-free production vision on stage. BT latency is acceptable because it carries only direction/identity, not the safety-critical proximity (haptics do). Nice-to-have, not required. |

---

## Test protocol (run when hardware is in hand — maps to the ticket's Test + Done-when)

1. **Stereo + pan check.** Plug the chosen wired buds into the loaner iQOO (native USB-C, or via active DAC dongle). Play a hard-left / center / hard-right test tone. **Pass:** L/R separation is clearly audible (this is what #18 depends on).
2. **Latency check.** Tap the screen to fire a test click. **Pass:** no perceptible lag between tap and sound (rough human check; anything that feels instant is fine).
3. **Ambient-hearing check.** Wearing the buds at demo volume, have someone clap/speak behind you. **Pass:** you can still hear them. If sealed in-ear blocks too much, switch to open-fit buds and re-test.
4. **Haptics-independent check (the key one).** Mute audio entirely. Run a proximity vibration ramp (a test ramp now; #21's graded haptics once built). **Pass:** you can tell "getting closer / farther" and steer toward/away from an obstacle on vibration alone. *This is the live proof the safety channel doesn't need the ears.*
5. **Write it down.** Record the chosen buds + adapter, and paste the stage sentence below into the pitch notes.

---

## The stage sentence (say this out loud in the pitch — §6 / §10)

> "We thought about ear occlusion. Our safety-critical proximity signal runs on the phone's haptic motor, not audio — so even with headphones in, the collision-avoidance channel never touches your hearing. Audio only carries direction and identity. A production unit swaps the wired buds for bone conduction to free the ears for ambient sound too; the architecture doesn't change, only the transducer."

---

## Done-when checklist

- [ ] Wired USB-C stereo buds sourced (or loaner's bundled earphones confirmed working)
- [ ] Active DAC dongle on hand *only if* using 3.5mm buds (else N/A — native USB-C)
- [ ] Test protocol steps 1–4 all pass on the actual loaner iQOO
- [ ] Optional bone-conduction prop decided (in or out)
- [ ] Stage sentence pasted into pitch notes; whole team can say it cold
