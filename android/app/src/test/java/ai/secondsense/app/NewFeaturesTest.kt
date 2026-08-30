package ai.secondsense.app

import ai.secondsense.app.inference.BBox
import ai.secondsense.app.inference.ConfidenceTier
import ai.secondsense.app.inference.Detection
import ai.secondsense.app.inference.FrameResult
import ai.secondsense.app.inference.CameraHealth
import ai.secondsense.app.inference.SettledSighting
import ai.secondsense.app.inference.decode.CameraHealthMonitor
import ai.secondsense.app.inference.decode.DetectionStabilizer
import ai.secondsense.app.inference.decode.DepthVerdict
import ai.secondsense.app.inference.decode.HazardFusion
import ai.secondsense.app.inference.decode.HazardState
import ai.secondsense.app.inference.decode.RawEvidence
import ai.secondsense.app.inference.decode.RestingStateVerifier
import ai.secondsense.app.memory.DeadReckoner
import ai.secondsense.app.memory.MemoryPhrase
import ai.secondsense.app.memory.ObjectMemory
import ai.secondsense.app.context.AppContext
import ai.secondsense.app.context.ContextAutoDetector
import ai.secondsense.app.context.ContextManager
import ai.secondsense.app.context.ContextProfile
import ai.secondsense.app.voice.IntentInterpreter
import ai.secondsense.app.voice.LlmPrompt
import ai.secondsense.app.voice.LlmResolution
import ai.secondsense.app.voice.SafetyAnchors
import ai.secondsense.app.voice.SafetyGate
import ai.secondsense.app.voice.SceneBrief
import ai.secondsense.app.voice.VoiceIntent
import ai.secondsense.app.perf.PerfPolicy
import ai.secondsense.app.perf.ThermalTier
import ai.secondsense.app.sonification.CueTarget
import ai.secondsense.app.sonification.ObstacleHabituation
import ai.secondsense.app.voice.SceneNarrator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

private fun det(label: String?, cx: Float, prox: Float, score: Float) = Detection(
    label = label, score = score,
    box = BBox(cx - 0.05f, 0.4f, cx + 0.05f, 0.6f),
    proximity = prox, tier = ConfidenceTier.WHITE,
)

/** Accuracy: a consistently re-seen detection is boosted; a first sighting is held down. */
class DetectionStabilizerTest {
    @Test fun firstSightingHeldDownThenBoostedOnceConfirmed() {
        val s = DetectionStabilizer()
        val d = det("chair", 0.5f, 0.5f, 0.50f)
        val f1 = s.update(listOf(d)).first().score
        assertTrue("first frame should be nudged down, was $f1", f1 < 0.50f)
        s.update(listOf(d)); s.update(listOf(d))
        val f4 = s.update(listOf(d)).first().score
        assertTrue("confirmed track should be boosted, was $f4", f4 > 0.50f)
    }

    @Test fun oneFrameFlickerIsForgotten() {
        val s = DetectionStabilizer()
        s.update(listOf(det("dog", 0.5f, 0.5f, 0.6f)))
        repeat(4) { s.update(emptyList()) }               // gone for > maxMissed frames
        val again = s.update(listOf(det("dog", 0.5f, 0.5f, 0.6f))).first().score
        assertTrue("re-appearance starts as a fresh (held-down) track", again < 0.6f)
    }
}

/** Standout: offline scene description assembled from a FrameResult, no LLM. */
class SceneNarratorTest {
    private fun frame(dets: List<Detection>, hazard: HazardState? = null) = FrameResult(
        detections = dets, frameWidth = 640, frameHeight = 480, inferenceMillis = 10L,
        hazardState = hazard,
    )

    @Test fun emptySceneIsHonest() {
        assertEquals("Nothing detected ahead", SceneNarrator.describe(frame(emptyList())))
        assertEquals("Scene not ready", SceneNarrator.describe(null))
    }

    @Test fun nearestObjectsWithDirectionAndDistance() {
        val s = SceneNarrator.describe(frame(listOf(
            det("person", cx = 0.2f, prox = 0.75f, score = 0.9f),
            det("chair", cx = 0.85f, prox = 0.3f, score = 0.8f),
        )))
        assertTrue(s, s.contains("person very close to your left"))
        assertTrue(s, s.contains("chair a few steps away to your right"))
    }

    @Test fun dropOffIsAnnouncedFirst() {
        val s = SceneNarrator.describe(frame(
            listOf(det("chair", 0.5f, 0.4f, 0.8f)),
            hazard = HazardState.DROP_CONFIRMED,
        ))
        assertTrue(s, s.startsWith("drop-off ahead"))
    }
}

/** Object memory: dead-reckoning geometry round-trips a placed object back to a bearing. */
class DeadReckonerTest {
    @Test fun objectStraightAheadThenWalkPastItIsBehindYou() {
        val start = DeadReckoner.Pose(x = 0f, z = 0f, headingDeg = 0f)
        // Seen 3 m dead ahead.
        val (wx, wz) = DeadReckoner.placeObject(start, distanceM = 3f, bearingDeg = 0f)
        assertTrue(abs(wx) < 1e-3f); assertEquals(3f, wz, 1e-3f)

        // Walk 5 m forward (same heading) -> object is now 2 m behind.
        val after = DeadReckoner.Pose(x = 0f, z = 5f, headingDeg = 0f)
        val (range, bearing) = DeadReckoner.relativeTo(after, wx, wz)
        assertEquals(2f, range, 1e-3f)
        assertTrue("expected ~180 (behind), got $bearing", abs(abs(bearing) - 180f) < 1f)
    }

    @Test fun turningRightPutsAForwardObjectOnYourLeft() {
        val start = DeadReckoner.Pose(0f, 0f, 0f)
        val (wx, wz) = DeadReckoner.placeObject(start, 2f, 0f)     // 2 m ahead
        val turnedRight = DeadReckoner.Pose(0f, 0f, 90f)          // now facing +X
        val (_, bearing) = DeadReckoner.relativeTo(turnedRight, wx, wz)
        assertTrue("object should be ~90° to the left, got $bearing", abs(bearing + 90f) < 1f)
    }

    @Test fun stepIntegrationAdvancesAlongHeading() {
        val dr = DeadReckoner()
        repeat(4) { dr.onStep(headingDeg = 0f, strideMeters = 0.75f) }
        val p = dr.pose(0f)
        assertEquals(3f, p.z, 1e-3f); assertTrue(abs(p.x) < 1e-3f)
    }
}

/** Object memory: only a settled object is emitted; a moving one is rejected. */
class RestingStateVerifierTest {
    private fun d(cx: Float, cy: Float = 0.5f) = Detection(
        label = "bottle", score = 0.8f,
        box = BBox(cx - 0.04f, cy - 0.06f, cx + 0.04f, cy + 0.06f),
        proximity = 0.5f, tier = ConfidenceTier.WHITE,
    )

    @Test fun stationaryObjectSettlesAfterWindow() {
        val v = RestingStateVerifier(minFrames = 5, minWindowMs = 300L)
        val all = mutableListOf<SettledSighting>()
        var t = 1_000L
        repeat(6) { all += v.update(listOf(d(0.5f) to 2.0f), t); t += 100L }
        assertEquals(1, all.size)
        assertEquals("bottle", all.first().label)
    }

    @Test fun movingObjectNeverSettles() {
        val v = RestingStateVerifier(minFrames = 5, minWindowMs = 300L)
        val all = mutableListOf<SettledSighting>()
        var t = 1_000L; var cx = 0.2f
        repeat(8) { all += v.update(listOf(d(cx) to 2.0f), t); t += 100L; cx += 0.06f }
        assertTrue("a panning object must not be logged", all.isEmpty())
    }
}

/** Object memory: store then recall gives a sane bearing + coarse phrase. */
class ObjectMemoryGeometryTest {
    @Test fun rememberThenRecallFromSamePoseIsAhead() {
        val mem = ObjectMemory()
        val pose = DeadReckoner.Pose(0f, 0f, 0f)
        mem.remember(SettledSighting("keys", distanceM = 2.5f, bearingDeg = 10f), pose, nowMs = 0L)
        val hit = mem.recall("keys", pose, nowMs = 1_000L)!!
        assertEquals(2.5f, hit.distanceM, 0.05f)
        assertTrue(abs(hit.bearingDeg - 10f) < 1f)
        assertEquals(1_000L, hit.ageMs)
    }

    @Test fun fuzzyLabelMatchAndMissReturnsNull() {
        val mem = ObjectMemory()
        val pose = DeadReckoner.Pose(0f, 0f, 0f)
        mem.remember(SettledSighting("bottle", 2f, 0f), pose, 0L)
        assertTrue(mem.recall("water bottle", pose, 0L) != null)   // fuzzy
        assertNull(mem.recall("umbrella", pose, 0L))
    }

    @Test fun phraseIsCoarseNotNumeric() {
        val hit = ObjectMemory.Hit("cup", distanceM = 3.4f, bearingDeg = -80f, ageMs = 20_000L)
        val p = MemoryPhrase.build("cup", hit)
        assertTrue(p, p.contains("a few steps away"))
        assertTrue(p, p.contains("left"))
        assertTrue(p, p.contains("just now"))
        assertTrue("must not read a raw number", !p.contains("3.4"))
    }
}

/** Habituation: a static obstacle you aren't approaching stops cueing; a worsening one doesn't. */
class ObstacleHabituationTest {
    private fun t(az: Float = 0.5f, prox: Float = 0.6f, approaching: Float = 0f) =
        CueTarget(azimuth = az, proximity = prox, label = "wall", tier = ConfidenceTier.WHITE, approaching = approaching)

    @Test fun staticObstacleGoesSilentAfterHold() {
        val h = ObstacleHabituation(alertHoldMs = 500L)
        var now = 0L
        assertTrue(h.filter(t(), walking = false, now) != null)          // first alert
        now = 400L
        assertTrue(h.filter(t(), walking = false, now) != null)          // still in hold window
        now = 900L
        assertNull("stationary + not approaching -> silence", h.filter(t(), walking = false, now))
        assertTrue(h.muted)
    }

    @Test fun approachingObstacleKeepsAlerting() {
        val h = ObstacleHabituation(alertHoldMs = 500L)
        var now = 0L
        var prox = 0.4f
        repeat(10) {
            val cue = h.filter(t(prox = prox, approaching = 0.2f), walking = true, now)
            assertTrue("must keep alerting while closing in (t=$now)", cue != null)
            now += 300L; prox = (prox + 0.05f).coerceAtMost(0.95f)
        }
        assertTrue(!h.muted)
    }

    @Test fun turningToADifferentObstacleReAlerts() {
        val h = ObstacleHabituation(alertHoldMs = 500L)
        h.filter(t(az = 0.5f), walking = false, 0L)
        assertNull(h.filter(t(az = 0.5f), walking = false, 900L))        // habituated
        // user turns -> obstacle now well off to the side
        assertTrue("a different bearing is a new situation", h.filter(t(az = 0.1f), walking = false, 950L) != null)
    }

    @Test fun imminentObstacleKeepsAFaintPulse() {
        val h = ObstacleHabituation(alertHoldMs = 500L, imminentProximity = 0.85f, faintProximity = 0.33f)
        h.filter(t(prox = 0.92f), walking = false, 0L)
        val cue = h.filter(t(prox = 0.92f), walking = false, 900L)
        assertTrue("about-to-hit-it is never fully silenced", cue != null)
        assertEquals(0.33f, cue!!.proximity, 1e-4f)
    }
}

/** Camera health: covered lens and off-vertical angle are caught (after debounce); normal is OK. */
class CameraHealthMonitorTest {
    private val dark = FloatArray(19_200) { 5f }                       // near-black, zero variance
    private val normal = FloatArray(19_200) { if (it % 2 == 0) 90f else 150f } // mean 120, std 30

    @Test fun normalLevelFrameIsOk() {
        val m = CameraHealthMonitor()
        assertEquals(CameraHealth.OK, m.update(normal, pitchDeg = 5f, rollDeg = 3f, imuCalibrated = true, nowMs = 0L))
    }

    @Test fun coveredLensReportsBlockedAfterSustain() {
        val m = CameraHealthMonitor(sustainMs = 1_000L)
        assertEquals(CameraHealth.OK, m.update(dark, 0f, 0f, imuCalibrated = false, nowMs = 0L))
        assertEquals(CameraHealth.BLOCKED, m.update(dark, 0f, 0f, imuCalibrated = false, nowMs = 1_100L))
    }

    @Test fun offVerticalAngleReportsMisalignedWhenCalibrated() {
        val m = CameraHealthMonitor(sustainMs = 1_000L, pitchToleranceDeg = 14f)
        m.update(normal, pitchDeg = 30f, rollDeg = 0f, imuCalibrated = true, nowMs = 0L)   // tilted down
        assertEquals(CameraHealth.MISALIGNED, m.update(normal, 30f, 0f, imuCalibrated = true, nowMs = 1_100L))
    }

    @Test fun noAngleWarningUntilCalibrated() {
        val m = CameraHealthMonitor(sustainMs = 500L)
        // Same big tilt, but calibration never done -> never MISALIGNED.
        repeat(6) { i -> m.update(normal, 40f, 0f, imuCalibrated = false, nowMs = i * 400L) }
        assertEquals(CameraHealth.OK, m.update(normal, 40f, 0f, imuCalibrated = false, nowMs = 3_000L))
        // Once calibrated, the same tilt is flagged after the sustain window.
        m.update(normal, 40f, 0f, imuCalibrated = true, nowMs = 3_100L)
        assertEquals(CameraHealth.MISALIGNED, m.update(normal, 40f, 0f, imuCalibrated = true, nowMs = 3_700L))
    }

    @Test fun recoversAfterClearWindow() {
        val m = CameraHealthMonitor(sustainMs = 500L, clearMs = 500L)
        m.update(dark, 0f, 0f, imuCalibrated = false, nowMs = 0L)
        assertEquals(CameraHealth.BLOCKED, m.update(dark, 0f, 0f, imuCalibrated = false, nowMs = 600L))
        assertEquals(CameraHealth.BLOCKED, m.update(normal, 0f, 0f, imuCalibrated = false, nowMs = 700L))
        assertEquals(CameraHealth.OK, m.update(normal, 0f, 0f, imuCalibrated = false, nowMs = 1_300L))
    }
}

/** Thermal governor: the perf policy escalates monotonically and keeps a safety floor. */
class PerfPolicyTest {
    private val tiers = ThermalTier.values()

    @Test fun cadencesNeverIncreaseAsItCoolsAndNeverHitZero() {
        var prevFrame = 0; var prevDepth = 0
        for (t in tiers) {
            val p = PerfPolicy.policyFor(t, walking = true)
            assertTrue("$t frameEveryN must not decrease", p.frameEveryN >= prevFrame)
            assertTrue("$t depthEveryN must not decrease", p.depthEveryN >= prevDepth)
            assertTrue("$t: no cadence may be < 1", p.frameEveryN >= 1 && p.depthEveryN >= 1 && p.hazardEveryN >= 1)
            prevFrame = p.frameEveryN; prevDepth = p.depthEveryN
        }
    }

    @Test fun idleRelaxesAtLeastAsMuchAsWalking() {
        for (t in tiers) {
            val w = PerfPolicy.policyFor(t, walking = true)
            val i = PerfPolicy.policyFor(t, walking = false)
            assertTrue("$t idle should skip >= walking", i.frameEveryN >= w.frameEveryN)
        }
    }

    @Test fun criticalWhileWalkingKeepsTheCoreLoopAlive() {
        val p = PerfPolicy.policyFor(ThermalTier.CRITICAL, walking = true)
        assertTrue("~10fps floor", p.frameEveryN <= 3)
        assertTrue("depth still ~7fps", p.depthEveryN <= 6)
    }

    @Test fun auxIsShedBeforeYamnetWhichIsShedBeforeTheCoreLoop() {
        val hot = PerfPolicy.policyFor(ThermalTier.HOT, walking = true)
        assertTrue("OCR/face off by HOT", !hot.auxEnabled)
        assertTrue("YamNet still on at HOT", hot.yamnetEnabled)
        assertTrue("YamNet off by CRITICAL", !PerfPolicy.policyFor(ThermalTier.CRITICAL, true).yamnetEnabled)
        assertTrue("low-res only when hot", !PerfPolicy.policyFor(ThermalTier.NOMINAL, true).lowRes &&
            PerfPolicy.policyFor(ThermalTier.HOT, true).lowRes)
    }

    @Test fun nominalWalkingIsFullRate() {
        val p = PerfPolicy.policyFor(ThermalTier.NOMINAL, walking = true)
        assertEquals(1, p.frameEveryN)
        assertTrue(p.auxEnabled && p.yamnetEnabled && !p.lowRes)
    }
}

/** Specular Trap veto A: a chromaticity-shadow edge downgrades a confirmed drop, nothing else. */
class SpecularShadowVetoTest {
    private fun ev(lattice: Float, edgeY: Float, depth: DepthVerdict, shadow: Float) = RawEvidence(
        latticeScore = lattice, nearestEdgeY = edgeY, depthVerdict = depth,
        highRotation = false, lowLight = false, sensorBlocked = false, shadowLikelihood = shadow,
    )

    @Test fun castShadowDowngradesAConfirmedDropButNotARealOne() {
        val real = ev(0.7f, 0.6f, DepthVerdict.SUPPORTS, shadow = 0.10f)
        assertEquals(HazardState.DROP_CONFIRMED, HazardFusion.classifySingleFrame(real))
        val shadow = ev(0.7f, 0.6f, DepthVerdict.SUPPORTS, shadow = 0.80f)
        assertEquals(HazardState.POSSIBLE_DROP, HazardFusion.classifySingleFrame(shadow))
    }

    @Test fun vetoNeverTouchesPossibleOrSafe() {
        // strong lattice, depth contradicts -> already only POSSIBLE_DROP; veto leaves it.
        assertEquals(
            HazardState.POSSIBLE_DROP,
            HazardFusion.classifySingleFrame(ev(0.7f, 0.6f, DepthVerdict.CONTRADICTS, shadow = 0.9f)),
        )
        // weak evidence -> SAFE stays SAFE.
        assertEquals(
            HazardState.SAFE,
            HazardFusion.classifySingleFrame(ev(0.1f, 0.6f, DepthVerdict.UNRELIABLE, shadow = 0.9f)),
        )
    }
}

/** Specular Trap veto B: flow that stays coplanar with the floor => flat wet/glossy, not a hole. */
class SpecularCoplanarVetoTest {
    private fun ev(lattice: Float, depth: DepthVerdict, coplanar: Float) = RawEvidence(
        latticeScore = lattice, nearestEdgeY = 0.6f, depthVerdict = depth,
        highRotation = false, lowLight = false, sensorBlocked = false, groundCoplanar = coplanar,
    )

    @Test fun coplanarFlowDowngradesAConfirmedDrop() {
        assertEquals(
            HazardState.DROP_CONFIRMED,
            HazardFusion.classifySingleFrame(ev(0.7f, DepthVerdict.SUPPORTS, coplanar = 0.1f)),
        )
        assertEquals(
            HazardState.POSSIBLE_DROP,
            HazardFusion.classifySingleFrame(ev(0.7f, DepthVerdict.SUPPORTS, coplanar = 0.85f)),
        )
    }

    @Test fun coplanarFlowClearsAPossibleDropToSafe() {
        assertEquals(
            HazardState.POSSIBLE_DROP,
            HazardFusion.classifySingleFrame(ev(0.7f, DepthVerdict.CONTRADICTS, coplanar = 0.1f)),
        )
        assertEquals(
            HazardState.SAFE,
            HazardFusion.classifySingleFrame(ev(0.7f, DepthVerdict.CONTRADICTS, coplanar = 0.85f)),
        )
    }
}

/** Activity contexts: profiles get quieter as the user gets less active; transit kills hazard. */
class ContextProfileTest {
    @Test fun walkingIsFullSittingIsQuiet() {
        val w = ContextProfile.profileFor(AppContext.WALKING)
        assertTrue(w.sonification && w.hazardEnabled)
        val s = ContextProfile.profileFor(AppContext.SITTING)
        assertTrue(!s.sonification && !s.hazardEnabled && s.auxPerception)
    }

    @Test fun transitAndConversationDisableHazard() {
        assertTrue(!ContextProfile.profileFor(AppContext.TRANSIT).hazardEnabled)
        assertTrue(!ContextProfile.profileFor(AppContext.CONVERSATION).hazardEnabled)
    }

    @Test fun detectCadenceRelaxesAsContextGetsLessActive() {
        var prev = 0
        for (c in listOf(AppContext.WALKING, AppContext.STANDING, AppContext.SITTING, AppContext.CONVERSATION)) {
            val p = ContextProfile.profileFor(c)
            assertTrue("$c detectEveryN should not decrease", p.detectEveryN >= prev)
            prev = p.detectEveryN
        }
    }
}

/** ContextManager: user set is immediate + sticky; a sensor suggestion needs sustained agreement. */
class ContextManagerTest {
    @Test fun voiceSetIsImmediateAndStickyThenAutoResumes() {
        val m = ContextManager(stickyMs = 1000, graceMs = 100)
        var applied: AppContext? = null
        m.onContext = { c, _ -> applied = c }
        m.set(AppContext.SITTING, nowMs = 0)
        assertEquals(AppContext.SITTING, m.context)
        assertEquals(AppContext.SITTING, applied)
        m.suggest(AppContext.WALKING, nowMs = 200)
        m.suggest(AppContext.WALKING, nowMs = 500)
        assertEquals("suggestions ignored while sticky", AppContext.SITTING, m.context)
        m.suggest(AppContext.WALKING, nowMs = 1100)
        m.suggest(AppContext.WALKING, nowMs = 1250)
        assertEquals("resumes after sticky + grace", AppContext.WALKING, m.context)
    }

    @Test fun suggestionNeedsGraceOfAgreement() {
        val m = ContextManager(stickyMs = 0, graceMs = 500)
        m.suggest(AppContext.SITTING, 0)
        assertEquals(AppContext.WALKING, m.context)
        m.suggest(AppContext.SITTING, 600)
        assertEquals(AppContext.SITTING, m.context)
    }
}

/** ContextAutoDetector.classify: motion signals -> a context guess (or null when ambiguous). */
class ContextAutoDetectorTest {
    private val d = ContextAutoDetector(
        manager = ContextManager(),
        walkingSupplier = { false },
        vibrationSupplier = { 0f },
        vehicleVibration = 0.22f,
        vehicleSettleMs = 4_000L,
    )

    @Test fun stepsMeanWalkingRegardlessOfVibration() {
        assertEquals(AppContext.WALKING, d.classify(walking = true, vibration = 0f, msSinceWalking = 0))
        assertEquals(AppContext.WALKING, d.classify(walking = true, vibration = 5f, msSinceWalking = 0))
    }

    @Test fun rightAfterWalkingIsAmbiguous() {
        // step energy still decaying — don't guess, leave the context alone
        assertEquals(null, d.classify(walking = false, vibration = 5f, msSinceWalking = 1_000))
    }

    @Test fun sustainedVibrationWithoutStepsIsTransit() {
        assertEquals(AppContext.TRANSIT, d.classify(walking = false, vibration = 0.5f, msSinceWalking = 10_000))
    }

    @Test fun stillAndQuietIsStanding() {
        assertEquals(AppContext.STANDING, d.classify(walking = false, vibration = 0.03f, msSinceWalking = 10_000))
    }
}

/** IntentInterpreter: free-form transcript -> one action from the closed set. */
class IntentInterpreterTest {
    private fun i(s: String, ctx: AppContext = AppContext.WALKING) = IntentInterpreter.interpret(s, ctx)

    @Test fun findVerbAndBareNoun() {
        assertEquals(VoiceIntent.Find("keys"), i("find my keys"))
        assertEquals(VoiceIntent.Find("door"), i("take me to the door"))
        assertEquals(VoiceIntent.Find("exit"), i("where's the exit"))
        // bare noun only in a navigating context
        assertEquals(VoiceIntent.Find("chair"), i("chair", AppContext.STANDING))
        assertTrue(i("chair", AppContext.CONVERSATION) is VoiceIntent.Unknown)
    }

    @Test fun recallVsFindVsStatus() {
        assertEquals(VoiceIntent.Recall("phone"), i("where did I leave my phone"))
        assertEquals(VoiceIntent.Recall("wallet"), i("where is my wallet last seen"))
        assertEquals(VoiceIntent.Status, i("where am I"))
        assertEquals(VoiceIntent.Status, i("what mode am I in"))
    }

    @Test fun safetyFamiliesTakeTheFastPath() {
        // Whole semantic families, not a phrase list. Anything that still slips past is caught
        // downstream by the LLM deflect-flag + green-light veto — this is an optimisation.
        assertEquals(VoiceIntent.SafetyCheck, i("is it safe to cross"))
        assertEquals(VoiceIntent.SafetyCheck, i("is it okay to cross"))
        assertEquals(VoiceIntent.SafetyCheck, i("can I walk"))          // was mis-parsed as Find("walk")
        assertEquals(VoiceIntent.SafetyCheck, i("can I go now"))
        assertEquals(VoiceIntent.SafetyCheck, i("should I move"))
        assertEquals(VoiceIntent.SafetyCheck, i("is anything in my way"))
        assertEquals(VoiceIntent.SafetyCheck, i("are there any obstacles ahead"))
        assertEquals(VoiceIntent.SafetyCheck, i("is the path clear"))
        assertEquals(VoiceIntent.SafetyCheck, i("is it safe"))
    }

    @Test fun moveVerbsAreNeverFindTargets() {
        assertTrue(i("walk", AppContext.WALKING) !is VoiceIntent.Find)
        assertTrue(i("go", AppContext.WALKING) !is VoiceIntent.Find)
    }

    /**
     * Broad demo sweep: every phrase a judge might try, mapped to the family it MUST land in.
     * Fails with a full report of every misroute so they can all be fixed at once.
     */
    @Test fun demoPhraseSweep() {
        fun kind(v: VoiceIntent): String = when (v) {
            is VoiceIntent.Find -> "Find"
            is VoiceIntent.Recall -> "Recall"
            VoiceIntent.Describe -> "Describe"
            VoiceIntent.SafetyCheck -> "SafetyCheck"
            VoiceIntent.ReadText -> "ReadText"
            VoiceIntent.Status -> "Status"
            VoiceIntent.RepeatLast -> "RepeatLast"
            is VoiceIntent.Cues -> "Cues"
            VoiceIntent.Pause -> "Pause"
            VoiceIntent.Resume -> "Resume"
            VoiceIntent.CancelSeek -> "CancelSeek"
            VoiceIntent.Help -> "Help"
            is VoiceIntent.CallContact -> "CallContact"
            is VoiceIntent.SetTimer -> "SetTimer"
            is VoiceIntent.SetLanguage -> "SetLanguage"
            is VoiceIntent.SwitchContext -> "SwitchContext"
            is VoiceIntent.Unknown -> "Unknown"
        }
        // (phrase, acceptable kinds). Unknown is acceptable ONLY where the LLM is a fine fallback.
        val cases = listOf(
            // find
            "find my keys" to setOf("Find"),
            "find the exit" to setOf("Find"),
            "where is the elevator" to setOf("Find"),
            "where's the nearest chair" to setOf("Find"),
            "take me to the door" to setOf("Find"),
            "guide me to the stairs" to setOf("Find"),
            "look for a trash can" to setOf("Find"),
            "i'm looking for my phone" to setOf("Find"),
            "locate the bench" to setOf("Find"),
            "i need the bathroom" to setOf("Find"),
            // recall
            "where did i leave my wallet" to setOf("Recall"),
            "where did i put my cup" to setOf("Recall"),
            "where's my bag" to setOf("Recall", "Find"),
            // describe
            "what's ahead" to setOf("Describe"),
            "what is around me" to setOf("Describe"),
            "describe the scene" to setOf("Describe"),
            "what do you see" to setOf("Describe"),
            "what's in front of me" to setOf("Describe"),
            "look around" to setOf("Describe"),
            "tell me what's there" to setOf("Describe", "Unknown"),
            // read
            "read this" to setOf("ReadText"),
            "read the sign" to setOf("ReadText"),
            "what does it say" to setOf("ReadText"),
            "read that label" to setOf("ReadText"),
            "what's written there" to setOf("ReadText"),
            "read out the menu" to setOf("ReadText"),
            // safety
            "is it safe to cross" to setOf("SafetyCheck"),
            "is it safe to cross the road" to setOf("SafetyCheck"),
            "can i walk" to setOf("SafetyCheck"),
            "can i go now" to setOf("SafetyCheck"),
            "should i cross" to setOf("SafetyCheck"),
            "is the path clear" to setOf("SafetyCheck"),
            "is the way ahead clear" to setOf("SafetyCheck"),
            "is anything in my way" to setOf("SafetyCheck"),
            "are there obstacles ahead" to setOf("SafetyCheck"),
            "is it clear to go" to setOf("SafetyCheck"),
            "am i safe to move" to setOf("SafetyCheck"),
            // status
            "status" to setOf("Status"),
            "how am i doing" to setOf("Status"),
            "where am i" to setOf("Status"),
            "what's my battery" to setOf("Status"),
            "what mode am i in" to setOf("Status"),
            "give me a status report" to setOf("Status"),
            // repeat
            "repeat" to setOf("RepeatLast"),
            "say that again" to setOf("RepeatLast"),
            "what did you say" to setOf("RepeatLast"),
            "come again" to setOf("RepeatLast"),
            // cues
            "stop the beeping" to setOf("Cues"),
            "mute" to setOf("Cues"),
            "be quiet" to setOf("Cues"),
            "turn off the sound" to setOf("Cues"),
            "turn on cues" to setOf("Cues"),
            "start the cues" to setOf("Cues"),
            // pause / resume
            "pause" to setOf("Pause"),
            "hold on" to setOf("Pause"),
            "resume" to setOf("Resume"),
            "carry on" to setOf("Resume"),
            // cancel
            "never mind" to setOf("CancelSeek"),
            "cancel" to setOf("CancelSeek"),
            "stop looking" to setOf("CancelSeek"),
            "forget it" to setOf("CancelSeek"),
            // context
            "i'm sitting down" to setOf("SwitchContext"),
            "getting on the bus" to setOf("SwitchContext"),
            "let's start walking" to setOf("SwitchContext"),
            "i'm at home now" to setOf("SwitchContext"),
            "i've stopped" to setOf("SwitchContext"),
            "conversation mode" to setOf("SwitchContext"),
            // language
            "speak hindi" to setOf("SetLanguage"),
            "switch to english" to setOf("SetLanguage"),
            "talk in hindi please" to setOf("SetLanguage"),
            // phone
            "call mom" to setOf("CallContact"),
            "phone dad" to setOf("CallContact"),
            "call my wife" to setOf("CallContact"),
            "set a timer for five minutes" to setOf("SetTimer"),
            "timer for 30 seconds" to setOf("SetTimer"),
            "set a ten minute timer" to setOf("SetTimer"),
            // help
            "help" to setOf("Help"),
            "what can i say" to setOf("Help"),
            "what can you do" to setOf("Help"),
        )
        val misroutes = cases.mapNotNull { (phrase, ok) ->
            val got = kind(IntentInterpreter.interpret(phrase, AppContext.WALKING))
            if (got in ok) null else "  \"$phrase\"  ->  $got   (want ${ok.joinToString("/")})"
        }
        assertTrue(
            "\n${misroutes.size} misrouted demo phrases:\n${misroutes.joinToString("\n")}\n",
            misroutes.isEmpty(),
        )
    }

    @Test fun describeAndRead() {
        assertEquals(VoiceIntent.Describe, i("what's ahead"))
        assertEquals(VoiceIntent.Describe, i("describe the scene please"))
        assertEquals(VoiceIntent.ReadText, i("read this"))
        assertEquals(VoiceIntent.ReadText, i("what does the sign say"))
    }

    @Test fun controlVerbs() {
        assertEquals(VoiceIntent.Cues(on = false), i("stop the beeping"))
        assertEquals(VoiceIntent.Cues(on = false), i("mute"))
        assertEquals(VoiceIntent.Cues(on = true), i("turn on cues"))
        assertEquals(VoiceIntent.Pause, i("pause"))
        assertEquals(VoiceIntent.Resume, i("resume"))
        assertEquals(VoiceIntent.RepeatLast, i("say that again"))
        assertEquals(VoiceIntent.Help, i("what can I say"))
        assertEquals(VoiceIntent.CancelSeek, i("never mind"))
    }

    @Test fun contextSwitchPhrases() {
        assertEquals(VoiceIntent.SwitchContext(AppContext.SITTING), i("I'm sitting down now"))
        assertEquals(VoiceIntent.SwitchContext(AppContext.TRANSIT), i("just getting on the bus"))
        assertEquals(VoiceIntent.SwitchContext(AppContext.WALKING), i("okay let's walk"))
    }

    @Test fun gibberishIsUnknown() {
        // no seek verb + too long for the bare-noun fallback
        assertTrue(i("the weather today is quite nice outside") is VoiceIntent.Unknown)
        // short, but a non-navigating context disables the bare-noun fallback
        assertTrue(i("banana helicopter", AppContext.CONVERSATION) is VoiceIntent.Unknown)
        assertTrue(i("") is VoiceIntent.Unknown)
    }

    @Test fun languageSwitch() {
        assertEquals(VoiceIntent.SetLanguage(true), i("speak hindi"))
        assertEquals(VoiceIntent.SetLanguage(true), i("switch to hindi please"))
        assertEquals(VoiceIntent.SetLanguage(false), i("talk in english"))
        // a bare mention with no language cue must NOT flip the language
        assertTrue(i("find the hindi newspaper", AppContext.WALKING) is VoiceIntent.Find)
    }

    @Test fun phoneTasks() {
        assertEquals(VoiceIntent.CallContact("mom"), i("call mom"))
        assertEquals(VoiceIntent.CallContact("dr smith"), i("phone dr smith"))
        assertEquals(VoiceIntent.SetTimer(300), i("set a timer for five minutes"))
        assertEquals(VoiceIntent.SetTimer(90), i("timer 90 seconds"))
        assertEquals(VoiceIntent.SetTimer(3600), i("set a timer for an hour"))
    }
}

/** LlmPrompt.parse: lenient JSON -> resolution; prompt build stays grounded. */
class LlmPromptTest {
    private val scene = SceneBrief(
        context = "walking", objectsAhead = listOf("door ahead"), hazard = null,
        batteryPct = 82, camera = "ok", lastSpoken = "Looking for door.",
    )

    @Test fun mapsJsonActionsToIntents() {
        assertEquals(
            LlmResolution.Action(VoiceIntent.Find("elevator")),
            LlmPrompt.parse("""{"action":"find","target":"elevator"}"""),
        )
        assertEquals(
            LlmResolution.Action(VoiceIntent.SwitchContext(AppContext.TRANSIT)),
            LlmPrompt.parse("""{"action":"context","context":"transit"}"""),
        )
        assertEquals(
            LlmResolution.Action(VoiceIntent.SetTimer(300)),
            LlmPrompt.parse("""{"action":"timer","seconds":300}"""),
        )
        assertEquals(
            LlmResolution.Action(VoiceIntent.CallContact("mom")),
            LlmPrompt.parse("""{"action":"call","name":"mom"}"""),
        )
    }

    @Test fun toleratesProseAndFencesAroundJson() {
        val r = LlmPrompt.parse("Sure! ```json\n{\"action\":\"say\",\"text\":\"A chair is on your right.\"}\n``` hope that helps")
        assertEquals(LlmResolution.Speak("A chair is on your right."), r)
    }

    @Test fun plainTextReplyBecomesSpeak() {
        val r = LlmPrompt.parse("There is a door straight ahead of you.")
        assertTrue(r is LlmResolution.Speak && r.text.contains("door"))
    }

    @Test fun blankIsNull() {
        assertEquals(null, LlmPrompt.parse(""))
        assertEquals(null, LlmPrompt.parse(null))
    }

    @Test fun rejectsParrotedPlaceholders() {
        // a weak model that copies the prompt template must NOT trigger a bogus call / find
        assertEquals(null, LlmPrompt.parse("""{"action":"call","name":"<contact>"}"""))
        assertEquals(null, LlmPrompt.parse("""{"action":"find","target":"the thing"}"""))
    }

    @Test fun sentenceAnswerToAQuestion() {
        val r = LlmPrompt.parse("I can't see the traffic, wait and listen for cars before crossing.")
        assertTrue(r is LlmResolution.Speak && r.text.contains("wait"))
    }

    @Test fun deflectFlagAndGreenLightBothDefer() {
        assertEquals(LlmResolution.Defer, LlmPrompt.parse("""{"kind":"deflect"}"""))
        // model free-typed a movement green-light -> must be overridden, not spoken
        assertEquals(LlmResolution.Defer, LlmPrompt.parse("Yes, you can cross now."))
        assertEquals(LlmResolution.Defer, LlmPrompt.parse("""{"kind":"answer","text":"It's safe to walk."}"""))
        assertEquals(LlmResolution.Defer, LlmPrompt.parse("The path is clear, go ahead."))
    }

    @Test fun promptCarriesSceneGrounding() {
        val p = LlmPrompt.build("is it clear ahead", scene)
        assertTrue(p.contains("walking") && p.contains("door ahead") && p.contains("82 percent"))
        assertTrue(p.contains("is it clear ahead"))
        assertTrue("wraps in the Gemma chat template", p.contains("<start_of_turn>model"))
    }

    @Test fun stripsChatTemplateArtefactsFromReply() {
        val r = LlmPrompt.parse("Wait for the cars to pass first.<end_of_turn>")
        assertTrue(r is LlmResolution.Speak && r.text == "Wait for the cars to pass first.")
    }
}

/** NgramSafetyGate: the learned Tier-2 classifier — safety questions in, everything else out. */
class NgramSafetyGateTest {
    private val g = ai.secondsense.app.voice.NgramSafetyGate()

    @Test fun catchesSafetyQuestionsAcrossPhrasingAndScript() {
        listOf(
            "is it safe to cross", "can I walk now", "is the path clear",
            "will I bump into anything", "am I clear to go", "are there obstacles ahead",
            "क्या रास्ता साफ है", "kya main ab chal sakta hoon",
        ).forEach { assertTrue("should flag: \"$it\"", g.isSafetyQuery(it)) }
    }

    @Test fun passesNonSafetySpeech() {
        listOf(
            "find my keys", "read the sign", "what's my battery", "call mom",
            "set a timer for five minutes", "speak hindi", "I'm sitting down",
            "what's ahead", "tell me a joke", "",
        ).forEach { assertFalse("should NOT flag: \"$it\"", g.isSafetyQuery(it)) }
    }
}

/** GoalGrounding: everyday spoken words reach the detector labels; unknowns are flagged. */
class GoalGroundingTest {
    private fun det(label: String, cx: Float = 0.5f, prox: Float = 0.5f) =
        ai.secondsense.app.inference.Detection(
            label = label, score = 0.9f,
            box = ai.secondsense.app.inference.BBox(cx - 0.1f, 0.4f, cx + 0.1f, 0.6f),
            proximity = prox,
        )

    @Test fun synonymsMatchDetectorLabels() {
        val gg = ai.secondsense.app.voice.GoalGrounding
        assertTrue("bag -> backpack", gg.match(listOf(det("backpack")), "bag") != null)
        assertTrue("sofa -> chair", gg.match(listOf(det("chair")), "sofa") != null)
        assertTrue("phone -> cell phone", gg.match(listOf(det("cell phone")), "phone") != null)
        assertNull("no chair in view", gg.match(listOf(det("person")), "bag"))
    }

    @Test fun groundableTellsFindableFromNot() {
        val gg = ai.secondsense.app.voice.GoalGrounding
        assertTrue(gg.isGroundable("bag"))
        assertTrue(gg.isGroundable("chairs"))
        assertTrue(gg.isGroundable("phone"))
        assertFalse(gg.isGroundable("keys"))
        assertFalse(gg.isGroundable("door"))
        assertFalse(gg.isGroundable(null))
    }
}

/** Earcons: every context gets a distinct, non-empty tone shape. */
class EarconsTest {
    @Test fun everyContextHasADistinctSequence() {
        val seqs = ai.secondsense.app.context.AppContext.values().associateWith {
            ai.secondsense.app.sonification.Earcons.sequenceFor(it)
        }
        seqs.values.forEach { assertTrue("non-empty", it.isNotEmpty()) }
        val walk = ai.secondsense.app.sonification.Earcons.sequenceFor(ai.secondsense.app.context.AppContext.WALKING)
        assertTrue("walking rises", walk.last().hz > walk.first().hz)
        val home = ai.secondsense.app.sonification.Earcons.sequenceFor(ai.secondsense.app.context.AppContext.HOME)
        assertTrue("home falls", home.last().hz < home.first().hz)
        val sit = ai.secondsense.app.sonification.Earcons.sequenceFor(ai.secondsense.app.context.AppContext.SITTING)
        assertTrue("sitting is low", sit.single().hz < 400.0)
        assertTrue("shapes differ", seqs.values.map { it.map { n -> n.hz } }.distinct().size >= 5)
    }
}

/** SafetyGate: worst-of-window hysteresis + LLM-output green-light detection. */
class SafetyGateTest {
    @Test fun mostSevereWins() {
        assertEquals(
            HazardState.DROP_CONFIRMED,
            SafetyGate.mostSevere(listOf(HazardState.SAFE, HazardState.DROP_CONFIRMED, HazardState.SAFE, null)),
        )
        assertEquals(HazardState.POSSIBLE_DROP, SafetyGate.mostSevere(listOf(HazardState.SAFE, HazardState.POSSIBLE_DROP)))
        assertEquals(null, SafetyGate.mostSevere(listOf<HazardState?>(null, null)))
    }

    @Test fun greenLightPhrases() {
        assertTrue(SafetyGate.looksLikeMovementGreenLight("Yes, you can walk."))
        assertTrue(SafetyGate.looksLikeMovementGreenLight("It's safe to cross."))
        assertTrue(SafetyGate.looksLikeMovementGreenLight("The road is clear, go ahead."))
        assertTrue(SafetyGate.looksLikeMovementGreenLight("You're good, you're safe."))
        assertFalse(SafetyGate.looksLikeMovementGreenLight("There is a chair on your right."))
        assertFalse(SafetyGate.looksLikeMovementGreenLight("I cannot judge that, use your cane."))
        assertFalse(SafetyGate.looksLikeMovementGreenLight(""))
    }

    @Test fun anchorsAreCuratedAndMultilingual() {
        assertTrue("enough anchors for a smooth boundary", SafetyAnchors.PHRASES.size >= 40)
        assertTrue("has Devanagari", SafetyAnchors.PHRASES.any { it.any { c -> c.code in 0x0900..0x097F } })
        assertTrue("has Kannada", SafetyAnchors.PHRASES.any { it.any { c -> c.code in 0x0C80..0x0CFF } })
        assertTrue("no blanks / dupes", SafetyAnchors.PHRASES.none { it.isBlank() } &&
            SafetyAnchors.PHRASES.size == SafetyAnchors.PHRASES.distinct().size)
    }
}
