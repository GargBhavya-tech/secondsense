package ai.secondsense.app

import ai.secondsense.app.inference.BBox
import ai.secondsense.app.inference.ConfidenceTier
import ai.secondsense.app.inference.Detection
import ai.secondsense.app.inference.FrameResult
import ai.secondsense.app.inference.SettledSighting
import ai.secondsense.app.inference.decode.DetectionStabilizer
import ai.secondsense.app.inference.decode.HazardState
import ai.secondsense.app.inference.decode.RestingStateVerifier
import ai.secondsense.app.memory.DeadReckoner
import ai.secondsense.app.memory.MemoryPhrase
import ai.secondsense.app.memory.ObjectMemory
import ai.secondsense.app.sonification.CueTarget
import ai.secondsense.app.sonification.ObstacleHabituation
import ai.secondsense.app.voice.SceneNarrator
import org.junit.Assert.assertEquals
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
