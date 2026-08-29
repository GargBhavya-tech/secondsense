package ai.secondsense.app

import ai.secondsense.app.inference.BBox
import ai.secondsense.app.inference.ConfidenceTier
import ai.secondsense.app.sonification.Calibration
import ai.secondsense.app.sonification.CueTarget
import ai.secondsense.app.sonification.TemporalSmoother
import ai.secondsense.app.voice.TargetNoun
import ai.secondsense.app.voice.VectorToGoalController
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** #16 — temporal smoothing gates flicker, confirms steady targets, then tracks live. */
class TemporalSmootherTest {
    private fun t(label: String, az: Float) =
        CueTarget(azimuth = az, proximity = 0.5f, label = label, tier = ConfidenceTier.WHITE)

    @Test fun steadyTargetConfirmsAfterThreeFrames() {
        val s = TemporalSmoother(confirmFrames = 3)
        assertNull(s.update(t("person", 0.5f)))   // frame 1 — warming
        assertNull(s.update(t("person", 0.5f)))   // frame 2 — warming
        assertEquals("person", s.update(t("person", 0.5f))?.label) // frame 3 — fires
        assertEquals("person", s.update(t("person", 0.51f))?.label) // then tracks live
    }

    @Test fun flickerNeverFires() {
        val s = TemporalSmoother(confirmFrames = 3)
        assertNull(s.update(t("person", 0.5f)))
        assertNull(s.update(t("dog", 0.5f)))      // identity changed -> streak resets
        assertNull(s.update(t("chair", 0.5f)))
        assertNull(s.update(t("vehicle", 0.5f)))
    }

    @Test fun nullResetsTheStreak() {
        val s = TemporalSmoother(confirmFrames = 2)
        assertNull(s.update(t("person", 0.5f)))
        assertNull(s.update(null))                 // lost -> reset
        assertNull(s.update(t("person", 0.5f)))    // must warm up again
        assertEquals("person", s.update(t("person", 0.5f))?.label)
    }
}

/** #7 — one-tap calibration re-references proximity against a captured baseline. */
class CalibrationTest {
    @Test fun uncalibratedIsIdentity() {
        val c = Calibration()
        assertFalse(c.isCalibrated)
        assertEquals(0.42f, c.apply(0.42f), 1e-4f)
    }

    @Test fun baselineRereferencesProximity() {
        val c = Calibration()
        c.capture(0.5f)
        assertTrue(c.isCalibrated)
        assertEquals(0f, c.apply(0.5f), 1e-4f)     // at baseline -> zero urgency
        assertEquals(1f, c.apply(1.0f), 1e-4f)     // touching -> max
        assertEquals(0f, c.apply(0.3f), 1e-4f)     // farther than baseline -> clamped 0
    }

    @Test fun clearRestoresIdentity() {
        val c = Calibration()
        c.capture(0.5f); c.clear()
        assertFalse(c.isCalibrated)
        assertEquals(0.3f, c.apply(0.3f), 1e-4f)
    }
}

/** #28 — vector-to-goal steering + arrival, and #26's noun extraction. */
class VectorToGoalControllerTest {
    private fun box(cx: Float) = BBox(cx - 0.05f, 0.4f, cx + 0.05f, 0.6f)

    @Test fun noGoalNoCue() {
        val v = VectorToGoalController()
        assertNull(v.cueFor(box(0.5f), 0.5f))
    }

    @Test fun goalProducesSteeringCue() {
        val v = VectorToGoalController()
        v.setGoal("door")
        val cue = v.cueFor(box(0.8f), 0.6f)
        assertEquals("door", cue?.label)
        assertEquals(0.8f, cue?.azimuth ?: -1f, 1e-4f)
    }

    @Test fun arrivalRequiresCenteredAndClose() {
        val v = VectorToGoalController()
        v.setGoal("door")
        assertTrue(v.hasArrived(box(0.5f), 0.9f))     // centered + close
        assertFalse(v.hasArrived(box(0.9f), 0.9f))    // off to the side
        assertFalse(v.hasArrived(box(0.5f), 0.3f))    // still far
    }
}

/** #26 — noun extraction from short spoken commands. */
class TargetNounTest {
    @Test fun extractsTheGoalNoun() {
        assertEquals("door", TargetNoun.extract("find the door"))
        assertEquals("exit", TargetNoun.extract("take me to the exit"))
        assertEquals("seat", TargetNoun.extract("where is my seat"))
    }

    @Test fun blankOrFillerOnlyIsNull() {
        assertNull(TargetNoun.extract(""))
        assertNull(TargetNoun.extract("   "))
        assertNull(TargetNoun.extract("find the"))
    }
}
