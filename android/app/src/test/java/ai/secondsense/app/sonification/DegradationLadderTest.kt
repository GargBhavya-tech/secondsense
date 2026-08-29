package ai.secondsense.app.sonification

import ai.secondsense.app.inference.ConfidenceTier.BLUE
import ai.secondsense.app.inference.ConfidenceTier.RED
import ai.secondsense.app.inference.ConfidenceTier.WHITE
import ai.secondsense.app.inference.ConfidenceTier
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for the #24 graceful-degradation ladder. Proves the ladder is TOTAL (every
 * input lands on a defined rung), FULL is gated correctly, and PANIC is an independent
 * floor that coexists with any audio rung — the "always steer off a wall" guarantee.
 */
class DegradationLadderTest {

    @Test fun everyCombinationMapsToADefinedRung() {
        val tiers = listOf(WHITE, BLUE, RED)
        val proxes = listOf<Float?>(null, 0f, 0.5f, 0.79f, 0.80f, 1f)
        for (t in tiers) for (p in proxes) for (depth in listOf(true, false)) for (has in listOf(true, false)) {
            val d = DegradationLadder.decide(t, p, depth, has)
            assertTrue(d.audioRung in LadderRung.values())
        }
    }

    @Test fun fullOnlyWhenWhiteDepthAndLabel() {
        assertEquals(LadderRung.FULL, DegradationLadder.decide(WHITE, 0.5f, true, true).audioRung)
        assertEquals(LadderRung.PROXIMITY, DegradationLadder.decide(WHITE, 0.5f, true, false).audioRung)
        assertEquals(LadderRung.PROXIMITY, DegradationLadder.decide(BLUE, 0.5f, true, true).audioRung)
        assertEquals(LadderRung.PROXIMITY, DegradationLadder.decide(RED, 0.5f, true, true).audioRung)
    }

    @Test fun anyDepthGivesAtLeastProximity() {
        // depth present but not the FULL case -> PROXIMITY, never SILENT.
        assertEquals(LadderRung.PROXIMITY, DegradationLadder.decide(RED, 0.3f, true, false).audioRung)
        assertEquals(LadderRung.PROXIMITY, DegradationLadder.decide(BLUE, 0.3f, true, true).audioRung)
    }

    @Test fun noDepthIsSilentAudio() {
        assertEquals(LadderRung.SILENT_AUDIO, DegradationLadder.decide(RED, null, false, false).audioRung)
        assertEquals(LadderRung.SILENT_AUDIO, DegradationLadder.decide(WHITE, null, false, true).audioRung)
    }

    @Test fun panicIsAnIndependentFloor() {
        assertTrue(DegradationLadder.decide(WHITE, 0.9f, true, true).panic)   // + FULL
        assertTrue(DegradationLadder.decide(RED, 0.9f, true, false).panic)    // + PROXIMITY
        assertFalse(DegradationLadder.decide(WHITE, 0.5f, true, true).panic)  // far
        assertFalse(DegradationLadder.decide(RED, 0.9f, false, false).panic)  // no depth -> can't trust prox
    }

    @Test fun panicThresholdBoundary() {
        assertTrue(DegradationLadder.decide(BLUE, 0.80f, true, false).panic)
        assertFalse(DegradationLadder.decide(BLUE, 0.79f, true, false).panic)
    }

    @Test fun depthOnlyCloseObjectGetsProximityAudioAndPanicHaptic() {
        val d = DegradationLadder.decide(RED, 0.95f, true, false)
        assertEquals(LadderRung.PROXIMITY, d.audioRung)
        assertTrue(d.panic)
    }
}
