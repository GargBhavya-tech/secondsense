package ai.secondsense.app.sonification

import ai.secondsense.app.inference.ConfidenceTier.BLUE
import ai.secondsense.app.inference.ConfidenceTier.RED
import ai.secondsense.app.inference.ConfidenceTier.WHITE
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tests for the #23 confidence-tier state machine. Pure logic, runs on the JVM.
 * Mirrors the numerical harness: mapping, no-depth/no-label => RED, hysteresis,
 * single-frame-dip stability, and asymmetric promotion thresholds.
 */
class TierClassifierTest {

    private fun settle(t: TierClassifier, score: Float, frames: Int = 3,
                       depth: Boolean = true, hasLabel: Boolean = true) =
        (1..frames).map { t.classify(score, depth, hasLabel) }.last()

    @Test fun highScoreReachesWhite() =
        assertEquals(WHITE, settle(TierClassifier(), 0.9f))

    @Test fun midScoreReachesBlue() {
        val t = TierClassifier()
        settle(t, 0.9f)
        assertEquals(BLUE, settle(t, 0.4f))
    }

    @Test fun lowScoreReachesRed() {
        val t = TierClassifier()
        settle(t, 0.9f)
        assertEquals(RED, settle(t, 0.1f))
    }

    @Test fun noDepthIsAlwaysRed() =
        assertEquals(RED, settle(TierClassifier(), 0.99f, depth = false))

    @Test fun noClassLabelIsAlwaysRed() =
        assertEquals(RED, settle(TierClassifier(), 0.99f, hasLabel = false))

    @Test fun singleLowFrameDoesNotDemoteFromWhite() {
        val t = TierClassifier()
        settle(t, 0.9f)
        assertEquals(WHITE, t.classify(0.1f))          // one bad frame
        assertEquals(WHITE, t.classify(0.9f))          // recovers
    }

    @Test fun boundaryJitterDoesNotFlipEveryFrame() {
        val t = TierClassifier()
        settle(t, 0.9f)                                 // WHITE
        val seq = listOf(0.55f, 0.50f, 0.55f, 0.50f, 0.55f, 0.50f)
        val out = seq.map { t.classify(it) }
        val changes = (1 until out.size).count { out[it] != out[it - 1] }
        assert(changes <= 1) { "tier flipped $changes times: $out" }
    }

    @Test fun promotionFromRedRequiresHigherBar() {
        val t = TierClassifier()
        // 0.55 is between blueEnter(0.32) and whiteEnter(0.62): should reach BLUE only.
        assertEquals(BLUE, settle(t, 0.55f))
        assertEquals(WHITE, settle(t, 0.7f))
    }

    @Test fun resetReturnsToRed() {
        val t = TierClassifier()
        settle(t, 0.9f)
        t.reset()
        assertEquals(RED, t.tier)
    }
}
