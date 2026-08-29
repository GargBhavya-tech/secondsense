package ai.secondsense.app.voice

import ai.secondsense.app.inference.BBox
import ai.secondsense.app.inference.ConfidenceTier
import ai.secondsense.app.inference.Detection
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** Phase 4 grounding (non-QNN path): spoken noun -> best matching COCO detection this frame. */
class GoalGroundingTest {

    private fun det(label: String?, cx: Float, prox: Float) = Detection(
        label = label,
        score = 0.9f,
        box = BBox(cx - 0.05f, 0.4f, cx + 0.05f, 0.6f),
        proximity = prox,
        tier = ConfidenceTier.WHITE,
    )

    @Test fun nullOrBlankGoalMatchesNothing() {
        assertNull(GoalGrounding.match(listOf(det("chair", 0.5f, 0.5f)), null))
        assertNull(GoalGrounding.match(listOf(det("chair", 0.5f, 0.5f)), "  "))
    }

    @Test fun goalNotPresentReturnsNull() {
        assertNull(GoalGrounding.match(listOf(det("person", 0.5f, 0.9f)), "chair"))
    }

    @Test fun matchesTheNamedClass() {
        val r = GoalGrounding.match(
            listOf(det("person", 0.5f, 0.9f), det("chair", 0.5f, 0.4f)),
            "chair",
        )
        assertEquals("chair", r?.label)
    }

    @Test fun amongSameClassNearestWins() {
        val r = GoalGrounding.match(
            listOf(
                det("chair", cx = 0.1f, prox = 0.9f),   // off-center but near
                det("chair", cx = 0.5f, prox = 0.4f),   // centered but far
            ),
            "chair",
        )
        assertEquals(0.1f, r?.box?.centerX ?: -9f, 1e-4f)
    }

    @Test fun caseInsensitiveAndSubstring() {
        assertEquals("chair", GoalGrounding.match(listOf(det("chair", 0.5f, 0.5f)), "CHAIR")?.label)
        assertEquals(
            "dining table",
            GoalGrounding.match(listOf(det("dining table", 0.5f, 0.5f)), "table")?.label,
        )
    }

    @Test fun ignoresUnlabeledDetections() {
        assertNull(GoalGrounding.match(listOf(det(null, 0.5f, 0.95f)), "chair"))
    }
}
