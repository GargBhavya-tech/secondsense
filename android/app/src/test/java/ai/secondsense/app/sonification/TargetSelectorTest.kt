package ai.secondsense.app.sonification

import ai.secondsense.app.inference.BBox
import ai.secondsense.app.inference.ConfidenceTier
import ai.secondsense.app.inference.Detection
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Pure-logic tests for #14 (center-crop) + #15 (closest-in-center, static/dynamic).
 * These run on the JVM with no device — the whole point of keeping targeting as
 * logic over Detection is that it's verifiable off the phone.
 */
class TargetSelectorTest {

    private val sel = TargetSelector()

    private fun det(label: String, cx: Float, prox: Float, moving: Boolean) = Detection(
        label = label,
        score = 0.9f,
        box = BBox(cx - 0.05f, 0.4f, cx + 0.05f, 0.6f),
        proximity = prox,
        moving = moving,
        tier = ConfidenceTier.WHITE,
    )

    @Test fun offCenterHighProximityObjectIsIgnored() {
        val r = sel.selectDetection(listOf(
            det("chair", cx = 0.05f, prox = 0.9f, moving = false),
            det("person", cx = 0.50f, prox = 0.4f, moving = false),
        ))
        assertEquals("person", r?.label)
    }

    @Test fun closestInCenterWinsWhenBothStatic() {
        val r = sel.selectDetection(listOf(
            det("chair", cx = 0.50f, prox = 0.3f, moving = false),
            det("door", cx = 0.52f, prox = 0.7f, moving = false),
        ))
        assertEquals("door", r?.label)
    }

    @Test fun movingBeatsStaticAtEqualProximity() {
        val r = sel.selectDetection(listOf(
            det("chair", cx = 0.50f, prox = 0.55f, moving = false),
            det("person", cx = 0.50f, prox = 0.50f, moving = true),
        ))
        assertEquals("person", r?.label)
    }

    @Test fun farMoverDoesNotBeatMuchCloserStatic() {
        val r = sel.selectDetection(listOf(
            det("chair", cx = 0.50f, prox = 0.90f, moving = false),
            det("person", cx = 0.50f, prox = 0.40f, moving = true),
        ))
        assertEquals("chair", r?.label)
    }

    @Test fun nothingCenteredGivesNoCue() {
        val r = sel.selectDetection(listOf(
            det("chair", cx = 0.05f, prox = 0.9f, moving = false),
        ))
        assertNull(r)
    }
}
