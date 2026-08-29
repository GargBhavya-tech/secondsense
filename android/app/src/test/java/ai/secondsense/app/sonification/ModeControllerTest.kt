package ai.secondsense.app.sonification

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for the #25 mode split. Confirms FLOW is sparse (center-crop on, one cue, no
 * voice) and SCAN_SEEK opens up (whole frame, more cues, voice accepted), and that
 * listeners fire on change.
 */
class ModeControllerTest {

    @Test fun flowIsSparse() {
        val m = ModeController(OperatingMode.FLOW)
        assertTrue(m.useCenterCrop)
        assertEquals(1, m.maxSimultaneousCues)
        assertFalse(m.acceptsVoiceCommands)
    }

    @Test fun scanSeekOpensUp() {
        val m = ModeController(OperatingMode.SCAN_SEEK)
        assertFalse(m.useCenterCrop)
        assertTrue(m.maxSimultaneousCues > 1)
        assertTrue(m.acceptsVoiceCommands)
    }

    @Test fun toggleFlips() {
        val m = ModeController(OperatingMode.FLOW)
        m.toggle()
        assertEquals(OperatingMode.SCAN_SEEK, m.mode)
        m.toggle()
        assertEquals(OperatingMode.FLOW, m.mode)
    }

    @Test fun listenerFiresOnAddAndChange() {
        val m = ModeController(OperatingMode.FLOW)
        val seen = mutableListOf<OperatingMode>()
        m.addListener { seen.add(it) }         // fires immediately with current
        m.set(OperatingMode.SCAN_SEEK)         // fires on change
        m.set(OperatingMode.SCAN_SEEK)         // no-op, must NOT fire again
        assertEquals(listOf(OperatingMode.FLOW, OperatingMode.SCAN_SEEK), seen)
    }
}
