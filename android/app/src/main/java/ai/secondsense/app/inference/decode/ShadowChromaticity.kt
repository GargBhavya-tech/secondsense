package ai.secondsense.app.inference.decode

import android.graphics.Bitmap
import kotlin.math.atan
import kotlin.math.max

/**
 * Problem Statement 3 ("The Specular Trap") — Veto A: is a candidate drop-off edge actually a
 * CAST SHADOW rather than a physical step?
 *
 * A hard shadow makes a sharp discontinuity in the LUMINANCE channel that the edge-lattice /
 * monocular-depth detectors read as a stair line. But a shadow is (almost) pure illumination
 * change — the surface REFLECTANCE, and therefore its hue, is unchanged across the boundary.
 * A real material/geometry change (concrete -> black rubber tread, floor -> void) shifts hue
 * too. So: strong luminance edge + negligible chromaticity edge => shadow.
 *
 * Chromaticity is the illumination-invariant c1c2c3 encoding (Gevers & Smeulders):
 *   c1 = atan( R / max(G,B) ),  c2 = atan( G / max(R,B) ),  c3 = atan( B / max(R,G) )
 * — ratios of channels, so scaling all three by the illuminant (what a shadow does) cancels.
 *
 * DARK-ASPHALT-TRAP GUARD: this is deliberately a *chromaticity* test, not a brightness test.
 * Black stairs / dark tactile paving carry a real hue edge, so they score LOW here and are
 * never vetoed. The caller also only lets this downgrade a hazard, never fully clear it.
 *
 * Returns 0f (real edge / can't tell) .. 1f (confident cast shadow). Cheap: two `getPixels`
 * reads + arctans over a few hundred sampled pixels, well under 1 ms.
 *
 * Caller: SceneAnalyzer (feeds RawEvidence.shadowLikelihood); unit test.
 */
object ShadowChromaticity {

    private const val MIN_DL = 0.12f       // luminance gap below which there's no edge to explain
    private const val DL_SPAN = 0.28f
    private const val CHROMA_EDGE = 0.12f  // mean chromaticity gap (rad) that means "hue changed"

    fun shadowLikelihood(frame: Bitmap, edgeRowFraction: Float, corridor: TraversableCorridor): Float {
        val w = frame.width
        val h = frame.height
        if (w < 8 || h < 16) return 0f

        val x0 = (corridor.x1.coerceIn(0f, 1f) * w).toInt().coerceIn(0, w - 2)
        val x1 = (corridor.x2.coerceIn(0f, 1f) * w).toInt().coerceIn(x0 + 1, w - 1)
        val bandPx = (0.05f * h).toInt().coerceAtLeast(2)
        val gap = (0.02f * h).toInt().coerceAtLeast(1)
        val edgeY = (edgeRowFraction.coerceIn(0f, 1f) * h).toInt()

        val aTop = edgeY - gap - bandPx
        val aBot = edgeY - gap
        val bTop = edgeY + gap
        val bBot = edgeY + gap + bandPx
        if (aTop < 0 || bBot > h) return 0f   // no room for a band on one side

        val above = sampleBand(frame, x0, x1, aTop, aBot) ?: return 0f
        val below = sampleBand(frame, x0, x1, bTop, bBot) ?: return 0f

        val dL = kotlin.math.abs(above.l - below.l)
        val dC = (kotlin.math.abs(above.c1 - below.c1) +
            kotlin.math.abs(above.c2 - below.c2) +
            kotlin.math.abs(above.c3 - below.c3)) / 3f

        val lumTerm = ((dL - MIN_DL) / DL_SPAN).coerceIn(0f, 1f)
        val hueTerm = (1f - dC / CHROMA_EDGE).coerceIn(0f, 1f)
        return lumTerm * hueTerm
    }

    private class Stat(val l: Float, val c1: Float, val c2: Float, val c3: Float)

    private fun sampleBand(frame: Bitmap, x0: Int, x1: Int, y0: Int, y1: Int): Stat? {
        val bw = x1 - x0
        val bh = y1 - y0
        if (bw < 2 || bh < 1) return null
        val px = IntArray(bw * bh)
        frame.getPixels(px, 0, bw, x0, y0, bw, bh)
        val stride = max(1, px.size / 400)
        var sl = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0; var n = 0
        var i = 0
        while (i < px.size) {
            val p = px[i]
            val r = ((p shr 16) and 0xFF).toFloat()
            val g = ((p shr 8) and 0xFF).toFloat()
            val b = (p and 0xFF).toFloat()
            sl += (0.299f * r + 0.587f * g + 0.114f * b) / 255f
            s1 += atan(r / max(1f, max(g, b))).toDouble()
            s2 += atan(g / max(1f, max(r, b))).toDouble()
            s3 += atan(b / max(1f, max(r, g))).toDouble()
            n++
            i += stride
        }
        if (n == 0) return null
        return Stat((sl / n).toFloat(), (s1 / n).toFloat(), (s2 / n).toFloat(), (s3 / n).toFloat())
    }
}
