package ai.secondsense.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import ai.secondsense.app.inference.Detection

/**
 * DEBUG ONLY (DebugActivity) — draws every current-frame [Detection]'s box + label directly
 * on top of the camera preview, so a wrong label or a misplaced box is visible AT A GLANCE
 * instead of read off a text HUD. Boxes are normalized (0f..1f) frame coordinates; this view
 * just scales them to its own pixel size, so it works whatever the preview's aspect crop is.
 *
 * Kept intentionally dumb: no state beyond "the last frame's detections". MainActivity's
 * production HUD stays text-only on purpose (Bible: audio/haptics are the real UI); this
 * view only exists for bring-up debugging and is never shown outside DebugActivity.
 */
class BoxOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    @Volatile private var detections: List<Detection> = emptyList()

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        color = Color.parseColor("#00E5FF")
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 34f
        isAntiAlias = true
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }
    private val textBgPaint = Paint().apply {
        color = Color.parseColor("#CC000000")
        style = Paint.Style.FILL
    }

    /** Call from the analysis callback with the latest frame's detections. */
    fun setDetections(dets: List<Detection>) {
        detections = dets
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat(); val h = height.toFloat()
        if (w <= 0f || h <= 0f) return
        for (d in detections) {
            val l = d.box.left * w; val t = d.box.top * h
            val r = d.box.right * w; val b = d.box.bottom * h
            canvas.drawRect(l, t, r, b, boxPaint)

            val label = "${d.label ?: "?"} ${"%.2f".format(d.score)} p=${"%.2f".format(d.proximity)}"
            val textW = textPaint.measureText(label)
            val labelTop = (t - 40f).coerceAtLeast(0f)
            canvas.drawRect(l, labelTop, l + textW + 12f, labelTop + 38f, textBgPaint)
            canvas.drawText(label, l + 6f, labelTop + 28f, textPaint)
        }
    }
}
