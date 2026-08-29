package ai.secondsense.app.inference.decode

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Turns a camera [Bitmap] into a square, model-sized float input buffer (letterboxed,
 * aspect-preserving), and remembers the transform so decoded boxes can be mapped back
 * to normalized original-frame coordinates.
 *
 * SHARED across runtimes on purpose: both the TFLite engine and the future QNN engine
 * feed a square RGB float tensor, so this preprocessing is identical for both. The only
 * thing that ever differs per runtime is who consumes the buffer.
 *
 * INVARIANT: this assumes [bitmap] is already UPRIGHT. CameraX ImageAnalysis delivers
 * frames in sensor orientation — the analyzer must rotate by image.imageInfo
 * .rotationDegrees BEFORE calling the engine, or every box (and therefore every pan/
 * azimuth cue) is measured along the wrong axis. See FrameAnalyzer.
 */
object Preprocess {

    /** Ultralytics letterbox pad color (neutral gray) so padding doesn't look like content. */
    private const val PAD = 114

    /**
     * @param scale  original->model scale factor (same for x and y, aspect preserved).
     * @param padX   left padding in model pixels.
     * @param padY   top padding in model pixels.
     * @param size   model input side length (square).
     * @param origW  original bitmap width.
     * @param origH  original bitmap height.
     */
    data class Letterbox(
        val buffer: ByteBuffer,
        val scale: Float,
        val padX: Int,
        val padY: Int,
        val size: Int,
        val origW: Int,
        val origH: Int,
    ) {
        /**
         * Map a box given in MODEL-PIXEL xyxy (0..size) back to NORMALIZED original-frame
         * coordinates (0..1), undoing scale + padding and clamping to frame.
         */
        fun toNormalizedFrame(x1: Float, y1: Float, x2: Float, y2: Float): FloatArray {
            val ox1 = ((x1 - padX) / scale) / origW
            val oy1 = ((y1 - padY) / scale) / origH
            val ox2 = ((x2 - padX) / scale) / origW
            val oy2 = ((y2 - padY) / scale) / origH
            return floatArrayOf(
                ox1.coerceIn(0f, 1f), oy1.coerceIn(0f, 1f),
                ox2.coerceIn(0f, 1f), oy2.coerceIn(0f, 1f),
            )
        }
    }

    /**
     * Letterbox [bitmap] to [size]x[size] and pack an NHWC float32 RGB buffer normalized
     * to [normalizeTo01] (0..1 is the qai-hub float-export convention).
     *
     * @param channelsFirst if the model wants NCHW instead of NHWC, set true. qai-hub
     *                      TFLite exports are NHWC (false); the flag exists so the QNN
     *                      engine can reuse this method if its binary expects NCHW.
     */
    fun letterbox(
        bitmap: Bitmap,
        size: Int,
        normalizeTo01: Boolean = true,
        channelsFirst: Boolean = false,
    ): Letterbox {
        val w = bitmap.width
        val h = bitmap.height
        val scale = minOf(size.toFloat() / w, size.toFloat() / h)
        val newW = Math.round(w * scale)
        val newH = Math.round(h * scale)
        val padX = (size - newW) / 2
        val padY = (size - newH) / 2

        val scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        val square = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(square).apply {
            drawColor(Color.rgb(PAD, PAD, PAD))
            drawBitmap(scaled, null, Rect(padX, padY, padX + newW, padY + newH), null)
        }
        if (scaled !== bitmap) scaled.recycle()

        val pixels = IntArray(size * size)
        square.getPixels(pixels, 0, size, 0, 0, size, size)
        square.recycle()

        val buf = ByteBuffer.allocateDirect(size * size * 3 * 4).order(ByteOrder.nativeOrder())
        val div = if (normalizeTo01) 255f else 1f

        if (!channelsFirst) {
            // NHWC: r,g,b per pixel in row-major order.
            for (p in pixels) {
                buf.putFloat(((p shr 16) and 0xFF) / div) // R
                buf.putFloat(((p shr 8) and 0xFF) / div)  // G
                buf.putFloat((p and 0xFF) / div)          // B
            }
        } else {
            // NCHW: all R, then all G, then all B.
            for (p in pixels) buf.putFloat(((p shr 16) and 0xFF) / div)
            for (p in pixels) buf.putFloat(((p shr 8) and 0xFF) / div)
            for (p in pixels) buf.putFloat((p and 0xFF) / div)
        }
        buf.rewind()
        return Letterbox(buf, scale, padX, padY, size, w, h)
    }
}
