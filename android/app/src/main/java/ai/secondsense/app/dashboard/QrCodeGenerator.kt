package ai.secondsense.app.dashboard

import android.graphics.Bitmap
import android.graphics.Color
import com.google.zxing.BarcodeFormat
import com.google.zxing.qrcode.QRCodeWriter

/**
 * Ticket #30 — turns the dashboard's local URL into a scannable QR bitmap, entirely offline
 * (ZXing core does no network I/O; it's pure bitmap-matrix encoding). A judge/demo-partner's
 * laptop scans this instead of someone reading an IP address aloud mid-demo.
 */
object QrCodeGenerator {

    fun generate(text: String, sizePx: Int = 512): Bitmap {
        val matrix = QRCodeWriter().encode(text, BarcodeFormat.QR_CODE, sizePx, sizePx)
        val bmp = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.RGB_565)
        for (x in 0 until sizePx) {
            for (y in 0 until sizePx) {
                bmp.setPixel(x, y, if (matrix.get(x, y)) Color.BLACK else Color.WHITE)
            }
        }
        return bmp
    }
}
