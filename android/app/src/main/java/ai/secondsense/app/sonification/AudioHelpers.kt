package ai.secondsense.app.sonification

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal WAV reader for the on-device-TTS output used by [Spearcon].
 * TTS writes 16-bit PCM WAV; we only need mono float samples back.
 */
object WavIo {

    /** Read a 16-bit PCM WAV into mono float (-1f..1f). Returns null on any surprise. */
    fun readMonoFloat(file: File): FloatArray? {
        return try {
            val bytes = file.readBytes()
            if (bytes.size < 44) return null
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

            // Parse just enough of the header to locate fmt + data.
            // RIFF....WAVE then chunks. Walk chunks to find "fmt " and "data".
            if (bytes.decodeAscii(0, 4) != "RIFF" || bytes.decodeAscii(8, 4) != "WAVE") return null

            var pos = 12
            var channels = 1
            var bits = 16
            var dataOffset = -1
            var dataLen = 0
            while (pos + 8 <= bytes.size) {
                val id = bytes.decodeAscii(pos, 4)
                val size = ByteBuffer.wrap(bytes, pos + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
                val body = pos + 8
                when (id) {
                    "fmt " -> {
                        channels = ByteBuffer.wrap(bytes, body + 2, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt()
                        bits = ByteBuffer.wrap(bytes, body + 14, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt()
                    }
                    "data" -> { dataOffset = body; dataLen = size }
                }
                pos = body + size + (size and 1)   // chunks are word-aligned
                if (dataOffset >= 0 && channels > 0) { /* keep scanning fmt if before data */ }
            }
            if (dataOffset < 0 || bits != 16) return null

            val end = (dataOffset + dataLen).coerceAtMost(bytes.size)
            val sampleCount = (end - dataOffset) / 2
            val framebb = ByteBuffer.wrap(bytes, dataOffset, end - dataOffset).order(ByteOrder.LITTLE_ENDIAN)
            val mono = FloatArray(sampleCount / channels.coerceAtLeast(1))
            var out = 0
            var i = 0
            while (i + channels <= sampleCount) {
                var acc = 0f
                for (c in 0 until channels) acc += framebb.short / 32768f
                mono[out++] = acc / channels
                i += channels
            }
            mono.copyOf(out)
        } catch (_: Throwable) {
            null
        }
    }

    private fun ByteArray.decodeAscii(off: Int, len: Int): String =
        String(this, off, len, Charsets.US_ASCII)
}

/**
 * Time-compression for spearcons — makes speech shorter/faster without a full
 * phase-vocoder. Linear resample is enough for a caricatured single word at ~1.8x;
 * pitch rises slightly, which actually reinforces the "spearcon, not a spoken word"
 * character. Good enough for identity legibility, cheap enough for the sprint.
 */
object TimeStretch {
    fun compress(input: FloatArray, factor: Float): FloatArray {
        if (input.isEmpty() || factor <= 1f) return input
        val outLen = (input.size / factor).toInt().coerceAtLeast(1)
        val out = FloatArray(outLen)
        for (i in 0 until outLen) {
            val src = i * factor
            val i0 = src.toInt()
            val i1 = (i0 + 1).coerceAtMost(input.size - 1)
            val frac = src - i0
            out[i] = input[i0] * (1 - frac) + input[i1] * frac
        }
        return out
    }
}
