package ai.secondsense.app.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Ticket #33 — turns raw 16kHz mono PCM into YAMNet's expected input: a [96,64] log-mel
 * spectrogram patch (0.96s of audio). Validated offline first (debug_yamnet.py, against the
 * real exported yamnet.tflite) — this is a direct Kotlin port of that validated math, not a
 * from-scratch guess. Parameters come from YAMNet's own published spec (torch_audioset's
 * params.py, itself sourced from Google's original TensorFlow implementation):
 *   16kHz, 25ms/10ms STFT window/hop, 512-pt FFT, 64 mel bins (125-7500Hz),
 *   MAGNITUDE (not power) spectrogram, log(mel + 0.001).
 *
 * Pure Kotlin, no external DSP library — the FFT is a standard radix-2 Cooley-Tukey
 * (FFT_LEN=512 is a power of 2, so radix-2 applies cleanly with no padding tricks needed).
 */
object MelSpectrogram {
    const val SAMPLE_RATE = 16000
    const val WIN_SAMPLES = 400   // 25ms
    const val HOP_SAMPLES = 160   // 10ms
    const val FFT_LEN = 512
    const val N_MELS = 64
    const val PATCH_FRAMES = 96   // 0.96s worth of 10ms hops
    private const val FMIN = 125.0
    private const val FMAX = 7500.0
    private const val LOG_OFFSET = 0.001

    private val hannWindow = FloatArray(WIN_SAMPLES) { i ->
        (0.5 - 0.5 * cos(2.0 * PI * i / (WIN_SAMPLES - 1))).toFloat()
    }

    // [N_MELS][FFT_LEN/2+1] triangular filterbank, computed once.
    private val filterbank: Array<FloatArray> = buildMelFilterbank()

    /**
     * @param pcm mono float samples in -1f..1f, at [SAMPLE_RATE]. Must contain at least
     *            enough samples for one [PATCH_FRAMES]-frame patch after reflect-padding;
     *            returns an empty list otherwise.
     * @return zero or more [PATCH_FRAMES]x[N_MELS] log-mel patches (one per full 0.96s of
     *         audio available), row-major flattened (patch[frame * N_MELS + mel]).
     */
    fun toLogMelPatches(pcm: FloatArray): List<FloatArray> {
        val pad = FFT_LEN / 2
        val padded = reflectPad(pcm, pad)
        val nFrames = 1 + (padded.size - FFT_LEN) / HOP_SAMPLES
        if (nFrames <= 0) return emptyList()

        val melFrames = Array(nFrames) { FloatArray(N_MELS) }
        val re = FloatArray(FFT_LEN)
        val im = FloatArray(FFT_LEN)
        for (f in 0 until nFrames) {
            val start = f * HOP_SAMPLES
            re.fill(0f); im.fill(0f)
            for (i in 0 until WIN_SAMPLES) {
                val sampleIdx = start + i
                val sample = if (sampleIdx < padded.size) padded[sampleIdx] else 0f
                re[i] = sample * hannWindow[i]
            }
            fftRadix2InPlace(re, im)
            // magnitude spectrum, bins 0..FFT_LEN/2 (real-input FFT is symmetric)
            val magnitude = FloatArray(FFT_LEN / 2 + 1) { k -> hypot(re[k].toDouble(), im[k].toDouble()).toFloat() }
            for (m in 0 until N_MELS) {
                var sum = 0f
                val fb = filterbank[m]
                for (k in fb.indices) sum += fb[k] * magnitude[k]
                melFrames[f][m] = ln(sum + LOG_OFFSET).toFloat()
            }
        }

        val nPatches = nFrames / PATCH_FRAMES
        if (nPatches <= 0) return emptyList()
        return (0 until nPatches).map { p ->
            val out = FloatArray(PATCH_FRAMES * N_MELS)
            for (t in 0 until PATCH_FRAMES) {
                val frame = melFrames[p * PATCH_FRAMES + t]
                System.arraycopy(frame, 0, out, t * N_MELS, N_MELS)
            }
            out
        }
    }

    private fun reflectPad(x: FloatArray, pad: Int): FloatArray {
        val out = FloatArray(x.size + 2 * pad)
        for (i in 0 until pad) out[i] = x[min(pad - i, x.size - 1).coerceAtLeast(0)]
        System.arraycopy(x, 0, out, pad, x.size)
        for (i in 0 until pad) {
            val srcIdx = (x.size - 2 - i).coerceIn(0, x.size - 1)
            out[pad + x.size + i] = x[srcIdx]
        }
        return out
    }

    private fun hzToMel(hz: Double) = 2595.0 * log10(1.0 + hz / 700.0)
    private fun melToHz(mel: Double) = 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0)

    private fun buildMelFilterbank(): Array<FloatArray> {
        val nFftBins = FFT_LEN / 2 + 1
        val melMin = hzToMel(FMIN)
        val melMax = hzToMel(FMAX)
        val melPoints = DoubleArray(N_MELS + 2) { melMin + (melMax - melMin) * it / (N_MELS + 1) }
        val hzPoints = melPoints.map { melToHz(it) }
        val binPoints = hzPoints.map { ((FFT_LEN + 1) * it / SAMPLE_RATE).toInt() }

        return Array(N_MELS) { m ->
            val fLeft = binPoints[m]
            val fCenter = binPoints[m + 1]
            val fRight = binPoints[m + 2]
            val fb = FloatArray(nFftBins)
            for (k in fLeft until fCenter) {
                if (k in 0 until nFftBins) fb[k] = (k - fLeft).toFloat() / max(fCenter - fLeft, 1)
            }
            for (k in fCenter until fRight) {
                if (k in 0 until nFftBins) fb[k] = (fRight - k).toFloat() / max(fRight - fCenter, 1)
            }
            fb
        }
    }

    /** In-place radix-2 Cooley-Tukey FFT. [re]/[im] length must be a power of 2 (FFT_LEN=512). */
    private fun fftRadix2InPlace(re: FloatArray, im: FloatArray) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wr = cos(ang).toFloat()
            val wi = sin(ang).toFloat()
            var i = 0
            while (i < n) {
                var curWr = 1f
                var curWi = 0f
                for (k in 0 until len / 2) {
                    val uRe = re[i + k]; val uIm = im[i + k]
                    val vRe = re[i + k + len / 2] * curWr - im[i + k + len / 2] * curWi
                    val vIm = re[i + k + len / 2] * curWi + im[i + k + len / 2] * curWr
                    re[i + k] = uRe + vRe; im[i + k] = uIm + vIm
                    re[i + k + len / 2] = uRe - vRe; im[i + k + len / 2] = uIm - vIm
                    val nextWr = curWr * wr - curWi * wi
                    val nextWi = curWr * wi + curWi * wr
                    curWr = nextWr; curWi = nextWi
                }
                i += len
            }
            len = len shl 1
        }
    }
}
