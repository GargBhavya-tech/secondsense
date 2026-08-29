package ai.secondsense.app.output

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import kotlin.math.PI
import kotlin.math.sin

/**
 * The audio output hook (build-map #6).
 *
 * For #6 this only has to prove "a test tone fires on a button tap" with no laptop
 * attached. But it is structured for what comes next so Phase 3 doesn't rewrite it:
 *
 *   - Stereo AudioTrack in STREAM mode -> the substrate for HRTF PAN (#18, direction).
 *   - A `pan` argument on the tone     -> direction channel, kept SEPARATE from...
 *   - pulse REPETITION timing (caller-driven) -> distance channel (#19).
 *
 * The cardinal Bible rule (§5.1): pitch encodes IDENTITY only; distance rides on
 * pulse REPETITION RATE, never pitch. This class never changes frequency to signal
 * distance, and the API gives distance no frequency knob — the invariant is enforced
 * by the shape of the interface, not just by discipline.
 *
 * On-device only.
 */
class AudioOutput {

    private val sampleRate = 44_100
    private var track: AudioTrack? = null

    fun initialize() {
        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_STEREO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(4096)

        // Headroom so one icon/spearcon write (a ~160 ms stereo 16-bit pulse is ~28 KB) does
        // NOT block the cue-loop thread waiting on buffer space. That blocking is what made the
        // "faster pulse = closer" distance channel plateau for near objects, where the target
        // inter-pulse interval (~120 ms) is shorter than a full write takes to drain.
        val trackBuf = maxOf(minBuf * 4, 64 * 1024)

        track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                    .build()
            )
            .setBufferSizeInBytes(trackBuf)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
            .apply { play() }
    }

    /** #6 done-condition: a test tone on button tap. Centered pan, fixed pitch. */
    fun testTone() {
        playBlip(frequencyHz = 660.0, durationMs = 180, pan = 0.5f)
    }

    /**
     * A single short blip.
     * @param frequencyHz IDENTITY only (which icon/timbre). Never used for distance.
     * @param pan         0f = hard left, 0.5f = center, 1f = hard right. DIRECTION (#18).
     * @param durationMs  length of this one pulse. Distance is conveyed by how OFTEN
     *                    the caller fires blips (repetition rate, #19), not by pitch.
     */
    fun playBlip(frequencyHz: Double, durationMs: Int, pan: Float) {
        val t = track ?: return
        val frames = sampleRate * durationMs / 1000
        val buf = ShortArray(frames * 2)
        val leftGain = (1f - pan).coerceIn(0f, 1f)
        val rightGain = pan.coerceIn(0f, 1f)
        for (i in 0 until frames) {
            // simple attack/decay envelope so blips don't click
            val env = envelope(i, frames)
            val s = (sin(2.0 * PI * frequencyHz * i / sampleRate) * env * Short.MAX_VALUE)
            buf[i * 2] = (s * leftGain).toInt().toShort()
            buf[i * 2 + 1] = (s * rightGain).toInt().toShort()
        }
        t.write(buf, 0, buf.size)
    }

    /**
     * Play an arbitrary mono buffer (e.g. a synthesized auditory icon or a spearcon),
     * panned. This is what the CueEngine (#22) uses so IDENTITY (the buffer's timbre)
     * and DIRECTION (pan) stay on separate channels.
     *
     * @param mono  samples in -1f..1f.
     * @param pan   0f left .. 0.5 center .. 1f right — DIRECTION (#18).
     * @param gain  overall amplitude 0f..1f (used for confidence/urgency shaping, NOT distance).
     */
    fun playMono(mono: FloatArray, pan: Float, gain: Float = 1f) {
        val t = track ?: return
        if (mono.isEmpty()) return
        val g = gain.coerceIn(0f, 1f)
        // Equal-power pan law — smoother, more "spatial" than linear (a lite HRTF stand-in).
        val angle = pan.coerceIn(0f, 1f) * (PI / 2)
        val leftGain = kotlin.math.cos(angle).toFloat() * g
        val rightGain = kotlin.math.sin(angle).toFloat() * g
        val buf = ShortArray(mono.size * 2)
        for (i in mono.indices) {
            val s = mono[i] * Short.MAX_VALUE
            buf[i * 2] = (s * leftGain).toInt().toShort()
            buf[i * 2 + 1] = (s * rightGain).toInt().toShort()
        }
        t.write(buf, 0, buf.size)
    }

    private fun envelope(i: Int, frames: Int): Double {
        val attack = frames * 0.1
        val release = frames * 0.2
        return when {
            i < attack -> i / attack
            i > frames - release -> (frames - i) / release
            else -> 1.0
        }.coerceIn(0.0, 1.0)
    }

    fun release() {
        track?.run { stop(); release() }
        track = null
    }
}
