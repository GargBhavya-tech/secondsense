package ai.secondsense.app.voice

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlin.concurrent.thread

/**
 * Ticket #26 — Voice command capture. In scan/seek mode, record a short spoken command from
 * the mic, hand it to the [SpeechRecognizer] (Whisper on QNN), and extract the target noun via
 * [TargetNoun]. Mic capture is real today; the transcription is the QNN model that goes live
 * with the native bridge — until then transcribe() returns null and [capture] reports that
 * honestly (recognizerReady=false) so the UI can say "voice model not loaded yet".
 *
 * Records mono 16 kHz PCM (Whisper's native rate) off the main thread. The CALLER must hold
 * RECORD_AUDIO permission before calling [capture]; this class does not prompt.
 */
class VoiceCommandCapture(
    private val recognizer: SpeechRecognizer,
    private val sampleRate: Int = 16_000,
    private val captureMillis: Int = 2500,
) {
    @Volatile private var recording = false

    /**
     * Capture ~[captureMillis] of audio, transcribe, and deliver the extracted target noun.
     * @param onResult (noun, transcript, recognizerReady) — noun/transcript are null when the
     *                 recognizer isn't ready yet; `recognizerReady` tells the UI whether to
     *                 say "voice model not loaded". Always invoked exactly once.
     */
    @SuppressLint("MissingPermission")
    fun capture(onResult: (noun: String?, transcript: String?, recognizerReady: Boolean) -> Unit) {
        if (recording) return
        recording = true
        thread(name = "voice-capture") {
            var recorder: AudioRecord? = null
            try {
                val minBuf = AudioRecord.getMinBufferSize(
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                ).coerceAtLeast(4096)
                recorder = AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    minBuf * 2,
                )
                val total = sampleRate * captureMillis / 1000
                val pcm = ShortArray(total)
                var read = 0
                recorder.startRecording()
                while (recording && read < total) {
                    val n = recorder.read(pcm, read, total - read)
                    if (n <= 0) break
                    read += n
                }
                recorder.stop()

                val ready = recognizer.isReady()
                val transcript = recognizer.transcribe(pcm.copyOf(read), sampleRate)
                onResult(TargetNoun.extract(transcript), transcript, ready)
            } catch (t: Throwable) {
                onResult(null, null, false)
            } finally {
                try { recorder?.release() } catch (_: Throwable) {}
                recording = false
            }
        }
    }

    fun cancel() { recording = false }
}
