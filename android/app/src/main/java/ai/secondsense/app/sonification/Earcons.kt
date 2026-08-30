package ai.secondsense.app.sonification

import ai.secondsense.app.context.AppContext
import ai.secondsense.app.output.AudioOutput
import java.util.concurrent.Executors

/**
 * Phase 5 — a short distinctive tone burst on every activity-context switch, so the wearer
 * hears *which* mode they're in without waiting for the spoken line. Also the panic earcon.
 *
 * Pure [sequenceFor] (list of notes) is unit-tested; [play]/[panic] just fire the blips on a
 * dedicated single thread so they never block the UI, and drop if one is already sounding.
 */
class Earcons(private val audio: AudioOutput) {

    data class Note(val hz: Double, val ms: Int, val gapMs: Int = 40)

    private val exec = Executors.newSingleThreadExecutor { r -> Thread(r, "earcons").apply { isDaemon = true } }
    @Volatile private var busy = false

    fun play(ctx: AppContext) = fire(sequenceFor(ctx))

    /** Attention pattern for the panic gesture: fast high triple, repeated. */
    fun panic() = fire(
        buildList {
            repeat(2) {
                add(Note(990.0, 90, 30)); add(Note(990.0, 90, 30)); add(Note(990.0, 90, 30))
                add(Note(660.0, 200, 120))
            }
        },
    )

    private fun fire(notes: List<Note>) {
        if (busy) return
        busy = true
        exec.execute {
            try {
                for (n in notes) {
                    audio.playBlip(frequencyHz = n.hz, durationMs = n.ms, pan = 0.5f)
                    if (n.gapMs > 0) Thread.sleep(n.gapMs.toLong())
                }
            } catch (_: InterruptedException) {
            } finally {
                busy = false
            }
        }
    }

    fun close() {
        exec.shutdownNow()
    }

    companion object {
        /** Each context gets a recognisably different shape (contour + register). */
        fun sequenceFor(ctx: AppContext): List<Note> = when (ctx) {
            // rising two notes = "moving / go"
            AppContext.WALKING -> listOf(Note(520.0, 90), Note(720.0, 110, 0))
            // one firm mid note = "stopped, still watching"
            AppContext.STANDING -> listOf(Note(600.0, 150, 0))
            // gentle falling pair = "home / relaxed, room memory"
            AppContext.HOME -> listOf(Note(560.0, 100), Note(460.0, 130, 0))
            // one low note = "settled"
            AppContext.SITTING -> listOf(Note(330.0, 190, 0))
            // soft quick chime = "in a vehicle"
            AppContext.TRANSIT -> listOf(Note(494.0, 70), Note(622.0, 70), Note(740.0, 110, 0))
            // very short low tick = "quiet, conversation"
            AppContext.CONVERSATION -> listOf(Note(300.0, 70, 0))
        }
    }
}
