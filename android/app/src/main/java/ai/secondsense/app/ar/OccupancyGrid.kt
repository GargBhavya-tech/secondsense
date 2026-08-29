package ai.secondsense.app.ar

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Top-down navigable-space map of the room, in ARCore's world X-Z plane.
 *
 * A fixed [cells]x[cells] grid at [cellM] metres, centred on the tracking origin (approx where
 * the scan started), so about +/-([cells]*[cellM]/2) m each way — ~8 m for the defaults,
 * enough for a room. Each cell accumulates FREE vs BLOCKED votes; an opposite vote also nudges
 * the other down, so a cell can change if furniture moves. State is the thresholded argmax.
 *
 * Fed from two sources by [RoomScanActivity]:
 *  - ARCore planes: horizontal-up planes near the floor -> FREE; vertical planes -> BLOCKED.
 *  - a sweep of the ARCore depth image: points at floor height -> FREE, points standing above
 *    the floor -> BLOCKED.
 *
 * [floorY] is the running-lowest horizontal surface seen — the reference every height test
 * uses. In-memory; synchronized (written from GL + worker threads, read from UI).
 */
class OccupancyGrid(
    val cells: Int = 64,
    val cellM: Float = 0.25f,
) {
    enum class State { UNKNOWN, FREE, BLOCKED }

    private val free = ShortArray(cells * cells)
    private val block = ShortArray(cells * cells)
    private val half = cells / 2

    @Volatile var floorY: Float = Float.NaN
        private set

    fun colOf(worldX: Float): Int = (worldX / cellM).roundToInt() + half
    fun rowOf(worldZ: Float): Int = (worldZ / cellM).roundToInt() + half
    private fun inBounds(c: Int, r: Int) = c in 0 until cells && r in 0 until cells

    @Synchronized
    fun noteFloorCandidate(y: Float) {
        floorY = when {
            floorY.isNaN() -> y
            y < floorY - 0.03f -> floorY + 0.3f * (y - floorY)
            else -> floorY
        }
    }

    @Synchronized
    fun voteFree(worldX: Float, worldZ: Float) = vote(colOf(worldX), rowOf(worldZ), freeVote = true)

    @Synchronized
    fun voteBlocked(worldX: Float, worldZ: Float) = vote(colOf(worldX), rowOf(worldZ), freeVote = false)

    private fun vote(c: Int, r: Int, freeVote: Boolean) {
        if (!inBounds(c, r)) return
        val i = r * cells + c
        if (freeVote) {
            free[i] = (free[i] + 2).coerceAtMost(60).toShort()
            block[i] = (block[i] - 1).coerceAtLeast(0).toShort()
        } else {
            block[i] = (block[i] + 2).coerceAtMost(60).toShort()
            free[i] = (free[i] - 1).coerceAtLeast(0).toShort()
        }
    }

    /** Classify one world point sampled from the depth image against the floor. */
    @Synchronized
    fun observeDepthPoint(wx: Float, wy: Float, wz: Float) {
        if (floorY.isNaN()) { noteFloorCandidate(wy); return }
        val h = wy - floorY
        when {
            abs(h) < 0.12f -> voteFree(wx, wz)
            h in 0.20f..2.20f -> voteBlocked(wx, wz)
            h < -0.15f -> noteFloorCandidate(wy)
        }
    }

    @Synchronized
    fun stateAt(c: Int, r: Int): State {
        if (!inBounds(c, r)) return State.UNKNOWN
        val i = r * cells + c
        return when {
            block[i] >= 3 && block[i] >= free[i] -> State.BLOCKED
            free[i] >= 3 -> State.FREE
            else -> State.UNKNOWN
        }
    }

    data class Stats(val free: Int, val blocked: Int, val unknown: Int, val floorY: Float)

    @Synchronized
    fun stats(): Stats {
        var f = 0; var b = 0; var u = 0
        for (r in 0 until cells) for (c in 0 until cells) when (stateAt(c, r)) {
            State.FREE -> f++
            State.BLOCKED -> b++
            State.UNKNOWN -> u++
        }
        return Stats(f, b, u, floorY)
    }

    /**
     * Small monospace map for the debug HUD. [camCol]/[camRow] draw the camera as '@'.
     * '#' blocked, middle-dot free, ' ' unknown. Downsampled to about [outW] columns.
     */
    @Synchronized
    fun miniMap(camCol: Int, camRow: Int, outW: Int = 44): String {
        val step = max(1, cells / outW)
        val sb = StringBuilder()
        var r = 0
        while (r < cells) {
            var c = 0
            while (c < cells) {
                if (camCol in c until c + step && camRow in r until r + step) {
                    sb.append('@'); c += step; continue
                }
                var b = 0; var f = 0
                for (rr in r until minOf(r + step, cells)) for (cc in c until minOf(c + step, cells)) {
                    when (stateAt(cc, rr)) {
                        State.BLOCKED -> b++
                        State.FREE -> f++
                        else -> {}
                    }
                }
                sb.append(if (b > 0) '#' else if (f > 0) '·' else ' ')
                c += step
            }
            sb.append('\n')
            r += step
        }
        return sb.toString()
    }

    @Synchronized
    fun clear() { free.fill(0); block.fill(0); floorY = Float.NaN }

    @Synchronized
    fun setFloorY(y: Float) { floorY = y }

    /** Visit every non-UNKNOWN cell as (worldX, worldZ, blocked) — for persistence. */
    @Synchronized
    fun forEachKnown(action: (Float, Float, Boolean) -> Unit) {
        for (r in 0 until cells) for (c in 0 until cells) {
            when (stateAt(c, r)) {
                State.FREE -> action((c - half) * cellM, (r - half) * cellM, false)
                State.BLOCKED -> action((c - half) * cellM, (r - half) * cellM, true)
                else -> {}
            }
        }
    }

    /** Load one cell from a saved map (strong vote so it survives without re-observation). */
    @Synchronized
    fun seed(worldX: Float, worldZ: Float, blocked: Boolean) {
        val c = colOf(worldX); val r = rowOf(worldZ)
        if (!inBounds(c, r)) return
        val i = r * cells + c
        if (blocked) block[i] = 8 else free[i] = 8
    }
}
