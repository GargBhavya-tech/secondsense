package ai.secondsense.app.ar

import android.content.Context
import android.util.Log
import com.google.ar.core.Pose
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Save / load one room scan to app-private storage (`filesDir/rooms/room.json`).
 *
 * Stores the semantic points, the known occupancy cells, the running floor height, and an
 * ANCHOR POSE — the ARCore pose captured when the user tapped "Anchor" at a memorable spot
 * (their doorway, say). Because every session starts a fresh ARCore world frame, a reloaded
 * map is realigned by standing at that spot again: `T = anchorNow . anchorSaved^-1` maps every
 * saved coordinate into the new session's frame (see RoomScanActivity.applyLoadedRoom).
 *
 * Single slot for now — one room. JSON so it's inspectable during bring-up.
 * Caller: RoomScanActivity.
 */
object RoomStore {

    private const val TAG = "SecondSense/ar"
    private const val VERSION = 1

    data class PointRec(val label: String, val x: Float, val y: Float, val z: Float, val conf: Float, val hits: Int)
    data class CellRec(val x: Float, val z: Float, val blocked: Boolean)
    data class Saved(val anchor: Pose?, val floorY: Float, val points: List<PointRec>, val cells: List<CellRec>)

    private fun file(context: Context): File =
        File(context.filesDir, "rooms").apply { mkdirs() }.let { File(it, "room.json") }

    fun exists(context: Context): Boolean = file(context).exists()

    fun delete(context: Context) { runCatching { file(context).delete() } }

    fun save(context: Context, map: RoomMap, grid: OccupancyGrid, anchor: Pose?) {
        val root = JSONObject()
        root.put("version", VERSION)
        root.put("floorY", grid.floorY.let { if (it.isNaN()) 0.0 else it.toDouble() })
        if (anchor != null) {
            val t = anchor.translation
            val q = anchor.rotationQuaternion
            root.put("anchor", JSONArray(listOf(t[0], t[1], t[2], q[0], q[1], q[2], q[3])))
        }
        val pts = JSONArray()
        for (p in map.confirmed()) {
            pts.put(JSONObject().apply {
                put("l", p.label); put("x", p.x.toDouble()); put("y", p.y.toDouble()); put("z", p.z.toDouble())
                put("c", p.confidence.toDouble()); put("h", p.hits)
            })
        }
        root.put("points", pts)
        val cells = JSONArray()
        grid.forEachKnown { x, z, blocked ->
            cells.put(JSONObject().apply { put("x", x.toDouble()); put("z", z.toDouble()); put("b", if (blocked) 1 else 0) })
        }
        root.put("cells", cells)
        runCatching { file(context).writeText(root.toString()) }
            .onSuccess { Log.i(TAG, "room saved: ${pts.length()} points, ${cells.length()} cells") }
            .onFailure { Log.w(TAG, "room save failed: ${it.message}") }
    }

    fun load(context: Context): Saved? {
        val f = file(context)
        if (!f.exists()) return null
        return runCatching {
            val root = JSONObject(f.readText())
            val anchor = root.optJSONArray("anchor")?.let { a ->
                Pose(
                    floatArrayOf(a.getDouble(0).toFloat(), a.getDouble(1).toFloat(), a.getDouble(2).toFloat()),
                    floatArrayOf(a.getDouble(3).toFloat(), a.getDouble(4).toFloat(), a.getDouble(5).toFloat(), a.getDouble(6).toFloat()),
                )
            }
            val floorY = root.optDouble("floorY", 0.0).toFloat()
            val points = ArrayList<PointRec>()
            root.optJSONArray("points")?.let { arr ->
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    points.add(PointRec(o.getString("l"), o.getDouble("x").toFloat(), o.getDouble("y").toFloat(),
                        o.getDouble("z").toFloat(), o.optDouble("c", 0.5).toFloat(), o.optInt("h", 3)))
                }
            }
            val cells = ArrayList<CellRec>()
            root.optJSONArray("cells")?.let { arr ->
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    cells.add(CellRec(o.getDouble("x").toFloat(), o.getDouble("z").toFloat(), o.getInt("b") == 1))
                }
            }
            Saved(anchor, floorY, points, cells)
        }.onFailure { Log.w(TAG, "room load failed: ${it.message}") }.getOrNull()
    }
}
