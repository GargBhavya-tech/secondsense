package ai.secondsense.app.ar

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import ai.secondsense.app.R
import ai.secondsense.app.inference.EngineConfig
import com.google.ar.core.Config
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import kotlin.math.max
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableException
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Path B — Steps 1–2: ARCore 6-DOF pose (step 1), plus the semantic layer (step 2) — route
 * ARCore's CPU camera image through the existing detector and pin each detection to a WORLD
 * coordinate using the frame's depth + pose. Points accumulate in [RoomMap].
 *
 * Isolated from the live-nav pipeline: owns the camera via an ARCore [Session] on a
 * [GLSurfaceView] render thread. It borrows MainActivity's already-loaded engine
 * ([EngineConfig.lastCreated]) — MainActivity's CameraX is stopped while this is foreground,
 * so the engine is idle. Detection runs on a worker thread, throttled, skip-if-busy, so it
 * never stalls ARCore's ~60 fps update loop.
 *
 * Still TODO (later steps): occupancy grid from ARCore depth+planes, persistence, routing.
 */
class RoomScanActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    private lateinit var glView: GLSurfaceView
    private lateinit var statusText: TextView

    private var session: Session? = null
    private var installRequested = false
    private var cameraTexId = 0

    private val roomMap = RoomMap()
    private val grid = OccupancyGrid()
    private val worker = Executors.newSingleThreadExecutor()
    private val detectBusy = AtomicBoolean(false)
    @Volatile private var scanning = false
    @Volatile private var depthEverWorked = false
    private var tts: TextToSpeech? = null
    private var lastSpokenConfirmed = 0
    private var lastSpeakMs = 0L

    private var lastPoseLine = "starting…"
    private var lastUiMs = 0L
    private var frames = 0L
    private var trackedFrames = 0L
    @Volatile private var detections = 0L
    /** elapsedRealtime when continuous TRACKING began; 0 while not tracking. */
    @Volatile private var stableSinceMs = 0L

    @Volatile private var lastPose: Pose? = null
    @Volatile private var anchorPose: Pose? = null
    @Volatile private var pendingLoad: RoomStore.Saved? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_room_scan)
        glView = findViewById(R.id.glSurface)
        statusText = findViewById(R.id.status)
        statusText.textSize = 10f
        findViewById<Button>(R.id.btnDone).setOnClickListener { finishScan() }
        findViewById<Button>(R.id.btnAnchor).apply {
            setOnClickListener { onAnchorTapped() }
            setOnLongClickListener { clearSavedRoom(); true }
        }

        tts = TextToSpeech(this) {}

        if (RoomStore.exists(this)) {
            pendingLoad = RoomStore.load(this)
            statusText.text = "Saved room found. Stand where you set the anchor, face the same way, then tap Anchor."
        }

        glView.preserveEGLContextOnPause = true
        glView.setEGLContextClientVersion(3)
        glView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        glView.setRenderer(this)
        glView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
    }

    override fun onResume() {
        super.onResume()
        if (session == null) {
            try {
                if (!ArSupport.ensureInstalled(this, userRequestedInstall = !installRequested)) {
                    installRequested = true
                    statusText.text = "Installing Google Play Services for AR…"
                    return
                }
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    != PackageManager.PERMISSION_GRANTED
                ) {
                    requestPermissions(arrayOf(Manifest.permission.CAMERA), 0)
                    return
                }
                session = Session(this).also { configure(it) }
                Log.i(TAG, "ARCore session created")
            } catch (e: UnavailableDeviceNotCompatibleException) {
                fail("This phone isn't AR-capable — room scan unavailable.")
                return
            } catch (e: UnavailableArcoreNotInstalledException) {
                installRequested = false
                statusText.text = "Install \"Google Play Services for AR\", then reopen Room Scan."
                return
            } catch (e: UnavailableApkTooOldException) {
                installRequested = false
                statusText.text = "Update \"Google Play Services for AR\", then reopen Room Scan."
                return
            } catch (e: UnavailableException) {
                fail("AR unavailable (${e.javaClass.simpleName}).")
                return
            } catch (e: Exception) {
                fail("Couldn't start AR: ${e.javaClass.simpleName} ${e.message}")
                return
            }
        }
        try {
            session?.resume()
        } catch (e: CameraNotAvailableException) {
            fail("Camera unavailable for AR. Close other camera apps and retry.")
            session = null
            return
        }
        scanning = true
        glView.onResume()
    }

    override fun onPause() {
        super.onPause()
        scanning = false
        glView.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.shutdown()
        tts?.run { stop(); shutdown() }
        session?.close()
        session = null
    }

    private fun finishScan() {
        val n = roomMap.confirmedCount()
        val gs = grid.stats()
        val areaM2 = gs.free * grid.cellM * grid.cellM
        RoomStore.save(this, roomMap, grid, anchorPose ?: lastPose)
        val msg = buildString {
            append(if (n == 0) "No objects mapped" else "$n objects mapped: ${roomMap.summary()}")
            append(". About %.0f square metres of clear floor. Room saved.".format(areaM2))
        }
        tts?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "scan-done")
        Log.i(TAG, "scan finished — $msg  (grid free=${gs.free} blocked=${gs.blocked})")
        finish()
    }

    /** Tap "Anchor": set the reference pose, or — if a saved room is pending — align + load it. */
    private fun onAnchorTapped() {
        val now = lastPose
        if (now == null) {
            Toast.makeText(this, "Wait for tracking first", Toast.LENGTH_SHORT).show()
            return
        }
        val pending = pendingLoad
        if (pending != null) {
            applyLoadedRoom(pending, now)
            pendingLoad = null
            anchorPose = now
            speak("Room loaded and aligned. Keep scanning to fill it in.")
        } else {
            anchorPose = now
            speak("Anchor set. This is your reference point.")
        }
    }

    private fun clearSavedRoom() {
        RoomStore.delete(this)
        pendingLoad = null
        roomMap.clear()
        grid.clear()
        anchorPose = null
        speak("Saved room cleared.")
    }

    /**
     * Realign a saved map into THIS session's ARCore frame. `T = anchorNow . anchorSaved^-1`
     * maps every saved coordinate to where it physically is now, given the user is standing at
     * the same spot facing the same way.
     */
    private fun applyLoadedRoom(saved: RoomStore.Saved, anchorNow: Pose) {
        val t = if (saved.anchor != null) anchorNow.compose(saved.anchor.inverse()) else Pose.IDENTITY
        val nowRt = SystemClock.elapsedRealtime()
        for (p in saved.points) {
            val w = t.transformPoint(floatArrayOf(p.x, p.y, p.z))
            roomMap.addRaw(p.label, w[0], w[1], w[2], p.conf, p.hits, nowRt)
        }
        grid.setFloorY(t.transformPoint(floatArrayOf(0f, saved.floorY, 0f))[1])
        for (c in saved.cells) {
            val w = t.transformPoint(floatArrayOf(c.x, saved.floorY, c.z))
            grid.seed(w[0], w[2], c.blocked)
        }
        Log.i(TAG, "loaded room: ${saved.points.size} points, ${saved.cells.size} cells, anchored=${saved.anchor != null}")
    }

    private fun speak(msg: String) {
        Log.i(TAG, msg)
        tts?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "ar")
    }

    private fun configure(s: Session) {
        val config = Config(s).apply {
            updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
            focusMode = Config.FocusMode.AUTO
            lightEstimationMode = Config.LightEstimationMode.DISABLED
            if (s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                depthMode = Config.DepthMode.AUTOMATIC
                Log.i(TAG, "ARCore depth: AUTOMATIC supported")
            } else {
                Log.i(TAG, "ARCore depth: NOT supported — objects can't be world-placed")
            }
        }
        s.configure(config)
    }

    // --- GLSurfaceView.Renderer -------------------------------------------------------------

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        val tex = IntArray(1)
        GLES20.glGenTextures(1, tex, 0)
        cameraTexId = tex[0]
        GLES20.glBindTexture(0x8D65 /* GL_TEXTURE_EXTERNAL_OES */, cameraTexId)
        GLES20.glTexParameteri(0x8D65, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(0x8D65, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        @Suppress("DEPRECATION")
        val rotation = windowManager.defaultDisplay.rotation
        session?.setDisplayGeometry(rotation, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        try {
            s.setCameraTextureName(cameraTexId)
            val frame = s.update()
            val cam = frame.camera
            frames++
            val t = cam.pose.translation
            val state = cam.trackingState
            if (state == TrackingState.TRACKING) lastPose = cam.pose
            val nowRt = SystemClock.elapsedRealtime()
            if (state == TrackingState.TRACKING) {
                trackedFrames++
                if (stableSinceMs == 0L) stableSinceMs = nowRt
            } else {
                stableSinceMs = 0L   // frame just shifted — don't log points across a reset
            }
            val trackingStable = stableSinceMs != 0L && nowRt - stableSinceMs > 800L

            // --- step 2: throttled world-anchored detection (only while tracking is solid) ---
            if (trackingStable && scanning &&
                frames % DETECT_EVERY == 0L && detectBusy.compareAndSet(false, true)
            ) {
                var w = 0; var h = 0
                val nv21 = runCatching {
                    frame.acquireCameraImage().use { img -> w = img.width; h = img.height; yuv420ToNv21(img) }
                }.getOrNull()
                val depth = runCatching {
                    frame.acquireDepthImage16Bits().use { copyDepth(it) }
                }.getOrNull()
                if (nv21 == null) {
                    detectBusy.set(false)
                } else {
                    val pose = cam.pose
                    val fl = cam.imageIntrinsics.focalLength
                    val pp = cam.imageIntrinsics.principalPoint
                    val dim = cam.imageIntrinsics.imageDimensions
                    worker.execute {
                        runCatching { processFrame(nv21, w, h, depth, pose, fl, pp, dim) }
                            .onFailure { Log.w(TAG, "processFrame: ${it.message}") }
                        detectBusy.set(false)
                    }
                }
            }

            val reason = if (state != TrackingState.TRACKING) " (${cam.trackingFailureReason})" else ""
            lastPoseLine = "%s%s\npos x=%.2f y=%.2f z=%.2f  frames %d\nobjects: %d   dets run: %d   depth: %s".format(
                state, reason, t[0], t[1], t[2], frames,
                roomMap.confirmedCount(), detections, if (depthEverWorked) "ok" else "—",
            )
            if (nowRt - lastUiMs > 700) {
                lastUiMs = nowRt
                roomMap.prune(nowRt)
                if (trackingStable) feedPlanes(s)
                val summary = roomMap.summary()
                val gs = grid.stats()
                val camCol = grid.colOf(t[0]); val camRow = grid.rowOf(t[2])
                val mapArt = grid.miniMap(camCol, camRow, outW = 48)
                val anchorLine = when {
                    pendingLoad != null -> "saved room PENDING — stand at anchor spot, tap Anchor"
                    anchorPose != null -> "anchor: set"
                    else -> "anchor: none (tap Anchor at a landmark before Done)"
                }
                runOnUiThread {
                    statusText.text = "$lastPoseLine\nobjects: $summary\n$anchorLine\n" +
                        "grid: free ${gs.free}  blocked ${gs.blocked}  floorY %.2f\n%s".format(gs.floorY, mapArt)
                }
                maybeSpeak(nowRt, summary)
            }
        } catch (e: CameraNotAvailableException) {
            Log.w(TAG, "camera lost mid-session: ${e.message}")
        } catch (e: Exception) {
            Log.w(TAG, "onDrawFrame: ${e.javaClass.simpleName} ${e.message}")
        }
    }

    private fun maybeSpeak(nowMs: Long, summary: String) {
        val c = roomMap.confirmedCount()
        if (c > lastSpokenConfirmed && nowMs - lastSpeakMs > 3500) {
            lastSpokenConfirmed = c
            lastSpeakMs = nowMs
            tts?.speak("$c objects: $summary", TextToSpeech.QUEUE_FLUSH, null, "scan")
        }
    }

    // --- detection + world placement (worker thread) --------------------------------------

    private fun processFrame(
        nv21: ByteArray, w: Int, h: Int,
        depth: DepthGrid?,
        pose: Pose,
        focal: FloatArray, principal: FloatArray, dims: IntArray,
    ) {
        val engine = EngineConfig.lastCreated ?: return
        if (!engine.isReady) return
        val bmp = nv21ToBitmap(nv21, w, h) ?: return
        val result = engine.infer(bmp, false)
        bmp.recycle()
        detections++
        if (depth == null) return
        depthEverWorked = true

        val fx = focal[0]; val fy = focal[1]
        val ppx = principal[0]; val ppy = principal[1]
        val iw = dims[0].toFloat(); val ih = dims[1].toFloat()
        val now = SystemClock.elapsedRealtime()

        for (d in result.detections) {
            val label = d.label ?: continue
            if (d.score < MIN_SCORE) continue
            val zMeters = depth.sample(d.box.centerX, d.box.centerY)
            if (zMeters <= 0.15f || zMeters > 8f) continue
            // Back-project bbox centre through the pinhole into the camera frame
            // (+X right, +Y up, -Z forward — ARCore/OpenGL convention), then to world.
            val px = d.box.centerX * iw
            val py = d.box.centerY * ih
            val xCam = (px - ppx) / fx * zMeters
            val yCam = -(py - ppy) / fy * zMeters
            val world = pose.transformPoint(floatArrayOf(xCam, yCam, -zMeters))
            val isNew = roomMap.observe(label, world[0], world[1], world[2], d.score, now)
            if (isNew) Log.i(TAG, "new: %s @ (%.2f, %.2f, %.2f)  %.0f%%".format(label, world[0], world[1], world[2], d.score * 100))
        }

        // --- step 3: sweep the depth image into the occupancy grid ---
        val sxD = iw / depth.w
        val syD = ih / depth.h
        var dy = 2
        while (dy < depth.h - 2) {
            var dx = 2
            while (dx < depth.w - 2) {
                val mm = depth.data[dy * depth.w + dx].toInt() and 0xFFFF
                if (mm in 300..6000) {
                    val zz = mm / 1000f
                    val xc = (dx * sxD - ppx) / fx * zz
                    val yc = -(dy * syD - ppy) / fy * zz
                    val wp = pose.transformPoint(floatArrayOf(xc, yc, -zz))
                    grid.observeDepthPoint(wp[0], wp[1], wp[2])
                }
                dx += 6
            }
            dy += 6
        }
    }

    /** Rasterize ARCore's tracked planes into the grid: floor -> FREE, walls -> BLOCKED. */
    private fun feedPlanes(s: Session) {
        for (p in s.getAllTrackables(Plane::class.java)) {
            if (p.trackingState != TrackingState.TRACKING || p.subsumedBy != null) continue
            val isFloor = p.type == Plane.Type.HORIZONTAL_UPWARD_FACING
            val isWall = p.type == Plane.Type.VERTICAL
            if (!isFloor && !isWall) continue
            val cp = p.centerPose
            val py = cp.ty()
            if (isFloor) {
                grid.noteFloorCandidate(py)
                if (!grid.floorY.isNaN() && kotlin.math.abs(py - grid.floorY) > 0.25f) continue
            }
            val ext = max(p.extentX, p.extentZ) / 2f + 0.3f
            val c0 = grid.colOf(cp.tx() - ext).coerceIn(0, grid.cells - 1)
            val c1 = grid.colOf(cp.tx() + ext).coerceIn(0, grid.cells - 1)
            val r0 = grid.rowOf(cp.tz() - ext).coerceIn(0, grid.cells - 1)
            val r1 = grid.rowOf(cp.tz() + ext).coerceIn(0, grid.cells - 1)
            for (r in r0..r1) for (c in c0..c1) {
                val wx = (c - grid.cells / 2) * grid.cellM
                val wz = (r - grid.cells / 2) * grid.cellM
                if (p.isPoseInPolygon(Pose.makeTranslation(wx, py, wz))) {
                    if (isWall) grid.voteBlocked(wx, wz) else grid.voteFree(wx, wz)
                }
            }
        }
    }

    // --- image plumbing ------------------------------------------------------------------

    /** DEPTH16 plane copied off the ARCore Image so the worker can read it after close(). */
    private class DepthGrid(val data: ShortArray, val w: Int, val h: Int) {
        /** metres at a normalized (0..1) image coordinate; 0f if no estimate. */
        fun sample(nx: Float, ny: Float): Float {
            val cx = (nx * w).toInt().coerceIn(1, w - 2)
            val cy = (ny * h).toInt().coerceIn(1, h - 2)
            var best = 0
            for (yy in cy - 1..cy + 1) for (xx in cx - 1..cx + 1) {
                val mm = data[yy * w + xx].toInt() and 0xFFFF
                if (mm in 1..8000 && (best == 0 || mm < best)) best = mm
            }
            return best / 1000f
        }
    }

    private fun copyDepth(img: Image): DepthGrid {
        val w = img.width; val h = img.height
        val plane = img.planes[0]
        val buf = plane.buffer
        val rowStride = plane.rowStride
        val out = ShortArray(w * h)
        val row = ByteArray(w * 2)
        for (y in 0 until h) {
            buf.position(y * rowStride)
            buf.get(row, 0, w * 2)
            for (x in 0 until w) {
                val lo = row[x * 2].toInt() and 0xFF
                val hi = row[x * 2 + 1].toInt() and 0xFF
                out[y * w + x] = ((hi shl 8) or lo).toShort()
            }
        }
        return DepthGrid(out, w, h)
    }

    private fun yuv420ToNv21(image: Image): ByteArray {
        val w = image.width; val h = image.height
        val out = ByteArray(w * h * 3 / 2)
        var pos = 0

        val yp = image.planes[0]
        val yb = yp.buffer
        val yrs = yp.rowStride
        val yps = yp.pixelStride
        for (row in 0 until h) {
            if (yps == 1) {
                yb.position(row * yrs)
                yb.get(out, pos, w)
                pos += w
            } else {
                var idx = row * yrs
                for (col in 0 until w) { out[pos++] = yb.get(idx); idx += yps }
            }
        }

        // NV21 = Y plane followed by interleaved V,U at quarter resolution.
        val up = image.planes[1]
        val vp = image.planes[2]
        val ub = up.buffer
        val vb = vp.buffer
        val urs = up.rowStride; val ups = up.pixelStride
        val vrs = vp.rowStride; val vps = vp.pixelStride
        val cw = w / 2; val ch = h / 2
        for (row in 0 until ch) {
            var uidx = row * urs
            var vidx = row * vrs
            for (col in 0 until cw) {
                out[pos++] = vb.get(vidx)
                out[pos++] = ub.get(uidx)
                uidx += ups
                vidx += vps
            }
        }
        return out
    }

    private fun nv21ToBitmap(nv21: ByteArray, w: Int, h: Int): Bitmap? {
        val yuv = YuvImage(nv21, ImageFormat.NV21, w, h, null)
        val baos = ByteArrayOutputStream()
        if (!yuv.compressToJpeg(Rect(0, 0, w, h), 85, baos)) return null
        val bytes = baos.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun fail(msg: String) {
        Log.w(TAG, msg)
        runOnUiThread {
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
            statusText.text = msg
        }
    }

    private companion object {
        const val TAG = "SecondSense/ar"
        const val DETECT_EVERY = 8L
        const val MIN_SCORE = 0.45f
    }
}
