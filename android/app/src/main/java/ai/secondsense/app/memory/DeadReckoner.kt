package ai.secondsense.app.memory

import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Bare-bones pedestrian dead-reckoning in a local 2D ground plane — the "no SLAM" substitute
 * for a persistent world frame.
 *
 * Origin is wherever the app was when [reset] last ran; +Z points along the initial heading
 * ("forward"), +X is 90° clockwise ("right"). Each footstep advances the position by one
 * stride along the heading *at that step* (so a turn mid-walk is captured, unlike a
 * start-vs-end approximation). Heading comes from [ai.secondsense.app.sensors.ImuTracker].
 *
 * HONEST LIMITS: no loop closure, no drift correction. Heading drifts a few deg/min and
 * stride is a constant, so absolute error grows ~5–10 % of distance and compounds with turns.
 * Good for "your bottle is a few steps behind you on the left" over ~1 room / ~1 minute;
 * useless for cross-building navigation. All methods are synchronized — stepped from the
 * sensor thread, read from the camera-analysis and voice threads.
 */
class DeadReckoner {

    data class Pose(val x: Float, val z: Float, val headingDeg: Float)

    private var x = 0f
    private var z = 0f
    private var lastHeadingDeg = 0f

    /** Advance one stride along [headingDeg]. Call once per [PedometerTracker] step event. */
    @Synchronized
    fun onStep(headingDeg: Float, strideMeters: Float) {
        val r = headingDeg / 180f * PI.toFloat()
        x += strideMeters * sin(r)
        z += strideMeters * cos(r)
        lastHeadingDeg = headingDeg
    }

    /** Current pose; pass the live heading so the returned [Pose.headingDeg] is fresh. */
    @Synchronized
    fun pose(currentHeadingDeg: Float): Pose {
        lastHeadingDeg = currentHeadingDeg
        return Pose(x, z, currentHeadingDeg)
    }

    @Synchronized
    fun reset() {
        x = 0f; z = 0f; lastHeadingDeg = 0f
    }

    companion object {
        /**
         * World (local-frame) position of an object seen [distanceM] away at [bearingDeg]
         * (0 = straight ahead, + = to the right) from an observer at [observer].
         */
        fun placeObject(observer: Pose, distanceM: Float, bearingDeg: Float): Pair<Float, Float> {
            val worldAngle = (observer.headingDeg + bearingDeg) / 180f * PI.toFloat()
            return (observer.x + distanceM * sin(worldAngle)) to (observer.z + distanceM * cos(worldAngle))
        }

        /**
         * Range (m) and relative bearing (deg, -180..180, + = right, ±180 = directly behind)
         * from [observer] to a world-frame point.
         */
        fun relativeTo(observer: Pose, worldX: Float, worldZ: Float): Pair<Float, Float> {
            val dx = worldX - observer.x
            val dz = worldZ - observer.z
            val range = hypot(dx, dz)
            val worldBearingDeg = atan2(dx, dz) * 180f / PI.toFloat()
            var rel = worldBearingDeg - observer.headingDeg
            while (rel > 180f) rel -= 360f
            while (rel < -180f) rel += 360f
            return range to rel
        }
    }
}
