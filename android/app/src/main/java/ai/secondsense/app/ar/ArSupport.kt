package ai.secondsense.app.ar

import android.app.Activity
import android.content.Context
import android.util.Log
import com.google.ar.core.ArCoreApk

/**
 * Thin wrapper around ARCore's availability + install handshake, so the rest of the app can
 * ask "can this device do the room scan?" without touching ARCore APIs directly.
 *
 * Device support is a RUNTIME fact: the iQOO 15 (or any phone) is only AR-capable if Google's
 * per-device calibration has shipped for it AND "Google Play Services for AR" is installed.
 * [availability] answers the first; [ensureInstalled] drives the (one-time, Play-Store) install
 * of the second.
 *
 * Callers: RoomScanActivity (gate + session create), MainActivity (enable/disable the button).
 */
object ArSupport {

    private const val TAG = "SecondSense/ar"

    enum class State { SUPPORTED, NEEDS_INSTALL, UNSUPPORTED, UNKNOWN }

    /**
     * Non-blocking check. ARCore may answer "checking, ask again" a few times right after
     * process start — callers should re-poll (ARCore caches the real answer within ~200 ms).
     */
    fun availability(context: Context): State = try {
        when (ArCoreApk.getInstance().checkAvailability(context)) {
            ArCoreApk.Availability.SUPPORTED_INSTALLED -> State.SUPPORTED
            ArCoreApk.Availability.SUPPORTED_APK_TOO_OLD,
            ArCoreApk.Availability.SUPPORTED_NOT_INSTALLED -> State.NEEDS_INSTALL
            ArCoreApk.Availability.UNSUPPORTED_DEVICE_NOT_CAPABLE -> State.UNSUPPORTED
            ArCoreApk.Availability.UNKNOWN_CHECKING,
            ArCoreApk.Availability.UNKNOWN_ERROR,
            ArCoreApk.Availability.UNKNOWN_TIMED_OUT -> State.UNKNOWN
            else -> State.UNKNOWN
        }
    } catch (t: Throwable) {
        Log.w(TAG, "availability check failed: ${t.message}")
        State.UNSUPPORTED
    }

    /**
     * Call from [Activity.onResume]. Returns true when Play Services for AR is present and we
     * can create a Session now; false means an install dialog was just launched (or declined)
     * and the caller should bail out of this resume and try again next resume.
     */
    fun ensureInstalled(activity: Activity, userRequestedInstall: Boolean): Boolean = try {
        when (ArCoreApk.getInstance().requestInstall(activity, userRequestedInstall)) {
            ArCoreApk.InstallStatus.INSTALLED -> true
            ArCoreApk.InstallStatus.INSTALL_REQUESTED -> false
            else -> false
        }
    } catch (t: Throwable) {
        Log.w(TAG, "requestInstall failed: ${t.message}")
        false
    }
}
