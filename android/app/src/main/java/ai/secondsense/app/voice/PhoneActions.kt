package ai.secondsense.app.voice

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.AlarmClock
import android.provider.ContactsContract
import androidx.core.content.ContextCompat

/**
 * The two concrete phone tasks the design chat called out for Phase 4: "call someone" and
 * "set a timer". Everything here is a single framework Intent — SecondSense is a navigation
 * aid, not a general phone assistant, so this stays deliberately tiny.
 *
 * Caller: MainActivity.handleVoiceIntent (CallContact / SetTimer). All methods are main-thread
 * safe (a contacts query of a few rows) and return a short spoken-result string.
 */
object PhoneActions {

    /**
     * Look up [spokenName] in Contacts and place a call. Requires CALL_PHONE + READ_CONTACTS
     * (the caller requests them). Returns a line to speak: what happened, or why it couldn't.
     * The call itself is the user's just-spoken instruction — we place it, we don't confirm
     * again (a blind user can't read a confirm dialog; hanging up is the undo).
     */
    fun call(context: Context, spokenName: String): String {
        if (!granted(context, Manifest.permission.READ_CONTACTS)) return "I need contacts permission to call."
        val (name, number) = lookup(context, spokenName) ?: return "I couldn't find $spokenName in your contacts."
        if (!granted(context, Manifest.permission.CALL_PHONE)) return "I need calling permission. Found $name's number though."
        return try {
            context.startActivity(
                Intent(Intent.ACTION_CALL, Uri.fromParts("tel", number, null))
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            )
            "Calling $name."
        } catch (t: Throwable) {
            "I couldn't start the call."
        }
    }

    /** Hand [seconds] to the system clock app's timer. No permission needed. */
    fun setTimer(context: Context, seconds: Int, label: String = "SecondSense"): String {
        return try {
            context.startActivity(
                Intent(AlarmClock.ACTION_SET_TIMER)
                    .putExtra(AlarmClock.EXTRA_LENGTH, seconds)
                    .putExtra(AlarmClock.EXTRA_MESSAGE, label)
                    .putExtra(AlarmClock.EXTRA_SKIP_UI, true)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            )
            "Timer set for ${spokenDuration(seconds)}."
        } catch (t: Throwable) {
            "No timer app is available."
        }
    }

    private fun granted(c: Context, p: String) =
        ContextCompat.checkSelfPermission(c, p) == PackageManager.PERMISSION_GRANTED

    /** Best-effort contacts match: exact display name first, then a LIKE prefix. */
    private fun lookup(context: Context, spokenName: String): Pair<String, String>? {
        val uri = ContactsContract.CommonDataKinds.Phone.CONTENT_URI
        val cols = arrayOf(
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
            ContactsContract.CommonDataKinds.Phone.NUMBER,
        )
        val clean = spokenName.trim()
        for (sel in listOf(
            "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} = ? COLLATE NOCASE" to arrayOf(clean),
            "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?" to arrayOf("$clean%"),
            "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?" to arrayOf("%$clean%"),
        )) {
            runCatching {
                context.contentResolver.query(uri, cols, sel.first, sel.second, null)?.use { c ->
                    if (c.moveToFirst()) {
                        val n = c.getString(0) ?: clean
                        val num = c.getString(1)?.replace(Regex("[^+0-9]"), "") ?: return@use
                        if (num.isNotBlank()) return n to num
                    }
                }
            }
        }
        return null
    }

    private fun spokenDuration(s: Int): String = when {
        s % 3600 == 0 -> "${s / 3600} hour${if (s / 3600 == 1) "" else "s"}"
        s % 60 == 0 -> "${s / 60} minute${if (s / 60 == 1) "" else "s"}"
        else -> "$s seconds"
    }
}
