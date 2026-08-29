package ai.secondsense.app.perception

import android.content.Context

/**
 * The user's spoken-language preference, persisted across launches.
 *
 * Drives two things:
 *  - which TTS locale the app speaks its OWN cues in (scene description, sign read-outs,
 *    "person facing you"), English (en-IN / en-US) or Hindi (hi-IN);
 *  - whether a sign whose script does NOT match that preference is run through on-device
 *    translation before being spoken ([translateSigns]) or just read in its native script.
 *
 * The blind user never sees the screen, so the toggles on the debug panel are for the
 * sighted helper during setup; the value sticks so it only has to be set once.
 */
class LanguagePrefs(context: Context) {
    private val sp = context.getSharedPreferences("secondsense_lang", Context.MODE_PRIVATE)

    /** true -> speak in Hindi (hi-IN); false -> speak in English. Default English. */
    var speakHindi: Boolean
        get() = sp.getBoolean(KEY_HINDI, false)
        set(v) { sp.edit().putBoolean(KEY_HINDI, v).apply() }

    /**
     * true -> when a sign's script differs from [speakHindi], translate it (Hindi<->English)
     * before speaking; false -> always read the sign in whatever script it's printed in.
     * Default true; the actual translation still degrades to native-script if the on-device
     * model pair hasn't been downloaded yet.
     */
    var translateSigns: Boolean
        get() = sp.getBoolean(KEY_TRANSLATE, true)
        set(v) { sp.edit().putBoolean(KEY_TRANSLATE, v).apply() }

    private companion object {
        const val KEY_HINDI = "speak_hindi"
        const val KEY_TRANSLATE = "translate_signs"
    }
}
