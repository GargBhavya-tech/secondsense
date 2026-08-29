package ai.secondsense.app.voice

import ai.secondsense.app.context.AppContext

/**
 * Phase 3 of the assistant layer: turn a free-form spoken transcript into one action from a
 * CLOSED set, deterministically and offline. This is the fast path that runs BEFORE the
 * on-device LLM (Phase 4) — the great majority of what a user says while navigating is one of
 * a dozen intents, and a rule/synonym grammar handles those in microseconds with no model.
 * Anything it can't place becomes [VoiceIntent.Unknown], which Phase 4 will hand to Llama for
 * real reasoning; today MainActivity just says "say help for a list".
 *
 * Pure Kotlin, no Android imports — unit-testable off-device. The one behavioural knob is
 * [AppContext]: a bare noun with no verb ("keys", "the door") is only taken as a Find request
 * in contexts where the user is actively getting around; in TRANSIT / SITTING / CONVERSATION
 * that same word is more likely to be conversation, so it falls through to Unknown.
 *
 * Caller: MainActivity.startVoiceCapture -> handleVoiceIntent. Also unit test.
 */
sealed interface VoiceIntent {
    /** "find my keys", "take me to the door", bare "chair" in a navigating context. */
    data class Find(val target: String) : VoiceIntent
    /** "where did I leave my phone", "where are my keys" — recall a set-down object from memory. */
    data class Recall(val target: String) : VoiceIntent
    /** "what's ahead", "describe", "what do you see". */
    object Describe : VoiceIntent
    /**
     * "is it safe to cross", "is the path clear", "can I go" — a safety question. Answered
     * DETERMINISTICALLY from the live hazard pipeline + detections, never by the LLM (a small
     * model will cheerfully hallucinate "yes, cross"). Always carries a "can't see traffic /
     * use your cane" caveat.
     */
    object SafetyCheck : VoiceIntent
    /** "read", "read the sign", "what does it say" — one-shot OCR of the current frame. */
    object ReadText : VoiceIntent
    /** "I'm sitting", "getting on the bus", "let's walk" — switch the activity context. */
    data class SwitchContext(val context: AppContext) : VoiceIntent
    /** "status", "where am I", "battery", "what mode". */
    object Status : VoiceIntent
    /** "say that again", "repeat". */
    object RepeatLast : VoiceIntent
    /** "mute" / "stop the beeping" => on=false; "cues on" / "guide me" => on=true. */
    data class Cues(val on: Boolean) : VoiceIntent
    object Pause : VoiceIntent
    object Resume : VoiceIntent
    /** "never mind", "stop looking" — drop the current Find/Recall goal. */
    object CancelSeek : VoiceIntent
    object Help : VoiceIntent
    /** "speak Hindi" / "talk in English" — switch the spoken-output language. */
    data class SetLanguage(val hindi: Boolean) : VoiceIntent
    /** "call mom" — dial a contact by name (Phase 4 phone task). */
    data class CallContact(val name: String) : VoiceIntent
    /** "set a timer for five minutes" — hands off to the system clock (Phase 4 phone task). */
    data class SetTimer(val seconds: Int) : VoiceIntent
    /** Nothing in the grammar matched — Phase 4 hands this to the LLM. */
    data class Unknown(val transcript: String) : VoiceIntent
}

object IntentInterpreter {

    fun interpret(raw: String?, ctx: AppContext): VoiceIntent {
        val t = raw?.lowercase()?.trim()?.replace(Regex("[.,!?]+"), "").orEmpty()
        if (t.isBlank()) return VoiceIntent.Unknown("")

        fun has(vararg phrases: String) = phrases.any { it in t }
        fun startsAny(vararg p: String) = p.any { t == it || t.startsWith("$it ") }

        // 1. Help — before everything, "what can I say" must not be read as a question to answer.
        if (has("what can i say", "what can you do", "what can i do", "list commands", "help me use",
                "how do i use", "what are my options") || t == "help" || t == "commands"
        ) return VoiceIntent.Help

        // 2. Repeat.
        if (has("say that again", "say it again", "repeat that", "come again", "what did you say") ||
            t == "repeat" || t == "again" || t == "pardon"
        ) return VoiceIntent.RepeatLast

        // 2b. Spoken-output language. Needs a "speak/talk/switch/..." cue so "find the hindi
        // newspaper" doesn't flip the language.
        val langCue = has("speak", "talk", "switch", "language", "change to", "back to", "mein", "में", "बोल")
        if ("हिंदी" in t || (Regex("\\bhindi\\b").containsMatchIn(t) && (langCue || t == "hindi"))) {
            return VoiceIntent.SetLanguage(hindi = true)
        }
        if (Regex("\\benglish\\b").containsMatchIn(t) && (langCue || t == "english")) {
            return VoiceIntent.SetLanguage(hindi = false)
        }

        // 3. Activity-context switch (same phrase set the voice path used inline before).
        contextPhrase(t)?.let { return VoiceIntent.SwitchContext(it) }

        // 4. Pause / resume.
        if (has("carry on", "keep going", "start again", "wake up", "un pause", "unpause") ||
            t == "resume" || t == "continue" || t == "go on"
        ) return VoiceIntent.Resume
        if (has("hold on", "take a break", "pause everything", "freeze") ||
            t == "pause" || t == "wait"
        ) return VoiceIntent.Pause

        // 5. Cues on / off. "stop the beeping" is the user's own words from the design chat.
        if (has("stop the beeping", "stop beeping", "stop the sound", "stop the noise", "stop the beep",
                "turn off cues", "turn off the cues", "turn off sound", "turn off the sound",
                "turn the sound off", "no cues", "no sound", "cues off", "sound off",
                "be quiet", "shut up") ||
            t == "mute" || t == "quiet" || t == "silence"
        ) return VoiceIntent.Cues(on = false)
        if (has("turn on cues", "turn on the cues", "turn on sound", "turn on the sound", "cues on",
                "sound on", "start cueing", "start the cues", "un mute", "unmute")
        ) return VoiceIntent.Cues(on = true)

        // 6. Cancel an active search.
        if (has("never mind", "nevermind", "forget it", "stop looking", "stop searching",
                "stop the search", "stop finding", "cancel that") || t == "cancel" || t == "stop it"
        ) return VoiceIntent.CancelSeek

        // 7. Read text / signs.
        if (has("read the", "read this", "read that", "read it", "read aloud", "read out",
                "does it say", "does this say", "does that say", "does the sign say", "sign say",
                "what is written", "whats written", "what's written") ||
            t == "read" || t.startsWith("read ")
        ) return VoiceIntent.ReadText

        // 8. Recall from memory — "where" + a set-down cue + a noun.
        val recallCue = has("did i leave", "did i put", "last seen", "i left", "i put down", "i set down") ||
            (t.startsWith("where") && has("my "))
        if (recallCue) {
            // Strip the recall scaffolding so the noun isn't buried mid-phrase ("... wallet last seen").
            val stripped = t.replace(
                Regex("\\b(where|did|i|leave|left|put|down|set|last|seen|is|are|was|the)\\b"), " ",
            )
            val noun = TargetNoun.extract(stripped) ?: TargetNoun.extract(t)
            if (noun != null && noun !in AMBIGUOUS_SUBJECTS) return VoiceIntent.Recall(noun)
        }

        // 8b. Safety-question fast path -> deterministic deflection, no LLM round-trip. Three
        // whole SEMANTIC FAMILIES, not a phrase list: (a) "<safe/ok/clear> ... <move verb>",
        // (b) "<can/should/may> I <move verb>", (c) "<anything/obstacle/hazard> ... <ahead/in
        // my way>". This is still NOT the safety guarantee — the LLM deflect-flag + the
        // green-light output veto in MainActivity are. The Tier-2 embedding gate replaces this.
        val moveVerb = "(cross|walk|go|move|proceed|continue|step forward|keep going)"
        if (Regex("\\b(safe|okay|ok|clear|alright|good) (to |for )?$moveVerb").containsMatchIn(t) ||
            Regex("\\b(can|could|should|may|shall) i (now )?$moveVerb\\b").containsMatchIn(t) ||
            Regex("\\b(anything|something|any obstacle|obstacles?|hazards?|a step|a drop)\\b.{0,24}\\b(ahead|in front|in (my|the) way|block|there)").containsMatchIn(t) ||
            Regex("\\bis (it|the way|the path|the road|the crosswalk|everything) .{0,20}(safe|clear)\\b").containsMatchIn(t) ||
            has("is it safe", "am i clear", "am i safe", "coast clear", "safe to move", "clear to go")
        ) return VoiceIntent.SafetyCheck

        // 9. Describe the scene.
        if (has("what's ahead", "whats ahead", "what is ahead", "what's around", "whats around",
                "what is around", "what's in front", "in front of me", "what do you see",
                "what can you see", "look around", "describe the scene", "describe what", "what's there") ||
            t == "describe" || t == "scene"
        ) return VoiceIntent.Describe

        // 10. Status / self-report. "where am I" lands here, not Recall. Any mention of battery
        // is a status query (there's no "find the battery" use case here).
        if (has("how am i doing", "where am i", "where are we", "what mode", "which mode",
                "battery", "system status", "give me a status", "how are things") ||
            t == "status" || t == "report"
        ) return VoiceIntent.Status

        // 10b. Phone tasks — call a contact, set a timer.
        if (startsAny("call", "phone", "dial", "ring") && !has("calling", "recalling")) {
            val name = t.removePrefix("please ").split(Regex("\\s+"), limit = 2).getOrNull(1)
                ?.replace(Regex("^(my|up)\\s+"), "")?.trim()
            if (!name.isNullOrBlank()) return VoiceIntent.CallContact(name)
        }
        if (has("timer", "countdown") || (has("set") && has("minute", "second"))) {
            parseDurationSeconds(t)?.let { return VoiceIntent.SetTimer(it) }
        }

        // 11. Explicit find — a seek verb + a target noun.
        val findVerb = startsAny("find", "locate", "search for", "look for", "take me to", "guide me to",
            "go to", "bring me to", "help me find", "i need", "i'm looking for", "im looking for",
            "where is", "where's", "wheres")
        if (findVerb) {
            val noun = TargetNoun.extract(t)
            if (noun != null && noun !in NOT_A_TARGET) return VoiceIntent.Find(noun)
        }

        // 12. Bare noun ("keys", "the exit") — only when the user is actively navigating.
        if (ctx == AppContext.WALKING || ctx == AppContext.STANDING || ctx == AppContext.HOME) {
            val words = t.split(Regex("\\s+"))
            if (words.size <= 4) {
                val noun = TargetNoun.extract(t)
                if (noun != null && noun !in NOT_A_TARGET) return VoiceIntent.Find(noun)
            }
        }

        return VoiceIntent.Unknown(t)
    }

    /** Spoken activity-context phrases -> [AppContext], or null. Kept in sync with the design chat. */
    fun contextPhrase(lc: String): AppContext? = when {
        "sitting" in lc || "sit down" in lc || "sat down" in lc || "i'm resting" in lc || "im resting" in lc ->
            AppContext.SITTING
        "on the bus" in lc || "on a bus" in lc || "on the train" in lc || "on a train" in lc ||
            "in a car" in lc || "in the car" in lc || "getting on" in lc || "transit mode" in lc ->
            AppContext.TRANSIT
        "let's walk" in lc || "lets walk" in lc || "let's go" in lc || "lets go" in lc ||
            "start walking" in lc || "start moving" in lc || "walking mode" in lc ->
            AppContext.WALKING
        "i'm standing" in lc || "im standing" in lc || "i've stopped" in lc || "ive stopped" in lc ||
            "just waiting" in lc || "standing here" in lc ->
            AppContext.STANDING
        "at home" in lc || "my room" in lc || "home mode" in lc || "i'm home" in lc || "im home" in lc ->
            AppContext.HOME
        "talking to someone" in lc || "in a conversation" in lc || "conversation mode" in lc ||
            "quiet mode" in lc ->
            AppContext.CONVERSATION
        else -> null
    }

    private val AMBIGUOUS_SUBJECTS = setOf("am", "i", "we", "here", "there", "you")

    /** Verbs / non-objects that must never be taken as a Find target ("can I walk" -> not find "walk"). */
    private val NOT_A_TARGET = setOf(
        "walk", "go", "move", "cross", "run", "stop", "wait", "proceed", "continue", "step",
        "safe", "clear", "okay", "ok", "now", "there", "here", "ahead", "please", "something",
        "anything", "it", "that", "this", "sound", "cues", "cue", "beeping", "noise", "battery",
        "status", "help", "mode", "volume",
    )

    private val NUMBER_WORDS = mapOf(
        "a" to 1, "an" to 1, "one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5,
        "six" to 6, "seven" to 7, "eight" to 8, "nine" to 9, "ten" to 10, "fifteen" to 15,
        "twenty" to 20, "thirty" to 30, "forty" to 40, "forty-five" to 45, "sixty" to 60, "half" to 0,
    )

    /** "for five minutes" / "10 min" / "90 seconds" / "an hour" -> seconds, or null. */
    fun parseDurationSeconds(t: String): Int? {
        if ("half an hour" in t) return 1800
        if ("an hour" in t || "one hour" in t) return 3600
        val m = Regex("(\\d+|[a-z-]+)\\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)").find(t) ?: return null
        val qty = m.groupValues[1].toIntOrNull() ?: NUMBER_WORDS[m.groupValues[1]] ?: return null
        val unit = m.groupValues[2]
        val secs = when {
            unit.startsWith("h") -> qty * 3600
            unit.startsWith("m") -> qty * 60
            else -> qty
        }
        return secs.takeIf { it in 1..86_400 }
    }
}
