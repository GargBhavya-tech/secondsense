package ai.secondsense.app.voice

import ai.secondsense.app.context.AppContext

/**
 * Pure prompt construction + reply parsing for [LlmAssistant]. No Android, no MediaPipe — unit
 * tested off-device so the contract with the model is pinned independently of whether the model
 * is present.
 *
 * The model is instructed to answer with ONE line of JSON. Either it routes the request onto
 * the existing closed action set:
 *   {"action":"find","target":"elevator"}
 *   {"action":"context","context":"transit"}
 *   {"action":"call","name":"mom"}
 *   {"action":"timer","seconds":300}
 * or it answers the user directly from the scene brief:
 *   {"action":"say","text":"There's a chair about two steps to your right."}
 *
 * [parse] is lenient: it finds the first {...} block, tolerates prose around it, and falls back
 * to Speak(rawText) if there's no JSON at all — a small model that just answers in plain text
 * still does something useful rather than nothing.
 */
object LlmPrompt {

    fun build(transcript: String, scene: SceneBrief): String {
        val objs = if (scene.objectsAhead.isEmpty()) "clear" else scene.objectsAhead.joinToString(", ")
        val hazard = scene.hazard ?: "none"
        val batt = if (scene.batteryPct in 0..100) "${scene.batteryPct} percent" else "unknown"
        // Wrapped in Gemma-3's chat template — MediaPipe's generateResponse() does NOT apply it,
        // and Gemma-1B is near-useless without the turn markers. Prompt speaks in 2nd person and
        // leads with ONE worked example; a bare fact-list header made the 1B model reply in the
        // third person ("They can sense..."). JSON is the rare action branch.
        val facts = buildString {
            append("You are walking. ")
            append("In front of you: $objs. ")
            append("Hazard: $hazard. ")
            append("Battery: $batt. ")
            append("Activity mode: ${scene.context}.")
            scene.lastSpoken?.let { append(" You last told them: \"$it\".") }
        }
        return buildString {
            append("<start_of_turn>user\n")
            append("You are an OBJECTIVE ENVIRONMENT DESCRIBER for a blind person. You do NOT have ")
            append("hazard sensors and you are NOT allowed to give safety verdicts or movement ")
            append("advice. You MUST NEVER tell them to walk, stop, cross, or that anything is ")
            append("safe or clear.\n\n")
            append("FACTS you may use: $facts\n\n")
            append("Reply with ONE short sentence spoken directly to them as \"you\". ")
            append("Example — \"how much battery do I have?\" -> \"You have $batt of battery left.\"\n\n")
            append("If they ask ANYTHING about safety, danger, obstacles in their path, or whether ")
            append("they can/should move, walk, cross, or go — reply with EXACTLY this line and ")
            append("nothing else:\n{\"kind\":\"deflect\"}\n\n")
            append("If they are telling the app to DO something (guide me to X, read the sign, ")
            append("what's my status, stop the sounds, I'm on the bus) — reply with ONLY one line ")
            append("like {\"action\":\"find\",\"target\":\"door\"} / {\"action\":\"read\"} / ")
            append("{\"action\":\"status\"} / {\"action\":\"cues_off\"} / ")
            append("{\"action\":\"context\",\"context\":\"transit\"}.\n\n")
            append("They say: \"$transcript\"<end_of_turn>\n")
            append("<start_of_turn>model\n")
        }
    }

    fun parse(rawReply: String?): LlmResolution? {
        if (rawReply.isNullOrBlank()) return null
        // Strip Gemma chat-template artefacts and any echoed role header.
        val reply = rawReply
            .replace("<end_of_turn>", " ")
            .replace("<start_of_turn>", " ")
            .replace(Regex("(?i)^\\s*(model|assistant)\\s*[:\\n]"), "")
            .trim()
        if (reply.isBlank()) return null
        val json = sliceJson(reply)
        if (json == null) {
            // Plain-text answer. Still run the layer-5 veto: if the model free-typed a movement
            // green-light, don't speak it — the caller substitutes a deterministic deflection.
            return if (SafetyGate.looksLikeMovementGreenLight(reply)) LlmResolution.Defer
            else LlmResolution.Speak(reply.take(240))
        }
        val kv = flatFields(json)
        if (kv.isEmpty()) return LlmResolution.Speak(reply.take(240))

        // Layer-4: explicit deflection flag from the constrained prompt.
        val kind = kv["kind"].orEmpty().trim().lowercase()
        if (kind in setOf("deflect", "deflection", "safety", "refuse", "refusal")) return LlmResolution.Defer

        val action = kv["action"].orEmpty().trim().lowercase()
        fun str(vararg keys: String): String? {
            for (k in keys) {
                val v = kv[k]?.trim() ?: continue
                if (v.isEmpty() || v == "null") continue
                // Reject a parroted prompt placeholder ("<contact>", "the thing", "NAME"...).
                if (v.startsWith("<") || v.lowercase() in PLACEHOLDERS) continue
                return v
            }
            return null
        }
        fun int(vararg keys: String): Int? {
            for (k in keys) kv[k]?.trim()?.toDoubleOrNull()?.let { return it.toInt() }
            return null
        }

        return when (action) {
            "find" -> str("target", "thing", "object")?.let { LlmResolution.Action(VoiceIntent.Find(it)) }
            "recall" -> str("target", "thing", "object")?.let { LlmResolution.Action(VoiceIntent.Recall(it)) }
            "describe" -> LlmResolution.Action(VoiceIntent.Describe)
            "read" -> LlmResolution.Action(VoiceIntent.ReadText)
            "status" -> LlmResolution.Action(VoiceIntent.Status)
            "repeat" -> LlmResolution.Action(VoiceIntent.RepeatLast)
            "cues_on", "cues-on", "cueson" -> LlmResolution.Action(VoiceIntent.Cues(on = true))
            "cues_off", "cues-off", "cuesoff", "mute" -> LlmResolution.Action(VoiceIntent.Cues(on = false))
            "pause" -> LlmResolution.Action(VoiceIntent.Pause)
            "resume" -> LlmResolution.Action(VoiceIntent.Resume)
            "cancel" -> LlmResolution.Action(VoiceIntent.CancelSeek)
            "help" -> LlmResolution.Action(VoiceIntent.Help)
            "context" -> parseContext(str("context", "mode"))?.let {
                LlmResolution.Action(VoiceIntent.SwitchContext(it))
            }
            "call" -> str("name", "contact", "target")?.let { LlmResolution.Action(VoiceIntent.CallContact(it)) }
            "timer" -> (int("seconds", "duration"))?.let {
                if (it in 1..86_400) LlmResolution.Action(VoiceIntent.SetTimer(it)) else null
            }
            "say", "answer", "reply", "" -> str("text", "answer", "message")?.let { spoken(it) }
            else -> str("text", "answer")?.let { spoken(it) }
        }
    }

    /** Speak the model's text — unless it reads as a movement green-light, then defer. */
    private fun spoken(text: String): LlmResolution =
        if (SafetyGate.looksLikeMovementGreenLight(text)) LlmResolution.Defer
        else LlmResolution.Speak(text.take(240))

    /**
     * Flat "key": value pairs out of a JSON-ish object body. Deliberately shallow — the reply
     * contract is one flat object — and dependency-free (no org.json, which isn't mocked in JVM
     * unit tests). String values are unescaped; numbers/bools are kept as their literal text.
     */
    private fun flatFields(json: String): Map<String, String> {
        val out = LinkedHashMap<String, String>()
        val re = Regex("\"([A-Za-z_]+)\"\\s*:\\s*(\"((?:[^\"\\\\]|\\\\.)*)\"|-?\\d+(?:\\.\\d+)?|true|false|null)")
        for (m in re.findAll(json)) {
            val key = m.groupValues[1].lowercase()
            val quoted = m.groupValues[3]
            val raw = m.groupValues[2]
            out[key] = if (raw.startsWith("\"")) {
                quoted.replace("\\\"", "\"").replace("\\n", " ").replace("\\t", " ").replace("\\\\", "\\")
            } else raw
        }
        return out
    }

    /** First balanced {...} run in [s], or null. Tolerates markdown fences / prose around it. */
    private fun sliceJson(s: String): String? {
        val start = s.indexOf('{')
        if (start < 0) return null
        var depth = 0
        var inStr = false
        var esc = false
        for (i in start until s.length) {
            val c = s[i]
            when {
                esc -> esc = false
                c == '\\' && inStr -> esc = true
                c == '"' -> inStr = !inStr
                !inStr && c == '{' -> depth++
                !inStr && c == '}' -> {
                    depth--
                    if (depth == 0) return s.substring(start, i + 1)
                }
            }
        }
        return null
    }

    private val PLACEHOLDERS = setOf(
        "contact", "contact name", "name", "thing", "the thing", "object", "the object",
        "target", "the target", "some object", "an object",
    )

    private fun parseContext(v: String?): AppContext? = when (v?.lowercase()?.trim()) {
        "walking", "walk" -> AppContext.WALKING
        "standing", "stand", "stopped" -> AppContext.STANDING
        "home" -> AppContext.HOME
        "sitting", "sit", "seated" -> AppContext.SITTING
        "transit", "vehicle", "bus", "train", "car" -> AppContext.TRANSIT
        "conversation", "talking", "quiet" -> AppContext.CONVERSATION
        else -> null
    }
}
