package ai.secondsense.app.voice

import ai.secondsense.app.inference.decode.HazardState

/**
 * Tier-1 deterministic safety gate for free-form spoken questions (deep-research architecture,
 * layers 1 + 5). A quantized 1B LLM must NEVER be the authority on "can I move" — it hedges and
 * hallucinates green-lights. This object holds the two pure pieces MainActivity needs:
 *
 *  - [mostSevere] — collapse a rolling window of hazard-state samples to the single WORST one,
 *    so a transient DROP_CONFIRMED that flickered mid-question is still the answer (temporal
 *    hysteresis, ~1.5 s window owned by the caller).
 *  - [looksLikeMovementGreenLight] — detect that the LLM's OWN output is telling the user it's
 *    fine to walk/cross. This is a regex over our generated-answer space (small, enumerable),
 *    NOT over the user's unbounded input. Any hit is overridden by the deterministic template.
 *
 * The multilingual-e5-small embedding gate (the robust primary defence) is Tier 2 and lives
 * elsewhere; this is the zero-new-model backstop.
 */
object SafetyGate {

    /** DROP_CONFIRMED > SCENE_NOT_TRAVERSABLE > SENSOR_BLOCKED > POSSIBLE_DROP > SAFE > null. */
    private val SEVERITY = listOf(
        HazardState.DROP_CONFIRMED,
        HazardState.SCENE_NOT_TRAVERSABLE,
        HazardState.SENSOR_BLOCKED,
        HazardState.POSSIBLE_DROP,
        HazardState.SAFE,
    )

    fun mostSevere(states: Iterable<HazardState?>): HazardState? {
        var best: HazardState? = null
        var bestRank = Int.MAX_VALUE
        for (s in states) {
            val r = SEVERITY.indexOf(s ?: continue)
            if (r in 0 until bestRank) { bestRank = r; best = s }
        }
        return best
    }

    private val GREEN_LIGHT = listOf(
        Regex("\\byou can (now )?(go|walk|cross|move|proceed|step)\\b"),
        Regex("\\b(it'?s|it is|you'?re|you are) (safe|clear|fine|ok|okay|good)( to| for)?\\b"),
        Regex("\\b(go ahead|go for it|off you go|feel free) (and )?(walk|cross|go|move|now)?\\b"),
        Regex("\\b(safe|clear|fine|okay) to (walk|cross|go|move|proceed)\\b"),
        Regex("\\b(the )?(path|way|road|coast) (is|looks) (clear|safe)\\b"),
        Regex("\\bno (obstacles|hazards|danger|cars|traffic)\\b.*\\b(cross|walk|go|proceed)\\b"),
        Regex("^\\s*(yes|yep|yeah|sure)\\b[.! ]*(you can)?\\s*(walk|cross|go|proceed)?\\s*[.!]?\\s*$"),
    )

    /** True if [text] (an LLM answer) tells the user it's OK to move. Case-insensitive. */
    fun looksLikeMovementGreenLight(text: String?): Boolean {
        val t = text?.lowercase()?.trim() ?: return false
        if (t.isEmpty()) return false
        return GREEN_LIGHT.any { it.containsMatchIn(t) }
    }
}
