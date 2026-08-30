package ai.secondsense.app.voice

import ai.secondsense.app.inference.Detection
import kotlin.math.abs

/**
 * Phase 4 — the GROUNDING half, non-QNN path (stand-in for #27 / OWL-ViT).
 *
 * OWL-ViT open-vocabulary grounding has no working TFLite export and the Hexagon NPU is
 * access-gated on this device, so "where is the <thing> I asked for" is answered instead by
 * matching the spoken noun against the object classes the already-running yolo26s detector
 * emits every frame. `Detection.label` is already collapsed to the icon vocab by
 * CocoLabels.toIconVocab, so the match is against that vocab plus raw COCO names.
 *
 * SCOPE, stated honestly: CLOSED-vocabulary — only the ~80 COCO classes (collapsed to
 * person / chair / vehicle / furniture / dog / raw-name). "find the chair" works; "find the
 * door" (door is not a COCO class) grounds nothing. [SYNONYMS] bridges everyday words the user
 * actually says ("bag", "sofa", "phone", "tv") onto the labels the detector emits.
 */
object GoalGrounding {

    /** Spoken word -> the detector labels that should satisfy it. Everyday-speech first. */
    private val SYNONYMS: Map<String, List<String>> = mapOf(
        "bag" to listOf("backpack", "handbag", "suitcase"),
        "backpack" to listOf("backpack"),
        "purse" to listOf("handbag"),
        "handbag" to listOf("handbag"),
        "luggage" to listOf("suitcase"),
        "suitcase" to listOf("suitcase"),
        "phone" to listOf("cell phone"),
        "mobile" to listOf("cell phone"),
        "cellphone" to listOf("cell phone"),
        "sofa" to listOf("chair", "couch"),
        "couch" to listOf("chair", "couch"),
        "seat" to listOf("chair"),
        "stool" to listOf("chair"),
        "bench" to listOf("chair", "bench"),
        "tv" to listOf("furniture", "tv"),
        "television" to listOf("furniture", "tv"),
        "laptop" to listOf("furniture", "laptop"),
        "computer" to listOf("furniture", "laptop"),
        "table" to listOf("furniture", "dining table"),
        "desk" to listOf("furniture", "dining table"),
        "fridge" to listOf("furniture", "refrigerator"),
        "refrigerator" to listOf("furniture", "refrigerator"),
        "water" to listOf("bottle"),
        "bottle" to listOf("bottle"),
        "mug" to listOf("cup"),
        "glass" to listOf("cup", "wine glass"),
        "plant" to listOf("potted plant"),
        "bike" to listOf("vehicle", "bicycle"),
        "cycle" to listOf("vehicle", "bicycle"),
        "car" to listOf("vehicle"),
        "bus" to listOf("vehicle"),
        "truck" to listOf("vehicle"),
        "person" to listOf("person"),
        "people" to listOf("person"),
        "someone" to listOf("person"),
        "man" to listOf("person"),
        "woman" to listOf("person"),
        "dog" to listOf("dog"),
        "puppy" to listOf("dog"),
        "tree" to listOf("potted plant"),
    )

    private fun expand(goal: String): List<String> {
        val g = goal.trim().lowercase()
        val syn = SYNONYMS[g] ?: SYNONYMS[g.removeSuffix("s")]
        return (listOf(g) + (syn ?: emptyList())).distinct()
    }

    private val KNOWN_LABELS = setOf(
        "person", "chair", "vehicle", "furniture", "dog", "bench", "couch", "bed",
        "backpack", "handbag", "suitcase", "bottle", "cup", "book", "clock", "cell phone",
        "laptop", "tv", "keyboard", "mouse", "potted plant", "umbrella", "vase", "bicycle",
        "car", "bus", "truck", "cat", "bird",
    )

    /** True if [goal] is a word we could ground at all (a COCO/icon label or a known synonym). */
    fun isGroundable(goal: String?): Boolean {
        val g = goal?.trim()?.lowercase() ?: return false
        if (g in SYNONYMS || g.removeSuffix("s") in SYNONYMS) return true
        return g in KNOWN_LABELS || g.removeSuffix("s") in KNOWN_LABELS
    }

    /**
     * This frame's best detection for [goal], or null if the goal isn't visible.
     * "Best" = closest to frame center, nearer objects preferred on a tie.
     */
    fun match(detections: List<Detection>, goal: String?): Detection? {
        val g = goal?.trim()?.lowercase() ?: return null
        if (g.isEmpty()) return null
        val targets = expand(g)
        val candidates = detections.filter { d ->
            val l = d.label?.lowercase() ?: return@filter false
            targets.any { t -> l == t || l.contains(t) || t.contains(l) }
        }
        if (candidates.isEmpty()) return null
        return candidates.minByOrNull { 0.5f * abs(it.box.centerX - 0.5f) - it.proximity }
    }
}
