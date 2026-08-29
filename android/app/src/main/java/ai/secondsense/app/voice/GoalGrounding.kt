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
 * SCOPE, stated honestly: CLOSED-vocabulary — only the ~80 COCO classes. "find the chair"
 * works; "find the door" (door is not a COCO class) spots the word but grounds nothing.
 * True open-vocabulary grounding is roadmap, waiting on the NPU bridge.
 */
object GoalGrounding {

    /**
     * This frame's best detection for [goal], or null if the goal isn't visible.
     * "Best" = closest to frame center, nearer objects preferred on a tie — the same intent
     * as TargetSelector, scoped to the single class the user asked for.
     *
     * @param goal the spoken target noun, already lower-cased/extracted (TargetNoun.extract).
     */
    fun match(detections: List<Detection>, goal: String?): Detection? {
        val g = goal?.trim()?.lowercase() ?: return null
        if (g.isEmpty()) return null
        val candidates = detections.filter { d ->
            val l = d.label?.lowercase() ?: return@filter false
            l == g || l.contains(g) || g.contains(l)
        }
        if (candidates.isEmpty()) return null
        // lower is better: nearest wins (you want to walk to the closest one), with centering
        // as a tie-breaker.
        return candidates.minByOrNull { 0.5f * abs(it.box.centerX - 0.5f) - it.proximity }
    }
}
