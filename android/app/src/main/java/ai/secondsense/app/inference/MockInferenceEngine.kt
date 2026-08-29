package ai.secondsense.app.inference

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random

/**
 * A stand-in for the real NPU pipeline (YOLOv11 + Depth-Anything-V2) that runs
 * with NO model binaries. It exists so tickets #14–#28 can be built and tested
 * on the laptop/emulator before conversion (tickets #8–#11) finishes.
 *
 * It does NOT look at pixels. It scripts a small, deterministic-ish scene that
 * exercises every downstream branch the Bible cares about:
 *   - an object drifting left↔right across center   -> exercises HRTF pan (#18)
 *   - proximity oscillating near↔far                -> exercises pulse-rate (#19) + haptics (#21)
 *   - a periodic identity change (person/chair/dog)  -> exercises auditory icons (#20)
 *   - a periodic confidence drop WHITE->BLUE->RED    -> exercises the tier layer (#23)
 *   - a moving flag that toggles                     -> exercises static/dynamic priority (#15)
 *   - occasional "identity unknown but depth present"-> exercises RED / proximity-only
 *
 * Everything here is REPLACED by QnnInferenceEngine at #12; the contract
 * (InferenceEngine + FrameResult) is identical, so nothing downstream changes.
 */
class MockInferenceEngine(
    private val seed: Long = 42L,
) : InferenceEngine {

    override val name: String = "mock"

    private var frameIndex: Long = 0
    private val rng = Random(seed)
    private val labels = listOf("person", "chair", "dog", "vehicle", "door")

    override fun initialize() {
        // Nothing to load. Simulate a tiny warmup so the "engine ready" path is real.
        frameIndex = 0
    }

    override fun infer(frame: Bitmap, centerCrop: Boolean): FrameResult {
        val started = System.nanoTime()
        frameIndex++

        // Phase drives all the periodic behavior. One full cycle ~ every 120 frames.
        val t = frameIndex / 120.0

        // --- primary target: drifts horizontally, oscillates in proximity ---
        val centerX = (0.5f + 0.35f * sin(t * 2 * Math.PI).toFloat()).coerceIn(0.05f, 0.95f)
        val proximity = (0.5f + 0.45f * sin(t * 2 * Math.PI * 0.7).toFloat()).coerceIn(0f, 1f)
        val approaching = 0.45f * kotlin.math.cos(t * 2 * Math.PI * 0.7).toFloat()

        // identity cycles slowly through the class list
        val labelIdx = ((frameIndex / 90) % labels.size).toInt()
        val label = labels[labelIdx]

        // confidence tier cycles WHITE -> BLUE -> RED on a slow clock so the
        // uncertainty layer (#23) has something to react to during dev.
        val tierPhase = (frameIndex / 200) % 3
        val tier = when (tierPhase) {
            0L -> ConfidenceTier.WHITE
            1L -> ConfidenceTier.BLUE
            else -> ConfidenceTier.RED
        }
        val score = when (tier) {
            ConfidenceTier.WHITE -> 0.85f + rng.nextFloat() * 0.1f
            ConfidenceTier.BLUE -> 0.45f + rng.nextFloat() * 0.15f
            ConfidenceTier.RED -> 0.2f + rng.nextFloat() * 0.1f
        }

        // RED tier = depth present, identity NOT claimed (Bible §5.3).
        val emittedLabel = if (tier == ConfidenceTier.RED) null else label

        // coarse moving/static flag (#15): toggles on a slow clock.
        val moving = ((frameIndex / 150) % 2) == 0L

        val halfW = 0.10f
        val halfH = 0.18f
        val box = BBox(
            left = (centerX - halfW).coerceIn(0f, 1f),
            top = (0.5f - halfH).coerceIn(0f, 1f),
            right = (centerX + halfW).coerceIn(0f, 1f),
            bottom = (0.5f + halfH).coerceIn(0f, 1f),
        )

        val primary = Detection(
            label = emittedLabel,
            score = score,
            box = box,
            proximity = proximity,
            approaching = approaching,
            moving = moving,
            tier = tier,
        )

        // --- a static peripheral object, to prove center-crop actually filters it (#14) ---
        val peripheral = Detection(
            label = "chair",
            score = 0.8f,
            box = BBox(0.02f, 0.4f, 0.14f, 0.75f),
            proximity = 0.3f,
            approaching = 0f,
            moving = false,
            tier = ConfidenceTier.WHITE,
        )

        val all = mutableListOf(primary, peripheral)
        val visible = if (centerCrop) all.filter { isCentered(it.box) } else all

        val elapsedMs = (System.nanoTime() - started) / 1_000_000
        return FrameResult(
            detections = visible,
            frameWidth = frame.width,
            frameHeight = frame.height,
            inferenceMillis = elapsedMs,
            depthAvailable = true,
        )
    }

    /** Center ~30% band test, mirrors the intent of ticket #14. */
    private fun isCentered(b: BBox): Boolean {
        val cx = b.centerX
        return abs(cx - 0.5f) <= 0.15f   // within center 30% horizontally
    }

    override fun close() { /* no native resources */ }
}
