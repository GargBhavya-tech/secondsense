package ai.secondsense.app.inference.decode

import ai.secondsense.app.inference.BBox
import kotlin.math.max
import kotlin.math.min

/**
 * Turns YOLOv11's raw output tensor(s) into a small list of scored, NMS-filtered boxes
 * in NORMALIZED original-frame coordinates. Runtime-agnostic: it consumes [RawTensor]s,
 * so the exact same decode runs whether TFLite or QNN produced them.
 *
 * THE LAYOUT PROBLEM (read this): qai-hub's `yolov11_det` export can emit its detections
 * in more than one shape depending on version/flags, and the model itself isn't in this
 * repo to introspect. So this decoder SNIFFS the output layout and supports the two forms
 * qai-hub realistically produces, and throws a shape-listing error if it sees neither —
 * which is exactly the signal you want on the first real run (it prints the true shapes
 * so you flip one constant instead of guessing). The two supported forms:
 *
 *   A) THREE tensors — boxes[1,N,4] (xyxy, model px) + scores[1,N] + classes[1,N].
 *      This is qai-hub's shared detection-postprocess contract (decoded, pre-NMS).
 *   B) ONE tensor — [1,84,N] or [1,N,84] raw head: 4 box (cxcywh, model px) + 80 class
 *      scores. Classic Ultralytics layout. We argmax the class block per anchor.
 *
 * If your export turns out to bake NMS in (boxes already final) you can lower NMS_IOU to
 * 1.0 to make NMS a no-op; the score threshold still applies.
 */
object YoloDecoder {

    private const val NUM_CLASSES = 80
    private const val VEC = 4 + NUM_CLASSES // 84

    /** A decoded detection before it's fused with depth. */
    data class RawDet(
        val box: BBox,          // normalized original-frame xyxy
        val score: Float,
        val cocoIndex: Int,
    )

    /**
     * @param outputs      the model's raw output tensors, in the interpreter's order.
     * @param lb           the letterbox transform used for this frame (for back-mapping).
     * @param confThresh   drop detections below this score.
     * @param iouThresh    NMS IoU threshold (set 1.0f to disable if the export pre-NMSes).
     * @param maxDet       cap the returned list (closest/highest-score kept by NMS order).
     */
    fun decode(
        outputs: List<RawTensor>,
        lb: Preprocess.Letterbox,
        confThresh: Float = 0.35f,
        iouThresh: Float = 0.50f,
        maxDet: Int = 32,
    ): List<RawDet> {
        val dets = when {
            looksLikeBoxesScoresClasses(outputs) -> decodeBoxesScoresClasses(outputs, lb, confThresh)
            looksLikeRawHead(outputs) -> decodeRawHead(outputs.first { it.hasDim(VEC) }, lb, confThresh)
            else -> throw IllegalStateException(
                "YoloDecoder: unrecognized output layout. Got ${outputs.size} tensor(s): " +
                    outputs.joinToString { it.shapeString() } +
                    ". Expected either boxes[1,N,4]+scores[1,N]+classes[1,N], or one [1,84,N]/[1,N,84] head. " +
                    "Adjust YoloDecoder to match, or set NMS_IOU=1.0 if the export already NMSes."
            )
        }
        return nms(dets, iouThresh).take(maxDet)
    }

    // ---- layout A: boxes + scores + classes --------------------------------

    private fun looksLikeBoxesScoresClasses(outs: List<RawTensor>): Boolean =
        outs.size >= 3 && outs.any { it.rank == 3 && it.shape.last() == 4 }

    private fun decodeBoxesScoresClasses(
        outs: List<RawTensor>,
        lb: Preprocess.Letterbox,
        confThresh: Float,
    ): List<RawDet> {
        val boxesT = outs.first { it.rank == 3 && it.shape.last() == 4 }
        val n = boxesT.shape[1]
        // scores/classes are the [1,N] float tensors; assume the higher-variance one is
        // scores. To stay robust we take: scores = the [1,N] whose values are in 0..1.
        val flatN = outs.filter { it.numElements == n && it !== boxesT }
        val scoresT = flatN.firstOrNull { allIn01(it.data) } ?: flatN.firstOrNull()
        val classesT = flatN.firstOrNull { it !== scoresT }
            ?: throw IllegalStateException("YoloDecoder(A): missing scores/classes [1,$n] tensors.")

        val out = ArrayList<RawDet>(n)
        for (i in 0 until n) {
            val s = scoresT!!.data[i]
            if (s < confThresh) continue
            val bx = boxesT.data
            val x1 = bx[i * 4]; val y1 = bx[i * 4 + 1]; val x2 = bx[i * 4 + 2]; val y2 = bx[i * 4 + 3]
            val nb = lb.toNormalizedFrame(x1, y1, x2, y2)
            out += RawDet(BBox(nb[0], nb[1], nb[2], nb[3]), s, classesT.data[i].toInt())
        }
        return out
    }

    // ---- layout B: single [1,84,N] / [1,N,84] head -------------------------

    private fun looksLikeRawHead(outs: List<RawTensor>): Boolean =
        outs.any { it.rank == 3 && it.hasDim(VEC) }

    private fun decodeRawHead(
        t: RawTensor,
        lb: Preprocess.Letterbox,
        confThresh: Float,
    ): List<RawDet> {
        // Resolve which axis is the 84-vector and which is the anchor count N.
        val channelsFirst = t.shape[1] == VEC          // [1,84,N]
        val n = if (channelsFirst) t.shape[2] else t.shape[1]
        val d = t.data

        // index helper into the flat [1, A, B] tensor
        fun at(vec: Int, anchor: Int): Float =
            if (channelsFirst) d[vec * n + anchor] else d[anchor * VEC + vec]

        val out = ArrayList<RawDet>(64)
        for (a in 0 until n) {
            // best class over the 80-score block
            var bestC = 0
            var bestS = at(4, a)
            for (c in 1 until NUM_CLASSES) {
                val s = at(4 + c, a)
                if (s > bestS) { bestS = s; bestC = c }
            }
            if (bestS < confThresh) continue
            // box is cxcywh in model pixels
            val cx = at(0, a); val cy = at(1, a); val w = at(2, a); val h = at(3, a)
            val nb = lb.toNormalizedFrame(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f)
            out += RawDet(BBox(nb[0], nb[1], nb[2], nb[3]), bestS, bestC)
        }
        return out
    }

    // ---- shared NMS --------------------------------------------------------

    /** Greedy class-aware NMS over normalized boxes. */
    private fun nms(dets: List<RawDet>, iouThresh: Float): List<RawDet> {
        val sorted = dets.sortedByDescending { it.score }.toMutableList()
        val kept = ArrayList<RawDet>()
        val removed = BooleanArray(sorted.size)
        for (i in sorted.indices) {
            if (removed[i]) continue
            val a = sorted[i]
            kept += a
            for (j in i + 1 until sorted.size) {
                if (removed[j]) continue
                val b = sorted[j]
                if (b.cocoIndex == a.cocoIndex && iou(a.box, b.box) > iouThresh) removed[j] = true
            }
        }
        return kept
    }

    private fun iou(a: BBox, b: BBox): Float {
        val ix1 = max(a.left, b.left); val iy1 = max(a.top, b.top)
        val ix2 = min(a.right, b.right); val iy2 = min(a.bottom, b.bottom)
        val iw = (ix2 - ix1).coerceAtLeast(0f); val ih = (iy2 - iy1).coerceAtLeast(0f)
        val inter = iw * ih
        val union = a.area + b.area - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun allIn01(v: FloatArray): Boolean {
        for (x in v) if (x < -0.001f || x > 1.001f) return false
        return true
    }
}
