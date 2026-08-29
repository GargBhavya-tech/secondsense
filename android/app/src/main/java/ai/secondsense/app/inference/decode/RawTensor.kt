package ai.secondsense.app.inference.decode

/**
 * A model output tensor, flattened to a FloatArray plus its shape.
 *
 * WHY FLAT + SHAPE (not nested arrays): every runtime — TFLite today, QNN later —
 * can hand back a directly-allocated float buffer. Keeping the shared decode layer
 * (YoloDecoder / DepthSampler) speaking in this one shape means the SAME decode code
 * runs unchanged regardless of who produced the numbers. That's the whole point of
 * the split: the engine is "run the model, give me RawTensors"; the decode layer is
 * "turn RawTensors into Detections". Neither knows about the other's runtime.
 *
 * @param data  row-major flattened values (NHWC / natural tensor order).
 * @param shape the tensor's dimensions, e.g. [1, 8400, 84] or [1, 518, 518].
 */
data class RawTensor(
    val data: FloatArray,
    val shape: IntArray,
) {
    val rank: Int get() = shape.size
    val numElements: Int get() = shape.fold(1) { a, b -> a * b }

    /** True if [dim] appears anywhere in the shape — used for layout sniffing. */
    fun hasDim(dim: Int): Boolean = shape.any { it == dim }

    fun shapeString(): String = shape.joinToString("x", prefix = "[", postfix = "]")

    // data class over an array needs these to behave; we only ever compare by identity
    // in practice, but override to satisfy the contract cleanly.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is RawTensor) return false
        return data.contentEquals(other.data) && shape.contentEquals(other.shape)
    }

    override fun hashCode(): Int = 31 * data.contentHashCode() + shape.contentHashCode()
}
