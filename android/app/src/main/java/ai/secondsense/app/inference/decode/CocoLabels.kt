package ai.secondsense.app.inference.decode

/**
 * The 80 COCO classes YOLOv11 predicts (canonical Ultralytics order), plus a mapping
 * into SecondSense's auditory-icon vocabulary (AuditoryIcon.ICON_CLASSES).
 *
 * WHY MAP: the sonification identity channel (#20) has a small set of bespoke icons —
 * person, chair, dog, vehicle, furniture (door comes from the OWL-ViT open-vocab path,
 * not YOLO). Mapping COCO's fine classes onto that vocabulary maximizes icon hits; any
 * class with no mapping passes through with its raw COCO name and the spearcon fallback
 * (sped-up TTS) handles it. Nothing here invents a class the detector didn't emit.
 */
object CocoLabels {

    /** Index-aligned with YOLOv11's class output. Do not reorder. */
    val NAMES: List<String> = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    )

    /**
     * Collapse a COCO class into the SecondSense icon vocabulary where a bespoke icon
     * exists; otherwise return the raw name (spearcon fallback handles it downstream).
     */
    fun toIconVocab(cocoName: String): String = when (cocoName) {
        "person" -> "person"
        "dog" -> "dog"
        "car", "truck", "bus", "motorcycle", "bicycle", "train" -> "vehicle"
        "chair", "couch", "bench", "bed", "toilet" -> "chair"
        "dining table", "tv", "laptop", "refrigerator", "oven", "microwave" -> "furniture"
        else -> cocoName
    }

    /** Safe lookup: out-of-range index -> null (treated as unknown identity, RED-eligible). */
    fun nameForIndex(idx: Int): String? = NAMES.getOrNull(idx)
}
