# SecondSense — TFLite test-path integration

Drop-in for running the **real** YOLOv11 + Depth-Anything-V2 models as `.tflite` on
CPU/GPU, behind the existing `InferenceEngine` seam. Nothing downstream of the seam
changes. A future `QnnInferenceEngine` reuses the same `inference/decode/` layer.

## Files in this patch

New (copy into the repo at the mirrored paths):
- `inference/decode/RawTensor.kt` — flat tensor the shared layer speaks in
- `inference/decode/CocoLabels.kt` — 80-class list + SecondSense icon-vocab mapping
- `inference/decode/Preprocess.kt` — letterbox → float buffer, with box back-mapping
- `inference/decode/YoloDecoder.kt` — layout-sniffing decode + NMS (shape-guarded)
- `inference/decode/DepthSampler.kt` — per-frame normalize → per-box relative proximity
- `inference/decode/MotionTracker.kt` — coarse moving/approaching flags
- `inference/TfliteInferenceEngine.kt` — loads + runs the two models, calls the decode layer
- `inference/EngineConfig.kt` — the engine switch (MOCK / TFLITE / QNN)

Changed (overwrite the existing files):
- `app/build.gradle.kts` — tflite dep, `x86_64` ABI for emulator, `noCompress += "tflite"`
- `app/src/main/java/.../ui/MainActivity.kt` — one-line swap to `EngineConfig.create(this)`

## Steps

1. **Copy the models** into `app/src/main/assets/models/`:
   - `yolov11_det.tflite`
   - `depth_anything_v2.tflite`
   (These are the tflite exports listed in `conversion_manifest.json`. YOLO already exists
   only as tflite; depth exists as both — use the `.tflite`.)

2. **Flip the switch** in `EngineConfig.kt`:
   ```kotlin
   val KIND: Kind = Kind.TFLITE
   ```
   Default is `MOCK` so the app still builds/runs with no models. If `TFLITE` is set but the
   assets are missing, it logs and falls back to `MOCK` — the demo never hard-crashes on a
   missing file.

3. **Sync + run.** Watch logcat, tag `SecondSense/tflite`, on the first run. It prints the
   real input/output shapes and dtypes of both models. This is your ground truth.

## The one thing to verify on first run (important)

I could not introspect the actual `.tflite` (only `yolo11n.pt` shipped in the zip), so the
YOLO output layout is **sniffed**, not assumed. The decoder handles the two forms qai-hub
realistically emits and **throws with the real shapes** if it sees neither. When you run it:

- If detections look right → the sniffed layout matched, you're done.
- If it throws `YoloDecoder: unrecognized output layout … [shapes]` → read the shapes it
  printed and either (a) the export baked in NMS (set `iouThreshold = 1.0f` in the engine
  to make NMS a no-op), or (b) adjust `YoloDecoder` to the printed layout. One constant,
  not a rewrite.

Two more assumptions worth a glance against the logged shapes/behavior:
- **Depth near/far ordering.** Depth-Anything-V2 is inverse-depth (**larger = nearer**), so
  `DepthSampler(nearIsHigh = true)`. If proximity feels inverted (things read "far" as you
  approach), flip it to `false` — single switch.
- **Depth input normalization.** Preprocess feeds 0..1 RGB. If depth output looks like noise,
  the export may want ImageNet mean/std; add it in `Preprocess` (the `normalizeTo01` path is
  where it goes).

## Keep tflite out of the latency metrics (#37)

This is a **behavior** test harness, not the NPU. CPU/GPU tflite timings are not the bible's
~20 ms YOLO / ~5 ms depth NPU figures. Measure the #37 latency table on the QNN path; use
tflite for logic/UX validation on an emulator or any device.

## Voice path (Phase 4) still needs tflite exports

The manifest has `whisper_tiny` and `owl_vit` as **QNN only** (no tflite). To test the
voice co-headline on this path, re-export them:
```
python convert.py --only whisper_tiny --runtime tflite --device "<your device>"
python convert.py --only owl_vit      --runtime tflite --device "<your device>"
```

## Not included on purpose

- **Frame rotation.** `FrameAnalyzer` still hands the engine the raw (sensor-orientation)
  bitmap. For real models this must be rotated by `image.imageInfo.rotationDegrees` first,
  or every box — and the pan/azimuth cue — is on the wrong axis. `Preprocess` assumes an
  upright bitmap. Fix in the analyzer; it's a small change but it's a real prerequisite.
- **Drop-off / negative-obstacle (#17).** Depth-based, separate ticket; not wired here.
- **QNN engine (#8–#11).** When binaries land, `QnnInferenceEngine` implements the same
  `InferenceEngine`, produces `RawTensor`s, and calls `YoloDecoder`/`DepthSampler`/
  `MotionTracker` unchanged. Then set `EngineConfig.KIND = QNN`.
