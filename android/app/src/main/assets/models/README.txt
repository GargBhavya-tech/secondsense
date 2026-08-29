Place the tflite models here:

  yolov11_det.tflite      <- NOTE: this filename is kept for code stability, but the
                             bundled model is now yolo26_det (small / yolo26s.pt), which
                             tests markedly more confident than the old yolov11n on real
                             photos. Same output contract (boxes/scores/class_idx), so no
                             decode changes. Re-export with:
                               python -m qai_hub_models.models.yolo26_det.export \
                                 --device "Snapdragon 8 Elite QRD" --target-runtime tflite \
                                 --ckpt-name yolo26s.pt --output-dir export_assets/...
  depth_anything_v2.tflite

These are the tflite exports from conversion_manifest.json. They must stay uncompressed
in the APK — that's handled by `androidResources { noCompress += "tflite" }` in
app/build.gradle.kts, so the engine can memory-map them.

With the files present, set EngineConfig.KIND = Kind.TFLITE. Without them, the app falls
back to the mock engine automatically.
