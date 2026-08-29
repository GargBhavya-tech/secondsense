# SecondSense — Export Assets

This folder organizes converted models into two distinct runtime tiers:

## 1. `qnn_binaries/` (Qualcomm Hexagon NPU Native)
Pre-compiled Ahead-Of-Time (AOT) context binaries (`.bin`) targeting the **Snapdragon 8 Elite** Hexagon Tensor Processor (HTP).

* [`qnn_binaries/depth_anything_v2/`](./qnn_binaries/depth_anything_v2/)
  * `depth_anything_v2.bin` (52.46 MB) — Monocular relative depth & negative obstacle (curbs/drop-offs) detection.
  * `metadata.json`
* [`qnn_binaries/whisper_tiny/`](./qnn_binaries/whisper_tiny/)
  * `encoder.bin` (19.85 MB) & `decoder.bin` (97.44 MB) — Offline speech-to-text for on-device voice commands.
  * `metadata.json`
* [`qnn_binaries/mobilenet_v3_small/`](./qnn_binaries/mobilenet_v3_small/)
  * `mobilenet_v3_small.bin` (6.32 MB) — Toolchain verification / baseline model.
  * `labels.txt`, `metadata.json`

---

## 2. `tflite_models/` (Universal Cross-Device Models)
TensorFlow Lite flatbuffer models (`.tflite`) that run on **any Android device** (Samsung, Pixel, OnePlus, Xiaomi, MediaTek, Exynos, Google Tensor, PC, emulator) with optional NPU/GPU/NNAPI delegation.

* [`tflite_models/yolov11_det/`](./tflite_models/yolov11_det/)
  * `yolov11_det.tflite` (10.64 MB) — Real-time obstacle detection & classification.
  * `labels.txt` — 80 COCO object classes.
  * `metadata.json`
* [`tflite_models/depth_anything_v2/`](./tflite_models/depth_anything_v2/)
  * `depth_anything_v2.tflite` (98.92 MB) — Relative depth map estimation.
  * `metadata.json`
* [`tflite_models/mobilenet_v3_small/`](./tflite_models/mobilenet_v3_small/)
  * `mobilenet_v3_small.tflite` (10.18 MB) — ImageNet classification / reference model.
  * `labels.txt`, `metadata.json`
