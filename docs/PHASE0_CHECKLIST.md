# Phase 0 → 1 — Master Checklist

Status of all Phase 0 tickets and Phase 1 model conversion work.

| # | Ticket | Type | Status | Where |
|---|---|---|---|---|
| 1 | Ear-Occlusion / Headphone Decision | Decision | ✅ decided | `docs/ticket01_headphone_decision.md` |
| 2 | Submission Logistics | Decision | ✅ fill-in | `docs/ticket02_submission_logistics.md` |
| 3 | Qualcomm AI Hub Toolchain | Code / setup | ✅ **COMPLETE** | `setup_qai_hub.ps1` + `convert.py` + `docs/ticket03_...runbook.md` |
| 4 | Office Kit Workflow | Rehearsal | ✅ checklist | `docs/ticket04_office_kit_rehearsal.md` |
| 5 | Mount + Demo Course | Physical | ✅ checklist | `docs/ticket05_mount_and_course.md` |
| 6 | On-Device App Skeleton + Camera | Phone code | ✅ **COMPLETE** | `android/` — camera + mock engine + spatial audio + haptics + HUD |
| 7 | One-Tap Calibration | Phone code | ⏳ in-app wiring | On-device when phone is in hand |

---

## Phase 1 — Model Conversion (COMPLETE)

| Model | QNN Binary | TFLite | Wired to App? |
|---|---|---|---|
| `mobilenet_v3_small` | ✅ `qnn_binaries/` | ✅ `tflite_models/` | Reference / hello-world |
| `yolov11_det` | ⚠️ QNN link failed | ✅ `tflite_models/` | **YES — TFLite engine** |
| `depth_anything_v2` | ✅ `qnn_binaries/` | ✅ `tflite_models/` | **YES — TFLite engine** |
| `whisper_tiny` | ✅ `qnn_binaries/` (encoder + decoder) | ❌ runtime unsupported | QNN path only |
| `owl_vit` | ❌ QNN link failed; skipped | — | Phase 4 |

All downloaded assets are in `export_assets/`. See `export_assets/README.md` for details.

---

## TFLite Test-Path — COMPLETE

The full vision + sonification stack now runs on ANY Android device or emulator:

- `android/app/src/main/java/.../inference/TfliteInferenceEngine.kt` — real model runner
- `android/app/src/main/java/.../inference/EngineConfig.kt` — `MOCK` / `TFLITE` / `QNN` switch
- `android/app/src/main/java/.../inference/decode/` — shared decode layer (runtime-agnostic):
  - `RawTensor.kt`, `CocoLabels.kt`, `Preprocess.kt`, `YoloDecoder.kt`, `DepthSampler.kt`, `MotionTracker.kt`
- `android/app/src/main/assets/models/yolov11_det.tflite` (10.6 MB) ✅
- `android/app/src/main/assets/models/depth_anything_v2.tflite` (98.9 MB) ✅

To activate: change `EngineConfig.KIND = Kind.TFLITE` (one line). Default remains `MOCK` so
the app always builds even with no models present.

See `TFLITE_INTEGRATION.md` for the full integration guide.

---

## Recommended order before doors open

1. ✅ `./setup_qai_hub.ps1` on all laptops
2. ✅ Get Qualcomm token + `qai-hub configure`
3. ✅ `python convert.py --dry-run`, then hello-world, then full spine
4. ✅ TFLite models downloaded + wired into Android app
5. Open `android/` in Android Studio → sync Gradle → run on any phone/emulator
6. Flip `EngineConfig.KIND = Kind.TFLITE` → verify logcat `SecondSense/tflite` shows real model shapes
7. When iQOO 15 is in hand: implement `QnnInferenceEngine`, deploy QNN `.bin` files, flip to `Kind.QNN`
