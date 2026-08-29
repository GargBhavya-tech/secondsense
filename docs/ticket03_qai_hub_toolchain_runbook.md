# Ticket #3 — Qualcomm AI Hub Toolchain ✅ COMPLETE

**Phase 1 gate · 🟢 GREEN (laptop) · Status: ALL DONE**
Companion to `secondsense_build_map_v3.md` #3 and `secondsense_bible_v4.md` §23.4.

**Goal:** a proven, repeatable command sequence on the laptop that gets an NPU-native model onto the phone's chipset. Everything downstream (Phase 1 model conversion #8–#11) is blocked on this working.

> **✅ COMPLETED 2026-08-27.** All Phase 1 core models have been converted and downloaded. See the model inventory below and `conversion_manifest.json` for the full record.

---

## What's verified vs. what you must do

**Verified on a clean machine (grounded, not guessed):**
- `pip install qai-hub qai-hub-models` installs cleanly from PyPI. Versions used: **qai-hub 0.55.0**, **qai-hub-models 0.61.0** (pulls torch ~2.x).
- The CLI exposes: `configure`, `list-devices`, `list-frameworks`, `submit-compile-job`, `submit-profile-job`, `submit-compile-and-profile-jobs`, `submit-link-job`.
- **All your target models exist in the zoo** under these exact names (this matters — a wrong name wastes a Green window):

  | Bible name | Exact zoo module | Note |
  |---|---|---|
  | YOLOv11 | `yolov11_det` | detection variant (also: `_pose`, `_seg`) |
  | Depth-Anything-V2 | `depth_anything_v2` | (`depth_anything_v3` also exists if you want to A/B) |
  | Whisper-Tiny | `whisper_tiny` | **not** `whisper_tiny_en` — that name fails |
  | OWL-ViT | `owl_vit` | present |
  | FastSam | `fastsam_x` | zoo ships `fastsam_x`, **not** `fastsam_s` — adjust the bible's "FastSam-S" |
  | YamNet | `yamnet` | present |
  | MediaPipe Hand | `mediapipe_hand` | present |
  | hello-world | `mobilenet_v3_small` | use this to prove the plumbing first |

  Bonus find: `yolo_world` (open-vocabulary detector) is in the zoo. Not your chosen path (bible = Whisper+OWL-ViT), but worth knowing it exists as a fallback for the open-vocab moat if OWL-ViT gives trouble.

- **NPU-native output = the `qnn_context_binary` runtime** (the export `--target-runtime` choices are: `tflite, qnn_dlc, qnn_context_binary, onnx, precompiled_qnn_onnx`). For the Hexagon NPU you want `qnn_context_binary` (or `qnn_dlc`). `tflite` runs via a delegate and is the easy fallback if a QNN compile balks.

**You must do (needs your Qualcomm account — this sandbox can't reach Qualcomm's servers):**
- Create the account, get the API token, and run every compile/profile job. The toolchain calls Qualcomm's device farm *before it even traces the model*, so nothing runs without the token. This is the exact auth wall the ticket's "Watch out" warns about — clear it early, not mid-event.

---

## Setup script (run once on the laptop)

```bash
# 1. Isolated env (don't fight system Python mid-hackathon)
python3 -m venv ~/secondsense-venv
source ~/secondsense-venv/bin/activate     # Windows: ~\secondsense-venv\Scripts\activate

# 2. Pin versions for reproducibility across all 3 laptops
pip install "qai-hub==0.55.0" "qai-hub-models==0.61.0"

# 3. Sanity check
qai-hub --help
python -c "import qai_hub_models; print('zoo OK')"
```

---

## Runbook — zero to first NPU model (do in this order)

### Step 1 — Account + token (the auth wall)
1. Request access at **https://aihub.qualcomm.com/** and sign in.
2. Settings → Account → copy your **API token**.
3. Configure the client:
   ```bash
   qai-hub configure --api_token <YOUR_TOKEN>
   ```
   This writes `~/.qai_hub/client.ini`. Verify:
   ```bash
   qai-hub list-devices          # should now return a device table, not an auth error
   ```
   If this errors, STOP — nothing else works until it doesn't. Don't proceed to Step 2.

### Step 2 — Find the right target device/chipset
The iQOO 15 runs **Snapdragon 8 Elite Gen 5**. The farm may not contain the iQOO 15 itself — that's fine: **you compile for the chipset, and the binary runs on any phone with that chip.** Find the target string:
```bash
qai-hub list-devices --device-attr chipset:qualcomm-snapdragon-8-elite
# or browse the whole list and pick the newest 8-Elite phone:
qai-hub list-devices | grep -i "8 Elite"
```
Write down the exact `--device "…"` name (or the `--chipset …` id) you'll reuse for every model. If no 8-Elite-Gen-5 device is listed yet, target the newest available 8-Elite device — same NPU family, valid binary; profiling numbers just come from that farm device.

### Step 3 — Prove the plumbing with the smallest model FIRST
Do **not** start with YOLOv11. Prove auth + compile + profile + download works end-to-end with a tiny model:
```powershell
# Use convert.py (auto-routes to qnn_binaries/ or tflite_models/ based on --runtime)
python convert.py --device "Snapdragon 8 Elite QRD" --only mobilenet_v3_small
```
Or directly:
```bash
python -m qai_hub_models.models.mobilenet_v3_small.export \
  --device "Snapdragon 8 Elite QRD" \
  --target-runtime qnn_context_binary \
  --output-dir ./export_assets/qnn_binaries/mobilenet_v3_small
```
Watch the job link it prints (runs on aihub.qualcomm.com). **Done with Step 3 when** you have a compiled `.bin` asset downloaded under `export_assets/qnn_binaries/mobilenet_v3_small/`.

> **✅ DONE:** `mobilenet_v3_small.bin` (6.32 MB) in `export_assets/qnn_binaries/mobilenet_v3_small/`.

### Step 4 — First real target: YOLOv11
```powershell
python convert.py --device "Snapdragon 8 Elite QRD" --only yolov11_det
```
**Note:** `yolov11_det` QNN context binary link failed (exit code 14 — unsupported op). The fallback is TFLite (still NPU-delegate accelerated, runs everywhere):
```powershell
python convert.py --device "Snapdragon 8 Elite QRD" --runtime tflite --only yolov11_det
```
> **✅ DONE (TFLite):** `yolov11_det.tflite` (10.64 MB) in `export_assets/tflite_models/yolov11_det/`. **Wired into the Android app's `TfliteInferenceEngine`.**

### Step 4b — Depth-Anything-V2 and Whisper-Tiny
```powershell
# QNN native (Snapdragon 8 Elite Hexagon NPU)
python convert.py --device "Snapdragon 8 Elite QRD" --only depth_anything_v2 whisper_tiny

# TFLite universal (depth works; whisper_tiny tflite export is unsupported — QNN only)
python convert.py --device "Snapdragon 8 Elite QRD" --runtime tflite --only depth_anything_v2
```
> **✅ DONE:**
> - `depth_anything_v2.bin` (52.46 MB) in `export_assets/qnn_binaries/depth_anything_v2/` + `.tflite` (98.92 MB) in `export_assets/tflite_models/depth_anything_v2/`. **Wired into the app.**
> - `whisper_tiny` encoder.bin (19.85 MB) + decoder.bin (97.44 MB) in `export_assets/qnn_binaries/whisper_tiny/`.

### Step 5 — Lock it in ✅
`convert.py` IS the deliverable. Reproduce any model with:
```powershell
# QNN (Snapdragon 8 Elite native)
python convert.py --device "Snapdragon 8 Elite QRD" --only <model>

# TFLite (any device)
python convert.py --device "Snapdragon 8 Elite QRD" --runtime tflite --only <model>
```
Outputs automatically route to `export_assets/qnn_binaries/<model>/` or `export_assets/tflite_models/<model>/`.

---

## Gotchas worth pre-empting
- **Whisper-Tiny and OWL-ViT are multi-component models** (encoder/decoder; text+image towers). Their export may produce several assets and take longer — budget more time for #10/#11 than for #8/#9.
- **First run downloads pretrained weights** (hundreds of MB per model) from Qualcomm/HF asset servers — make sure the venue network allows it, or pre-download at home. This is a classic "works at home, dies on venue wifi" trap.
- **QNN compile occasionally fails on an op** a model uses; the documented fallback is `--target-runtime tflite` (still on-NPU via delegate, slightly slower). Have that fallback in your pocket for the demo rather than debugging a QNN op mid-event.
- **Pin versions across all three laptops.** A teammate on a different qai-hub-models version can get different export flags — the exact "not found" name traps above are version-sensitive.

---

## Done-when checklist ✅ ALL COMPLETE
- [x] `qai-hub configure` done; `qai-hub list-devices` returns 79 devices including Snapdragon 8 Elite
- [x] Target device: **`"Snapdragon 8 Elite QRD"`** — used for all conversions
- [x] `mobilenet_v3_small` exported: QNN `.bin` (6.32 MB) + TFLite `.tflite` (10.18 MB)
- [x] `yolov11_det` exported: TFLite `.tflite` (10.64 MB) — QNN link failed, TFLite is the app path
- [x] `depth_anything_v2` exported: QNN `.bin` (52.46 MB) + TFLite `.tflite` (98.92 MB)
- [x] `whisper_tiny` exported: QNN encoder.bin (19.85 MB) + decoder.bin (97.44 MB)
- [x] `convert.py` is the deliverable (auto-routing to `qnn_binaries/` or `tflite_models/`)
- [x] Versions pinned: **qai-hub 0.55.0 / qai-hub-models 0.61.0 / Python 3.12**
- [x] Both `.tflite` models bundled into `android/app/src/main/assets/models/`
- [x] `TfliteInferenceEngine` wired into Android app; `EngineConfig.KIND = TFLITE` to activate
