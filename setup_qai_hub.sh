#!/usr/bin/env bash
# SecondSense — Ticket #3: Qualcomm AI Hub toolchain setup (Green Light, laptop)
# Run once per laptop. Verified against qai-hub 0.55.0 / qai-hub-models 0.61.0.
set -euo pipefail

VENV="${1:-$HOME/secondsense-venv}"

echo ">> Creating venv at $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">> Installing pinned toolchain (this pulls torch; large download)"
pip install --upgrade pip
pip install "qai-hub==0.55.0" "qai-hub-models==0.61.0"

echo ">> Sanity checks"
qai-hub --help >/dev/null && echo "   qai-hub CLI OK"
python -c "import qai_hub_models; print('   zoo import OK')"

# Confirm the exact target-model module names resolve (no network needed for this)
python - <<'PY'
import importlib.util as u, os, qai_hub_models
base = os.path.join(os.path.dirname(qai_hub_models.__file__), "models")
targets = ["mobilenet_v3_small","yolov11_det","depth_anything_v2",
           "whisper_tiny","owl_vit","fastsam_x","yamnet","mediapipe_hand"]
missing = [m for m in targets if not os.path.isdir(os.path.join(base, m))]
print("   all target model modules present" if not missing else f"   MISSING: {missing}")
PY

cat <<'NEXT'

>> Toolchain installed. NEXT STEPS (need your Qualcomm account — cannot be scripted):
   1. Get API token at https://aihub.qualcomm.com/  (Settings -> Account)
   2. qai-hub configure --api_token <YOUR_TOKEN>
   3. qai-hub list-devices | grep -i "8 Elite"      # pick the Snapdragon 8 Elite target
   4. python -m qai_hub_models.models.mobilenet_v3_small.export \
        --device "<THAT DEVICE>" --target-runtime qnn_context_binary \
        --output-dir ./export_assets/hello
   See ticket03_qai_hub_toolchain_runbook.md for the full runbook.
NEXT
