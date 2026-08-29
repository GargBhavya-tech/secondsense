# SecondSense — Ticket #3: Qualcomm AI Hub toolchain setup for Windows PowerShell
# Run once per machine. Verified against qai-hub 0.55.0 / qai-hub-models 0.61.0 on Python 3.12.
param (
    [string]$VenvPath = "$HOME\secondsense-venv"
)

$ErrorActionPreference = "Stop"

Write-Host ">> Creating virtual environment at $VenvPath using Python 3.12..."
py -3.12 -m venv $VenvPath

$pip = Join-Path $VenvPath "Scripts\pip.exe"
$python = Join-Path $VenvPath "Scripts\python.exe"
$qaiHub = Join-Path $VenvPath "Scripts\qai-hub.exe"

Write-Host ">> Upgrading pip..."
& $pip install --upgrade pip

Write-Host ">> Installing pinned toolchain (qai-hub==0.55.0, qai-hub-models==0.61.0)..."
Write-Host "   (Note: this downloads PyTorch and pre-trained dependencies; may take a few minutes)"
& $pip install "qai-hub==0.55.0" "qai-hub-models==0.61.0"

Write-Host ">> Running Sanity Checks..."
& $qaiHub --help | Out-Null
Write-Host "   [PASS] qai-hub CLI OK"

& $python -c "import qai_hub_models; print('   [PASS] zoo import OK')"

Write-Host ">> Verifying target model zoo modules..."
& $python -c @"
import os, sys, qai_hub_models
base = os.path.join(os.path.dirname(qai_hub_models.__file__), 'models')
targets = ['mobilenet_v3_small','yolov11_det','depth_anything_v2',
           'whisper_tiny','owl_vit','fastsam_x','yamnet','mediapipe_hand']
missing = [m for m in targets if not os.path.isdir(os.path.join(base, m))]
if missing:
    print(f'   [FAIL] MISSING: {missing}')
    sys.exit(1)
else:
    print('   [PASS] All target model modules present')
"@

Write-Host ">> Running scripts/verify_setup.py..."
& $python scripts/verify_setup.py

Write-Host @"

>> Toolchain installed and verified!
To activate this environment in PowerShell:
   & `"$VenvPath\Scripts\Activate.ps1`"
Or run scripts directly with:
   & `"$python`" convert.py --dry-run
"@
