#!/usr/bin/env python3
"""
SecondSense — verify the Qualcomm AI Hub toolchain is ready.

Runs OFFLINE checks (install + model names) always, and the AUTH/network check
only if a token is configured. Safe to run anytime; tells you the next action.

    python scripts/verify_setup.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError


def pkg_version(dist):
    try:
        return version(dist)
    except PackageNotFoundError:
        return None

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["mobilenet_v3_small", "yolov11_det", "depth_anything_v2",
           "whisper_tiny", "owl_vit", "yamnet", "fastsam_x"]
CONFIG_INI = Path.home() / ".qai_hub" / "client.ini"

ok = True

def check(label, cond, hint=""):
    global ok
    mark = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"  [{mark}] {label}" + (f"  -> {hint}" if (hint and not cond) else ""))
    return cond

print("== 1. Toolchain install ==")
try:
    import qai_hub
    check(f"qai_hub {pkg_version('qai-hub') or '?'}", True)
except Exception as e:
    check("qai_hub import", False, f"pip install qai-hub==0.55.0  ({e})")

zoo = None
try:
    import qai_hub_models
    check(f"qai_hub_models {pkg_version('qai-hub-models') or '?'}", True)
    zoo = Path(qai_hub_models.__file__).parent / "models"
except Exception as e:
    check("qai_hub_models import", False, f"pip install qai-hub-models==0.61.0  ({e})")

print("\n== 2. Target model modules present (exact names) ==")
if zoo:
    for m in TARGETS:
        check(m, (zoo / m).is_dir(), "wrong name / version mismatch")

print("\n== 3. Auth / device access ==")
if not CONFIG_INI.exists():
    print(f"  [TODO] no token yet: {CONFIG_INI} missing")
    print("         -> get token at https://aihub.qualcomm.com/ , then:")
    print("            qai-hub configure --api_token <TOKEN>")
    print("         (offline checks above still tell you the install is fine.)")
else:
    print(f"  [ok] token file present: {CONFIG_INI}")
    try:
        import qai_hub
        devs = qai_hub.get_devices()
        print(f"  [PASS] reached AI Hub — {len(devs)} devices visible")
        elite = [d.name for d in devs if "8 Elite" in d.name]
        print(f"         Snapdragon 8 Elite devices: {elite or '(none listed — target by chipset)'}")
    except Exception as e:
        ok = False
        print(f"  [FAIL] token present but AI Hub call failed -> {e}")

print("\n== RESULT ==")
print("  READY for conversion." if ok and CONFIG_INI.exists()
      else "  Install looks fine; finish the auth step, then re-run." if ok
      else "  Fix the FAIL items above.")
sys.exit(0 if ok else 1)
