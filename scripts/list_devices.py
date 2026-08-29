#!/usr/bin/env python3
"""
SecondSense — find the right target device for the iQOO 15 (Snapdragon 8 Elite Gen 5).

Needs `qai-hub configure` done first. Prints 8-Elite devices; you copy the exact
name into `convert.py --device "..."`. If none are listed, target by chipset —
the compiled QNN binary runs on any phone with that chip, incl. the iQOO 15.

    python scripts/list_devices.py
"""
import sys
try:
    import qai_hub
except Exception as e:
    sys.exit(f"qai_hub not installed: {e}")

try:
    devices = qai_hub.get_devices()
except Exception as e:
    sys.exit("Could not reach AI Hub. Run `qai-hub configure --api_token <TOKEN>` first.\n"
             f"  ({e})")

print(f"Total devices in farm: {len(devices)}\n")

elite = [d for d in devices if "8 Elite" in d.name]
print("== Snapdragon 8 Elite candidates (prefer newest = closest to iQOO 15) ==")
if elite:
    for d in elite:
        attrs = ",".join(getattr(d, "attributes", []) or [])
        print(f'  --device "{d.name}"   os={getattr(d,"os","?")}   {attrs}')
else:
    print("  (none named '8 Elite' — list all and target by chipset attribute instead)")

print("\nTip: reuse ONE device string for every model so profiling numbers are comparable.")
print('Then: python convert.py --device "<the name above>"')
