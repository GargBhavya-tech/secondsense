#!/usr/bin/env python3
"""
SecondSense — model conversion driver (Green Light, laptop).

Wraps `python -m qai_hub_models.models.<module>.export` for each target model,
targeting the phone's chipset, and writes a manifest of what converted.

Command shape verified against qai-hub 0.55.0 / qai-hub-models 0.61.0:
    python -m qai_hub_models.models.<module>.export \
        --device "<DEVICE>" --target-runtime qnn_context_binary \
        --output-dir export_assets/<module>

USAGE
    # See exactly what would run, without a Qualcomm account (safe anywhere):
    python convert.py --dry-run

    # Real run (needs `qai-hub configure` done first + network to Qualcomm):
    python convert.py --device "Snapdragon 8 Elite QRD"          # from `qai-hub list-devices`
    python convert.py --device "..." --group core                # just the R1/R2 spine
    python convert.py --device "..." --only yolov11_det          # one model
    python convert.py --device "..." --tflite-fallback           # retry tflite if QNN fails

The device string comes from `python scripts/list_devices.py` (or `qai-hub list-devices`).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_JSON = ROOT / "models.json"
ASSETS = ROOT / "export_assets"
LOGS = ROOT / "logs"
MANIFEST = ROOT / "conversion_manifest.json"
CONFIG_INI = Path.home() / ".qai_hub" / "client.ini"


def load_models():
    data = json.loads(MODELS_JSON.read_text())
    return data["models"]


def select(models, args):
    if args.only:
        wanted = set(args.only)
        chosen = [m for m in models if m["module"] in wanted]
        missing = wanted - {m["module"] for m in chosen}
        if missing:
            sys.exit(f"[convert] unknown model(s) in --only: {sorted(missing)}")
    elif args.group == "all":
        chosen = list(models)
    elif args.group == "core":
        # hello-world first, then the R1/R2 spine
        chosen = [m for m in models if m["group"] in ("hello", "core")]
    else:
        chosen = [m for m in models if m["group"] == args.group]
    # always put hello-world first so a failure isolates to the toolchain
    chosen.sort(key=lambda m: 0 if m["group"] == "hello" else 1)
    return chosen


def build_cmd(module, device, runtime, quantize, outdir, ckpt_name=None):
    cmd = [sys.executable, "-m", f"qai_hub_models.models.{module}.export",
           "--device", device,
           "--target-runtime", runtime,
           "--output-dir", str(outdir)]
    if quantize:
        cmd += ["--quantize", quantize]
    if ckpt_name:
        cmd += ["--ckpt-name", ckpt_name]
    return cmd


def run_one(m, args):
    module = m["module"]
    log = LOGS / f"{module}.log"
    quantize = args.quantize or m.get("quantize")

    def get_outdir(runtime):
        subfolder = "tflite_models" if runtime == "tflite" else "qnn_binaries"
        d = ASSETS / subfolder / module
        d.mkdir(parents=True, exist_ok=True)
        return d

    def attempt(runtime):
        outdir = get_outdir(runtime)
        cmd = build_cmd(module, args.device, runtime, quantize, outdir, m.get("ckpt_name"))
        printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        print(f"\n[{module}] runtime={runtime}")
        print(f"  $ {printable}")
        if args.dry_run:
            return "dry-run", printable
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
        return ("ok" if proc.returncode == 0 else "failed"), printable

    status, cmdline = attempt(args.runtime)
    used_runtime = args.runtime
    if status == "failed" and args.tflite_fallback and args.runtime != "tflite":
        print(f"  [{module}] QNN failed -> retrying with tflite (still on-NPU via delegate)")
        status, cmdline = attempt("tflite")
        used_runtime = "tflite"

    actual_outdir = get_outdir(used_runtime)
    return {
        "module": module,
        "role": m.get("role", ""),
        "status": status,
        "runtime": used_runtime,
        "quantize": quantize,
        "assets": str(actual_outdir.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)) if not args.dry_run else None,
        "cmd": cmdline,
    }


def main():
    ap = argparse.ArgumentParser(description="Convert SecondSense target models via Qualcomm AI Hub.")
    ap.add_argument("--device", default=os.environ.get("SECONDSENSE_DEVICE", ""),
                    help='Target device string from `qai-hub list-devices` (e.g. "Snapdragon 8 Elite QRD"). Or set SECONDSENSE_DEVICE.')
    ap.add_argument("--runtime", default="qnn_context_binary",
                    choices=["qnn_context_binary", "qnn_dlc", "tflite", "onnx", "precompiled_qnn_onnx"],
                    help="On-device runtime. qnn_context_binary = NPU-native (default).")
    ap.add_argument("--group", default="core", choices=["hello", "core", "optional", "all"],
                    help="Which set to convert. core = hello-world + R1/R2 spine (default).")
    ap.add_argument("--only", nargs="+", help="Explicit module name(s), overrides --group.")
    ap.add_argument("--quantize", default=None,
                    help="Global quantize flag passthrough (per-model; check each model's export --help first).")
    ap.add_argument("--tflite-fallback", action="store_true",
                    help="If a QNN compile fails, retry that model with --target-runtime tflite.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the exact commands without running (no account needed).")
    args = ap.parse_args()

    models = select(load_models(), args)

    # Guardrails for real runs
    if not args.dry_run:
        if not args.device:
            sys.exit("[convert] --device is required for a real run. Run `python scripts/list_devices.py` "
                     "to find your Snapdragon 8 Elite target, or use --dry-run.")
        if not CONFIG_INI.exists():
            sys.exit(f"[convert] {CONFIG_INI} not found — run `qai-hub configure --api_token <TOKEN>` first "
                     "(get the token at https://aihub.qualcomm.com/). See docs/ticket03_qai_hub_toolchain_runbook.md.")
    else:
        if not args.device:
            args.device = "<DEVICE-FROM-list-devices>"

    print(f"[convert] models: {[m['module'] for m in models]}")
    print(f"[convert] device={args.device!r}  runtime={args.runtime}  dry_run={args.dry_run}")

    results = [run_one(m, args) for m in models]

    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "runtime": args.runtime,
        "dry_run": args.dry_run,
        "results": results,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['status']:>8}  {r['module']:<20} -> {r['assets']}")
    print(f"\nmanifest: {MANIFEST.relative_to(ROOT)}")
    if not args.dry_run:
        failed = [r["module"] for r in results if r["status"] == "failed"]
        if failed:
            print(f"FAILED: {failed} — check logs/<module>.log")
            sys.exit(1)


if __name__ == "__main__":
    main()
