"""
Record a room sweep from the phone's /frame.jpg feed to an .mp4, so app.py can reconstruct it
offline at full quality (no live-framerate pressure). Shows a live preview + elapsed time so
you can see your coverage while you sweep.

    python -m room3d.record --source phone:10.156.105.9 --out sweep.mp4 --seconds 120
    python -m room3d.app   --source file:sweep.mp4 --max-width 560 --voxel 0.02

Sweep tips: slow, steady, keep translating (small sideways drift as you turn). Overlap heavily
— each new view should still contain ~half of the previous one. Do the walls, then tilt down
for the floor, up for the ceiling. Don't rush; 90-150 s for one room.
"""

from __future__ import annotations

import argparse
import time

import cv2

from .ingest import open_source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="camera", help="phone:<ip> | http://... | camera")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--out", default="sweep.mp4")
    ap.add_argument("--fps", type=float, default=12.0, help="output file fps (playback speed only)")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--max-width", type=int, default=960)
    args = ap.parse_args()

    src = open_source(args.source, adb_serial=args.serial)
    print(f"[record] {args.source} -> {args.out}   ({args.seconds:.0f}s, ESC to stop early)", flush=True)

    f0 = None
    t0 = time.time()
    while f0 is None and time.time() - t0 < 20:
        f0 = src.read()
        time.sleep(0.05)
    if f0 is None:
        src.close()
        raise RuntimeError("no frames — is the phone app running and on the same Wi-Fi?")

    scale = min(1.0, args.max_width / f0.shape[1])
    w = int(round(f0.shape[1] * scale)); h = int(round(f0.shape[0] * scale))
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    t0 = time.time()
    last = None
    n = 0
    period = 1.0 / max(1.0, args.fps)
    t_next = t0
    try:
        while time.time() - t0 < args.seconds:
            raw = src.read()
            now = time.time()
            if raw is not None:
                last = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
            if last is None:
                time.sleep(0.02); continue
            if now >= t_next:
                vw.write(last)
                n += 1
                t_next += period
            disp = last.copy()
            blur = cv2.Laplacian(cv2.cvtColor(last, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            cv2.putText(disp, f"REC {now - t0:5.1f}s / {args.seconds:.0f}   frames {n}   blur {blur:5.0f}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("record sweep — ESC to stop", disp)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        vw.release()
        src.close()
        cv2.destroyAllWindows()
        print(f"[record] wrote {n} frames -> {args.out}", flush=True)
        print(f"[record] now:  python -m room3d.app --source file:{args.out} --max-width 560 --voxel 0.02", flush=True)


if __name__ == "__main__":
    main()
