"""
Exact Python port of DropOffDetector.kt's V2 algorithm (Sobel-style gradient + local sign
check), so we can debug why a real photo does/doesn't fire without a phone rebuild cycle.

USAGE:
    venv\\Scripts\\python.exe debug_dropoff_v2.py stairs1.jpeg
"""
import sys
import numpy as np
import cv2
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = "export_assets/tflite_models/depth_anything_v2/depth_anything_v2.tflite"

CENTER_HALF_WIDTH = 0.16
EDGE_STRENGTH_THRESHOLD = 0.10
LOCAL_DROP_DELTA = 0.05


def letterbox(img: Image.Image, size: int):
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    r = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(r, (px, py))
    return canvas


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "stairs1.jpeg"
    img = Image.open(path).convert("RGB")

    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    ind = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    size = ind["shape"][1]

    lb = letterbox(img, size)
    arr = (np.asarray(lb).astype(np.float32) / 255.0)[None, ...]
    interp.set_tensor(ind["index"], arr)
    interp.invoke()
    depth_raw = interp.get_tensor(outd["index"])[0]
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0] if depth_raw.shape[-1] == 1 else depth_raw[0]
    depth_raw = depth_raw.astype(np.float32)
    h, w = depth_raw.shape

    # match Kotlin's frame.lo/frame.hi: 2nd/98th percentile of the RAW (unnormalized) map
    sample = depth_raw.flatten()[::4]
    lo = np.percentile(sample, 2)
    hi = np.percentile(sample, 98)
    rng = hi - lo
    print(f"depth shape={depth_raw.shape}  lo={lo:.3f} hi={hi:.3f} range={rng:.3f}")
    if rng <= 1e-6:
        print("INVALID depth frame (range too small) -> detect() returns null immediately")
        return

    x1 = int((0.5 - CENTER_HALF_WIDTH) * w)
    x2 = int((0.5 + CENTER_HALF_WIDTH) * w)
    y_start = max(1, h // 2)

    # TRUE Sobel Gy (3x3, matches cv2.Sobel — this is what was actually validated earlier,
    # NOT a plain row-1/row+1 difference, which under-weights by ~4x and misses real edges).
    depth_norm = np.clip((depth_raw - lo) / rng, 0, 1)
    sobel_y = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(sobel_y[:, x1:x2 + 1]).mean(axis=1)

    best_row = int(np.argmax(grad[y_start:]) + y_start)
    best_strength = float(grad[best_row])

    print(f"\nSTEP 1: best gradient row={best_row} ({best_row/h*100:.0f}% down)  strength={best_strength:.4f}")
    print(f"        threshold={EDGE_STRENGTH_THRESHOLD}  -> {'PASS' if best_strength >= EDGE_STRENGTH_THRESHOLD else 'FAIL (too weak/noisy, rejected here)'}")
    top5 = np.argsort(-grad[y_start:])[:5] + y_start
    print(f"        (top 5 candidate rows by strength, for context:)")
    for r in top5:
        print(f"          row {r:3d} ({r/h*100:.0f}%): strength={grad[r]:.4f}")

    if best_row < 0 or best_strength < EDGE_STRENGTH_THRESHOLD:
        print("\n-> detect() returns null at STEP 1")
        return

    band_px = max(6, h // 40)
    above_rows = depth_raw[max(0, best_row - band_px):best_row, x1:x2 + 1]
    below_rows = depth_raw[best_row:min(h, best_row + band_px), x1:x2 + 1]
    above_prox = ((above_rows - lo) / rng).mean()
    below_prox = ((below_rows - lo) / rng).mean()
    local_diff = above_prox - below_prox

    print(f"\nSTEP 2: aboveProx={above_prox:.3f}  belowProx={below_prox:.3f}  local_diff={local_diff:.3f}")
    print(f"        threshold={LOCAL_DROP_DELTA}  -> {'PASS -> FIRES' if local_diff >= LOCAL_DROP_DELTA else 'FAIL (wrong sign/too weak, rejected here)'}")


if __name__ == "__main__":
    main()
