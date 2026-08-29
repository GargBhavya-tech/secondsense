"""
Offline prototype: does Sobel on the depth map localize edges better than the app's
current band-average DropOffDetector? No phone, no rebuild — same workflow as
debug_yolo.py, just pointed at depth_anything_v2.tflite instead.

USAGE:
    venv\\Scripts\\python.exe debug_sobel_dropoff.py bottle1.jpeg
"""
import sys
import numpy as np
import cv2
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = "export_assets/tflite_models/depth_anything_v2/depth_anything_v2.tflite"


def letterbox(img: Image.Image, size: int):
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    r = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(r, (px, py))
    return canvas


def normalize(depth: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
    return np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bottle1.jpeg"
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
    depth = normalize(depth_raw.astype(np.float32))
    print(f"depth map shape: {depth.shape}  range after normalize: {depth.min():.2f}..{depth.max():.2f}")

    h, w = depth.shape
    center_x1, center_x2 = int(w * 0.35), int(w * 0.65)
    col_band = depth[:, center_x1:center_x2].mean(axis=1)  # avg proximity per row, center strip

    # --- METHOD 1: current app logic — compare two fixed bands ---
    near_top, near_bot = int(h * 0.80), int(h * 0.99)
    ref_top, ref_bot = int(h * 0.55), int(h * 0.72)
    near_prox = col_band[near_top:near_bot].mean()
    ref_prox = col_band[ref_top:ref_bot].mean()
    band_diff = ref_prox - near_prox
    print(f"\n--- METHOD 1: band-average (current app logic) ---")
    print(f"near band [{near_top}:{near_bot}] avg proximity = {near_prox:.3f}")
    print(f"ref  band [{ref_top}:{ref_bot}] avg proximity = {ref_prox:.3f}")
    print(f"diff = {band_diff:.3f}  (fires if >= 0.20)  -> {'FIRES' if band_diff >= 0.20 else 'no fire'}")
    print("(tells you SOMETHING changed in the lower frame — not WHERE, not how far)")

    # --- METHOD 2: Sobel vertical gradient on the depth map, localized ---
    sobel_y = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_strength = np.abs(sobel_y[:, center_x1:center_x2]).mean(axis=1)
    # only look in the lower half of frame (floor region) to avoid picking up
    # gradients from objects/furniture higher up
    lower_half = grad_strength[h // 2:]
    peak_row = np.argmax(lower_half) + h // 2
    peak_strength = grad_strength[peak_row]
    print(f"\n--- METHOD 2: Sobel vertical gradient (proposed) ---")
    print(f"sharpest depth discontinuity in lower half: row {peak_row}/{h}  "
          f"({peak_row / h * 100:.0f}% down the frame)  strength={peak_strength:.3f}")
    print("(tells you the EXACT row where the floor/edge discontinuity is)")
    # show the gradient profile around the peak for eyeballing sharpness vs noise
    lo, hi = max(0, peak_row - 5), min(h, peak_row + 6)
    print("gradient magnitude around the peak (row: value):")
    for r in range(lo, hi):
        marker = "  <-- PEAK" if r == peak_row else ""
        print(f"  row {r:3d}: {grad_strength[r]:.3f}{marker}")


if __name__ == "__main__":
    main()
