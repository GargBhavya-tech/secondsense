"""
Validate V-disparity + RANSAC ground-plane fitting for drop-off detection, against all four
real test photos used this session (including the known-failure case for the current V2
detector: stairs2.jpeg's depth model reads the wrong sign).

METHOD:
1. For each row y in the lower half of the depth map, take the MEDIAN proximity across the
   center columns -> a "V-disparity profile" prox(y). On a flat/sloped-but-consistent ground
   plane, this profile is close to LINEAR (nearer as you look further down the frame).
2. Fit a line to that profile with RANSAC (robust to a minority of rows being disrupted by
   furniture/stairs/etc in the profile itself).
3. Compare the ACTUAL profile against the FITTED line for the bottom-most rows (right in
   front of the feet): if actual proximity reads meaningfully FARTHER than the line predicts,
   that's the ground-plane signature of a drop-off — the floor isn't where the established
   ground-plane trend says it should be.

USAGE:
    venv\\Scripts\\python.exe debug_vdisparity.py <photo>
"""
import sys
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL = "export_assets/tflite_models/depth_anything_v2/depth_anything_v2.tflite"


def letterbox(img, size):
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    r = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(r, (px, py))
    return canvas


def normalize(d):
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)


def ransac_line_fit(xs, ys, n_iters=200, inlier_thresh=0.04):
    """Fit y = a*x + b via RANSAC. Returns (a, b, inlier_mask)."""
    best_inliers = None
    best_count = -1
    n = len(xs)
    rng = np.random.default_rng(0)
    for _ in range(n_iters):
        i1, i2 = rng.choice(n, 2, replace=False)
        x1, x2 = xs[i1], xs[i2]
        if x1 == x2:
            continue
        a = (ys[i2] - ys[i1]) / (x2 - x1)
        b = ys[i1] - a * x1
        pred = a * xs + b
        inliers = np.abs(pred - ys) <= inlier_thresh
        count = inliers.sum()
        if count > best_count:
            best_count = count
            best_inliers = inliers
    # refit on inliers via least squares for a cleaner final line
    xi, yi = xs[best_inliers], ys[best_inliers]
    a, b = np.polyfit(xi, yi, 1)
    return a, b, best_inliers


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "stairs2.jpeg"
    img = Image.open(path).convert("RGB")

    interp = Interpreter(model_path=MODEL)
    interp.allocate_tensors()
    ind = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    lb = letterbox(img, ind["shape"][1])
    arr = (np.asarray(lb).astype(np.float32) / 255.0)[None, ...]
    interp.set_tensor(ind["index"], arr)
    interp.invoke()
    d = interp.get_tensor(outd["index"])[0]
    if d.ndim == 3:
        d = d[..., 0] if d.shape[-1] == 1 else d[0]
    depth = normalize(d.astype(np.float32))
    h, w = depth.shape
    cx1, cx2 = int(w * 0.35), int(w * 0.65)

    # V-disparity profile, FIT window only (55%-85%) — deliberately does NOT include the
    # check zone (90%-100%) below, so the "trusted ground plane" reference can't be
    # contaminated by the very hazard we're trying to detect.
    fit_start, fit_end = int(h * 0.55), int(h * 0.85)
    rows = np.arange(fit_start, fit_end)
    profile = np.array([np.median(depth[y, cx1:cx2]) for y in rows])
    rows_norm = rows.astype(np.float64) / h

    a, b, inliers = ransac_line_fit(rows_norm, profile)
    print(f"{path}: fitted ground-plane line (rows {fit_start}-{fit_end}): "
          f"proximity = {a:.3f} * row_frac + {b:.3f}  "
          f"({inliers.sum()}/{len(rows)} inlier rows, {inliers.sum()/len(rows)*100:.0f}%)")

    # Check the bottom-most rows (right in front of the feet) against the fitted line
    check_start = int(h * 0.90)
    for y in range(check_start, h, max(1, (h - check_start) // 5)):
        actual = np.median(depth[y, cx1:cx2])
        predicted = a * (y / h) + b
        deviation = predicted - actual  # positive = actual is FARTHER than the ground trend predicts
        flag = " <-- DROP-OFF signature" if deviation > 0.08 else ""
        print(f"  row {y:3d} ({y/h*100:3.0f}%): actual={actual:.3f}  predicted={predicted:.3f}  "
              f"deviation={deviation:+.3f}{flag}")


if __name__ == "__main__":
    main()
