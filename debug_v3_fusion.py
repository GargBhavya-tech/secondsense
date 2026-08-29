"""
Offline validation for V3 drop-off detection's core, data-free algorithms:
  1. Classical RGB edge lattice (Canny + Hough, no training) — finds parallel
     horizontal-ish stair-nosing lines.
  2. Depth evidence channel (reuses the V-disparity/RANSAC ground-plane fit from
     debug_vdisparity.py, but reports SUPPORTS/CONTRADICTS/UNRELIABLE instead of a veto).
  3. Rule-baseline fusion -> SAFE / POSSIBLE_DROP / DROP_CONFIRMED per the plan's output
     contract (single-frame only here; the temporal state machine is Kotlin-only, tested
     with synthetic frame sequences separately).

Regression fixtures (per the plan's release-gate section):
  stairs1.jpeg  -> ascending stairs, looking UP  -> expect NOT DROP_CONFIRMED
  stairs.jpg    -> descending stairs (V2 already catches) -> expect DROP_CONFIRMED or POSSIBLE
  stairs2.jpeg  -> descending stairs, depth sign WRONG (the hard case) -> expect
                   POSSIBLE_DROP or DROP_CONFIRMED via RGB lattice even though depth is weak/wrong
  bottle.jpeg   -> flat floor, no stairs -> expect SAFE

USAGE: venv\\Scripts\\python.exe debug_v3_fusion.py <photo>
"""
import sys
import numpy as np
import cv2
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

DEPTH_MODEL = "export_assets/tflite_models/depth_anything_v2/depth_anything_v2.tflite"


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


def run_depth(path, size=518):
    img = Image.open(path).convert("RGB")
    interp = Interpreter(model_path=DEPTH_MODEL)
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
    return normalize(d.astype(np.float32)), np.array(lb)


# ---------------------------------------------------------------------------
# 1b. DEPENDENCY-FREE row-profile lattice (no OpenCV — this is what actually ports to
#     Kotlin, since the Android app has zero CV-library dependency by design, matching
#     GroundPlaneAnalyzer's own pure-array approach). Horizontal-gradient row energy,
#     peak-picking, periodicity scoring — same spirit as the Hough version above, kept
#     as a separate function so both can be compared on the same fixtures.
# ---------------------------------------------------------------------------
def edge_lattice_portable(rgb):
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    x1, x2 = int(w * 0.15), int(w * 0.85)
    y1, y2 = int(h * 0.35), int(h * 1.0)
    corridor = gray[y1:y2, x1:x2]
    cw = corridor.shape[1]

    # Per-pixel vertical gradient (vs the row-average tried first, which couldn't tell
    # "a real horizontal ridge" from "scattered vertical texture in that row" — that's
    # why it missed stairs2.jpeg). Threshold to a binary edge map, THEN require genuine
    # horizontal CONTIGUITY per row (a run of consecutive edge pixels), not just count.
    grad = np.abs(corridor[2:] - corridor[:-2])  # shape (rows-2, cw)
    edge_bin = grad > (grad.mean() + 0.75 * grad.std())

    min_run = int(0.10 * cw)  # a real stair nosing spans a real fraction of the corridor width
    row_run_frac = np.zeros(edge_bin.shape[0])
    for i in range(edge_bin.shape[0]):
        row = edge_bin[i]
        # longest run of True values in this row, vectorized via diff-of-cumsum trick
        if not row.any():
            continue
        idx = np.flatnonzero(np.diff(np.r_[0, row.view(np.int8), 0]))
        runs = idx[1::2] - idx[0::2]
        row_run_frac[i] = runs.max() / cw if len(runs) else 0.0

    min_sep = max(1, int(0.02 * corridor.shape[0]))
    thresh = 0.10  # longest horizontal run must cover >=10% of corridor width
    peaks = []
    i = 0
    while i < len(row_run_frac):
        if row_run_frac[i] > thresh:
            window = row_run_frac[i:i + min_sep]
            local_max_idx = i + int(np.argmax(window))
            peaks.append(local_max_idx)
            i = local_max_idx + min_sep
        else:
            i += 1

    if len(peaks) < 3:
        return 0.0, peaks

    rows_norm = [(y1 + p) / h for p in peaks]
    spacings = np.diff(rows_norm)
    mean_spacing = float(np.mean(spacings))
    spacing_consistency = 1.0 - min(1.0, np.std(spacings) / max(mean_spacing, 1e-3))
    count_support = min(1.0, len(peaks) / 5.0)
    score = float(np.clip(0.75 * spacing_consistency + 0.25 * count_support, 0, 1))
    return score, peaks


# ---------------------------------------------------------------------------
# 1. Classical RGB edge lattice (no training data needed)
# ---------------------------------------------------------------------------
def edge_lattice(rgb):
    """Find near-horizontal parallel line segments (candidate stair nosings) inside
    a wide center corridor. Returns (score 0..1, list of (y, angle_deg, length))."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Local illumination normalization (CLAHE) to reduce broad shadow bands.
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(gray, 40, 120)

    # Wide center corridor (stand-in for the IMU-stabilized corridor at single-frame
    # level, since we don't have live IMU data on a static photo).
    x1, x2 = int(w * 0.15), int(w * 0.85)
    y1, y2 = int(h * 0.35), int(h * 1.0)
    mask = np.zeros_like(edges)
    mask[y1:y2, x1:x2] = 1
    edges = edges * mask

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                             minLineLength=int(w * 0.12), maxLineGap=15)
    if lines is None:
        return 0.0, []

    segs = []
    for l in lines[:, 0]:
        x1s, y1s, x2s, y2s = l
        dx, dy = x2s - x1s, y2s - y1s
        length = float(np.hypot(dx, dy))
        angle = float(np.degrees(np.arctan2(dy, dx)))
        # REVISED (matches EdgeLattice.kt's tightened thresholds after the real desk/keyboard
        # false positive): narrower angle band, and support requirement raised elsewhere.
        if abs(angle) < 12:
            segs.append(((y1s + y2s) / 2.0 / h, angle, length))

    if len(segs) < 3:
        return 0.0, segs

    # Cluster near-duplicate rows first (a real stair nosing produces several close,
    # near-identical Hough segments — naive counting double-rewards that, which is
    # exactly what let bottle.jpeg's random clutter and stairs1's real-but-irrelevant
    # lines through in the first offline validation pass).
    rows_sorted = sorted(s[0] for s in segs)
    clustered = [rows_sorted[0]]
    for r in rows_sorted[1:]:
        if r - clustered[-1] > 0.02:  # >2% of frame height apart -> a distinct nosing
            clustered.append(r)
        # else: merge into the same nosing, don't double-count

    if len(clustered) < 3:
        return 0.0, segs  # a real staircase shows several distinct nosings, not 1-2

    spacings = np.diff(clustered)
    mean_spacing = float(np.mean(spacings))
    # Periodicity is the real distinguishing signature of a stair lattice vs random
    # clutter (shelf edges, table lines) — weight it far more than raw segment count.
    spacing_consistency = 1.0 - min(1.0, np.std(spacings) / max(mean_spacing, 1e-3))
    count_support = min(1.0, len(clustered) / 5.0)
    score = float(np.clip(0.75 * spacing_consistency + 0.25 * count_support, 0, 1))
    return score, segs


# ---------------------------------------------------------------------------
# 2. Depth as an EVIDENCE channel (not a veto) — V-disparity/RANSAC ground-plane fit,
#    reused from debug_vdisparity.py, but reporting a 3-way verdict.
# ---------------------------------------------------------------------------
def ransac_line_fit(xs, ys, n_iters=200, inlier_thresh=0.04):
    best_inliers, best_count = None, -1
    n = len(xs)
    rng = np.random.default_rng(0)
    for _ in range(n_iters):
        i1, i2 = rng.choice(n, 2, replace=False)
        x1, x2 = xs[i1], xs[i2]
        if x1 == x2:
            continue
        a = (ys[i2] - ys[i1]) / (x2 - x1)
        b = ys[i1] - a * x1
        inliers = np.abs(a * xs + b - ys) <= inlier_thresh
        if inliers.sum() > best_count:
            best_count, best_inliers = inliers.sum(), inliers
    xi, yi = xs[best_inliers], ys[best_inliers]
    a, b = np.polyfit(xi, yi, 1)
    return a, b


def depth_evidence(depth):
    """Returns (verdict, deviation) where verdict in {SUPPORTS, CONTRADICTS, UNRELIABLE}."""
    h, w = depth.shape
    cx1, cx2 = int(w * 0.35), int(w * 0.65)
    fit_start, fit_end = int(h * 0.55), int(h * 0.85)
    rows = np.arange(fit_start, fit_end)
    profile = np.array([np.median(depth[y, cx1:cx2]) for y in rows])
    rows_norm = rows.astype(np.float64) / h

    if np.std(profile) < 1e-4:
        return "UNRELIABLE", 0.0  # locally flat/smoothed map -> can't trust it

    a, b = ransac_line_fit(rows_norm, profile)
    check_start = int(h * 0.90)
    worst_dev = 0.0
    col_signs = []
    for x0 in range(cx1, cx2, max(1, (cx2 - cx1) // 5)):
        col_devs = []
        for y in range(check_start, h):
            predicted = a * (y / h) + b
            actual = depth[y, x0]
            col_devs.append(predicted - actual)
        if col_devs:
            col_signs.append(np.sign(np.mean(col_devs)))
            worst = max(col_devs, key=abs)
            if abs(worst) > abs(worst_dev):
                worst_dev = worst

    agreement = abs(np.mean(col_signs)) if col_signs else 0.0  # 1.0 = all columns agree
    if abs(worst_dev) < 0.06:
        return "UNRELIABLE", worst_dev
    if agreement < 0.5:
        return "UNRELIABLE", worst_dev  # columns disagree -> not trustworthy either way
    return ("SUPPORTS" if worst_dev > 0 else "CONTRADICTS"), worst_dev


# ---------------------------------------------------------------------------
# 3. Rule-baseline fusion (per plan section 7's explicit rule baseline)
# ---------------------------------------------------------------------------
def fuse(lattice_score, depth_verdict):
    strong_lattice = lattice_score >= 0.55
    weak_lattice = lattice_score >= 0.3

    # WITHOUT the (deferred) semantic ascending/descending classifier, a strong lattice
    # alone cannot distinguish real descending stairs from ascending stairs or other
    # periodic clutter — that ambiguity is exactly what stairs1.jpeg's false positive
    # exposed in the first two offline validation passes. Depth SUPPORT is therefore
    # required, not just "not actively contradicting", before ever confirming.
    if strong_lattice and depth_verdict == "SUPPORTS":
        return "DROP_CONFIRMED"
    if strong_lattice and depth_verdict == "UNRELIABLE":
        return "POSSIBLE_DROP"  # geometry alone isn't enough to confirm without depth or a classifier
    if strong_lattice and depth_verdict == "CONTRADICTS":
        return "POSSIBLE_DROP"  # geometry says drop, depth disagrees -> be cautious, not silent
    if weak_lattice and depth_verdict == "SUPPORTS":
        return "POSSIBLE_DROP"
    return "SAFE"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "stairs2.jpeg"
    depth, lb_rgb = run_depth(path)
    lattice_score, segs = edge_lattice(lb_rgb)
    portable_score, peaks = edge_lattice_portable(lb_rgb)
    depth_verdict, dev = depth_evidence(depth)
    state = fuse(lattice_score, depth_verdict)
    state_portable = fuse(portable_score, depth_verdict)

    print(f"{path}")
    print(f"  edge lattice (Hough):    score={lattice_score:.2f}  segments={len(segs)}")
    print(f"  edge lattice (portable): score={portable_score:.2f}  peaks={len(peaks)}")
    print(f"  depth evidence: {depth_verdict} (deviation={dev:+.3f})")
    print(f"  FUSED STATE (Hough)   : {state}")
    print(f"  FUSED STATE (portable): {state_portable}")


if __name__ == "__main__":
    main()
