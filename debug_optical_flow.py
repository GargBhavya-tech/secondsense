"""
Validate a from-scratch single-scale Lucas-Kanade sparse optical flow implementation
against a KNOWN synthetic translation, before porting the identical math to Kotlin.

Method: take a real photo, shift it by a known (dx, dy) in pixels to simulate camera pan
(no real second frame needed — the true answer is known exactly), run LK tracking on a grid
of points, and check the estimated flow matches the known shift.

USAGE:
    venv\\Scripts\\python.exe debug_optical_flow.py
"""
import numpy as np
from PIL import Image

WINDOW_RADIUS = 7
NUM_ITERS = 5


def to_gray(img: Image.Image, target_w=160, target_h=120):
    small = img.convert("L").resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(small).astype(np.float32)


def bilinear_sample(img, x, y):
    h, w = img.shape
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return None
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
            img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def track_point(prev_gray, cur_gray, x0, y0, window=WINDOW_RADIUS, iters=NUM_ITERS):
    h, w = prev_gray.shape
    # precompute spatial gradients of prev_gray once (Sobel-ish central difference)
    gx = np.zeros_like(prev_gray); gy = np.zeros_like(prev_gray)
    gx[:, 1:-1] = (prev_gray[:, 2:] - prev_gray[:, :-2]) / 2.0
    gy[1:-1, :] = (prev_gray[2:, :] - prev_gray[:-2, :]) / 2.0

    u, v = 0.0, 0.0  # accumulated flow estimate
    xi, yi = int(round(x0)), int(round(y0))
    x1, x2 = max(0, xi - window), min(w, xi + window + 1)
    y1, y2 = max(0, yi - window), min(h, yi + window + 1)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None

    Ix = gx[y1:y2, x1:x2].flatten()
    Iy = gy[y1:y2, x1:x2].flatten()
    A = np.stack([Ix, Iy], axis=1)  # [N,2]
    AtA = A.T @ A
    if np.linalg.cond(AtA) > 1e6:  # ill-conditioned window (flat/no texture) -> unreliable
        return None
    AtA_inv = np.linalg.inv(AtA)

    for _ in range(iters):
        # sample the warped current-frame window at (x+u, y+v)
        warped = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        ok = True
        for r, yy in enumerate(range(y1, y2)):
            for c, xx in enumerate(range(x1, x2)):
                s = bilinear_sample(cur_gray, xx + u, yy + v)
                if s is None:
                    ok = False
                    break
                warped[r, c] = s
            if not ok:
                break
        if not ok:
            break
        It = (warped - prev_gray[y1:y2, x1:x2]).flatten()
        b = -(A.T @ It)
        duv = AtA_inv @ b
        u += duv[0]; v += duv[1]
        if abs(duv[0]) < 0.01 and abs(duv[1]) < 0.01:
            break
    return (u, v)


def main():
    img = Image.open("bottle1.jpeg")
    gray = to_gray(img)
    h, w = gray.shape

    known_dx, known_dy = 4.0, -2.5
    # simulate a shifted "next frame" via subpixel affine shift
    shifted_img = img.convert("L").resize((160, 120), Image.BILINEAR)
    shifted_img = shifted_img.transform(
        (160, 120), Image.AFFINE, (1, 0, -known_dx, 0, 1, -known_dy), resample=Image.BILINEAR
    )
    shifted_gray = np.asarray(shifted_img).astype(np.float32)

    # track a grid of points, skip near-edge points
    grid_pts = [(x, y) for x in range(30, w - 30, 25) for y in range(30, h - 30, 25)]
    estimates = []
    for (x0, y0) in grid_pts:
        flow = track_point(gray, shifted_gray, x0, y0)
        if flow is not None:
            estimates.append(flow)

    estimates = np.array(estimates)
    print(f"tracked {len(estimates)}/{len(grid_pts)} points")
    print(f"known shift:     dx={known_dx:.2f}  dy={known_dy:.2f}")
    print(f"median estimate: dx={np.median(estimates[:,0]):.2f}  dy={np.median(estimates[:,1]):.2f}")
    print(f"mean estimate:   dx={estimates[:,0].mean():.2f}  dy={estimates[:,1].mean():.2f}")
    print(f"std dev:         dx={estimates[:,0].std():.2f}  dy={estimates[:,1].std():.2f}")


if __name__ == "__main__":
    main()
