"""
Monocular metric depth for the 3D room demo.

Model: ``depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`` — the metric indoor
fine-tune of DAv2-Small. It outputs depth in METRES directly, so there is no affine scale/shift
to recover per frame (the research's main argument against the base affine-invariant model).
ViT-S backbone, ~25 M params: ~13 ms on an RTX GPU, ~0.4 s CPU.

Two stabilisers the research calls for, both here:
  * TemporalDepthFilter  — photometric-gated EMA. Blends depth toward the previous frame ONLY
    where the greyscale image barely changed (static geometry); moving/new pixels pass
    through untouched, so there is no "comet tail" ghosting.
  * blur_score           — Laplacian variance. app.py skips odometry+integration on frames
    below a threshold so a fast, blurry pan makes the map "coast" instead of diverging.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_DEF_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
_REL_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian — low == blurry / low-texture."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class MetricDepth:
    def __init__(self, model_id: str = _DEF_MODEL, device: Optional[str] = None, size: int = 518):
        import torch
        from transformers import AutoModelForDepthEstimation

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size  # DINOv2 patch 14 -> keep divisible by 14
        self._model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()
        self._half = self.device == "cuda"
        if self._half:
            self._model = self._model.half()

    @property
    def is_gpu(self) -> bool:
        return self.device == "cuda"

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 (H,W,3) -> metric depth float32 (H,W), in metres."""
        torch = self._torch
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        if self._half:
            x = x.half()
        with torch.no_grad():
            pred = self._model(pixel_values=x).predicted_depth  # (1, s, s), metres
        depth = pred.squeeze().float().cpu().numpy()
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


class RelDepth:
    """
    Depth-Anything-V2-**Small** (relative inverse depth) via the transformers pipeline — the
    "photo mode" model. Per frame: normalise to 0..1, remap onto a guessed [dmin, dmax] metre
    range. NOT metric. [infer_metricish] also applies an optional per-frame scale factor so a
    caller can lock this frame's central-median depth to a running reference (keeps the
    accumulating point cloud from drifting in scale — the fix for "overlapping different
    things onto each other").
    """

    def __init__(self, device: Optional[str] = None, model: str = _REL_MODEL):
        import torch
        from transformers import pipeline
        dev = 0 if (device == "cuda" or (device is None and torch.cuda.is_available())) else -1
        self.device = "cuda" if dev == 0 else "cpu"
        self._pipe = pipeline("depth-estimation", model=model, device=dev)

    def infer_metricish(self, bgr: np.ndarray, dmin: float, dmax: float,
                        scale: float = 1.0) -> np.ndarray:
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = self._pipe(Image.fromarray(rgb))["predicted_depth"]
        inv = out.squeeze().float().cpu().numpy() if hasattr(out, "cpu") else np.asarray(out, np.float32)
        inv = cv2.resize(inv, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        lo, hi = float(inv.min()), float(inv.max())
        inv = (inv - lo) / (hi - lo + 1e-6)                       # 0..1, bigger = closer
        d = (dmin + (1.0 - inv) * (dmax - dmin)).astype(np.float32)   # guessed metres
        return d * float(scale)


def backproject(bgr: np.ndarray, depth_m: np.ndarray, stride: int, fx_scale: float,
                cam_h: float, dmin: float, dmax: float):
    """Stride-sampled coloured point cloud from a depth map. Generic intrinsics fx=fy=fx_scale*W,
    camera assumed ~cam_h metres up (matches the original project's back-projection)."""
    import open3d as o3d
    h, w = depth_m.shape
    fx = fy = fx_scale * w
    cx, cy = w / 2.0, h / 2.0
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    d = depth_m[ys, xs]
    m = (d > dmin) & (d < dmax)
    d, xs, ys = d[m], xs[m], ys[m]
    pts = np.stack([(xs - cx) * d / fx,
                    -(ys - cy) * d / fy + cam_h,
                    d], axis=1).astype(np.float64)
    col = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[ys, xs].astype(np.float64) / 255.0
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(col)
    return pc


def central_median_depth(depth_m: np.ndarray) -> float:
    hh, ww = depth_m.shape
    c = depth_m[hh // 4:3 * hh // 4, ww // 4:3 * ww // 4]
    c = c[c > 0]
    return float(np.median(c)) if c.size else 0.0


class TemporalDepthFilter:
    """Photometric-gated EMA on the depth map + a gentle global scale lock."""

    def __init__(self, alpha: float = 0.6, photo_thresh: int = 12, median_lock: bool = True):
        self.alpha = alpha          # weight of the PREVIOUS depth where a pixel is static
        self.photo_thresh = photo_thresh
        self.median_lock = median_lock
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_depth: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_depth = None

    def apply(self, gray: np.ndarray, depth: np.ndarray) -> np.ndarray:
        if self._prev_depth is None or self._prev_gray is None:
            self._prev_gray, self._prev_depth = gray.copy(), depth.copy()
            return depth

        if self.median_lock:
            hh, ww = depth.shape
            c = (slice(hh // 4, 3 * hh // 4), slice(ww // 4, 3 * ww // 4))
            m_now = float(np.median(depth[c]))
            m_prev = float(np.median(self._prev_depth[c]))
            if m_now > 1e-3 and 0.5 < (m_prev / m_now) < 2.0:
                depth = depth * (m_prev / m_now)

        diff = cv2.absdiff(gray, self._prev_gray)
        static = (diff < self.photo_thresh).astype(np.float32)
        static = cv2.GaussianBlur(static, (0, 0), 2.0)
        blended = static * (self.alpha * self._prev_depth + (1.0 - self.alpha) * depth) + \
            (1.0 - static) * depth

        self._prev_gray, self._prev_depth = gray.copy(), blended.astype(np.float32)
        return self._prev_depth
