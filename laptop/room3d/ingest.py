"""
Frame ingestion for the laptop-side 3D room demo. ZERO code runs on the phone — scrcpy's
temporary server is injected over adb, streams the *camera* (not the screen) as H.264, and we
decode it in-process with PyAV. Window-scraping and stdout-pipe approaches were both rejected
by the research (fragile on Windows); this spawns scrcpy with a TCP video tunnel we own.

Sources (``--source``):
  camera            scrcpy --video-source=camera  (the real demo path; needs scrcpy >= 2.4 on PATH)
  file:<path>       decode a recorded .mkv/.mp4 — offline testing, no phone
  window:<title>    grab a visible window with mss — last-resort dev fallback

Every source yields the latest decoded BGR uint8 frame from ``read()`` (drops backlog — a stale
frame is worse than a skipped one), or ``None`` when the stream has ended.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Optional

import numpy as np


class FrameSource:
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------------------------
# scrcpy camera -> local TCP -> PyAV
# --------------------------------------------------------------------------------------------
class ScrcpyCameraSource(FrameSource):
    """
    scrcpy can only *record to a file* on Windows (no v4l2, no stdout stream, and the
    py-scrcpy-client socket libs don't build on Python 3.12). So: launch
    ``scrcpy --video-source=camera --record=<tmp.mkv>`` and TAIL that growing mkv with PyAV —
    periodically re-open, seek near the end, keep the last decoded frame. Latency ~1-2 s; fine
    for a slow room pan. For a rock-solid demo use two phases instead: record a clip, then run
    with ``--source file:<clip>``.
    """

    def __init__(self, adb_serial: Optional[str] = None, camera_id: int = 0,
                 max_size: int = 1280, fps: int = 30, poll: float = 0.4):
        if shutil.which("scrcpy") is None:
            raise RuntimeError("scrcpy not found on PATH. Install scrcpy >= 2.4 (https://github.com/Genymobile/scrcpy).")
        try:
            import av  # noqa: F401
        except ImportError as e:
            raise RuntimeError("PyAV missing — pip install av") from e

        import tempfile
        self._path = tempfile.NamedTemporaryFile(suffix=".mkv", delete=False).name
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._poll = poll

        args = ["scrcpy",
                "--video-source=camera",
                f"--camera-id={camera_id}",
                f"--max-size={max_size}",
                f"--max-fps={fps}",
                "--video-codec=h264",
                "--no-audio", "--no-control", "--no-playback", "--no-window",
                f"--record={self._path}", "--record-format=mkv"]
        if adb_serial:
            args += ["--serial", adb_serial]
        self._proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        self._thread = threading.Thread(target=self._run, name="scrcpy-tail", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import av
        # wait for scrcpy to create + prime the file
        for _ in range(150):
            if self._stop.is_set():
                return
            try:
                if os.path.getsize(self._path) > 200_000:
                    break
            except OSError:
                pass
            time.sleep(0.1)
        while not self._stop.is_set():
            try:
                c = av.open(self._path, mode="r")
                vs = c.streams.video[0]
                if vs.duration and vs.time_base:
                    dur = float(vs.duration * vs.time_base)
                    if dur > 1.5:
                        c.seek(int((dur - 1.2) / vs.time_base), stream=vs, any_frame=False)
                last = None
                for frame in c.decode(vs):
                    if self._stop.is_set():
                        break
                    last = frame
                c.close()
                if last is not None:
                    img = last.to_ndarray(format="bgr24")
                    with self._lock:
                        self._latest = img
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self._poll)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def close(self) -> None:
        self._stop.set()
        try:
            self._proc.terminate()
        except Exception:
            pass


# --------------------------------------------------------------------------------------------
# recorded file (offline testing)
# --------------------------------------------------------------------------------------------
class FileSource(FrameSource):
    def __init__(self, path: str, loop: bool = False, realtime: bool = True):
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self._loop = loop
        self._realtime = realtime
        self._period = 1.0 / max(1.0, self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._last = 0.0

    def read(self) -> Optional[np.ndarray]:
        if self._realtime:
            dt = time.time() - self._last
            if dt < self._period:
                time.sleep(self._period - dt)
        ok, frame = self._cap.read()
        if not ok:
            if self._loop:
                self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return None
        self._last = time.time()
        return frame

    def close(self) -> None:
        self._cap.release()


# --------------------------------------------------------------------------------------------
# visible window grab (dev fallback only — research flags this as fragile)
# --------------------------------------------------------------------------------------------
class WindowSource(FrameSource):
    def __init__(self, title_substr: str):
        import cv2
        import mss
        self._cv2 = cv2
        self._sct = mss.mss()
        self._region = self._find_window(title_substr)

    def _find_window(self, sub: str):
        try:
            import pygetwindow as gw
        except ImportError:
            raise RuntimeError("window: source needs `pip install pygetwindow`")
        wins = [w for w in gw.getAllWindows() if sub.lower() in (w.title or "").lower() and w.width > 0]
        if not wins:
            raise RuntimeError(f'no visible window matching "{sub}"')
        w = wins[0]
        return {"left": w.left, "top": w.top, "width": w.width, "height": w.height}

    def read(self) -> Optional[np.ndarray]:
        raw = np.asarray(self._sct.grab(self._region))  # BGRA
        return self._cv2.cvtColor(raw, self._cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._sct.close()


class HttpFrameSource(FrameSource):
    """
    Polls ``http://<phone-ip>:8085/frame.jpg`` published by the SecondSense DashboardServer
    (`publishFrame`). The chosen path when scrcpy camera capture is blocked by the ROM. Phone
    + laptop must be on the same Wi-Fi; the app must be running (any screen).
    """

    def __init__(self, target: str, poll: float = 0.06):
        import cv2
        from urllib.request import urlopen
        self._cv2 = cv2
        self._urlopen = urlopen
        if target.startswith("http"):
            self._url = target if target.endswith(".jpg") else target.rstrip("/") + "/frame.jpg"
        else:  # bare ip or ip:port
            hostport = target if ":" in target else f"{target}:8085"
            self._url = f"http://{hostport}/frame.jpg"
        self._poll = poll
        self._latest: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="http-frame", daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._urlopen(self._url, timeout=2.0) as r:
                    if getattr(r, "status", 200) == 200:
                        buf = np.frombuffer(r.read(), np.uint8)
                        img = self._cv2.imdecode(buf, self._cv2.IMREAD_COLOR)
                        if img is not None:
                            self._latest = img
            except Exception:  # noqa: BLE001
                time.sleep(0.3)
            time.sleep(self._poll)

    def read(self) -> Optional[np.ndarray]:
        return None if self._latest is None else self._latest.copy()

    def close(self) -> None:
        self._stop.set()


def open_source(spec: str, adb_serial: Optional[str] = None, **kw) -> FrameSource:
    if spec == "camera":
        return ScrcpyCameraSource(adb_serial=adb_serial, **kw)
    if spec.startswith(("http://", "https://")):
        return HttpFrameSource(spec)
    if spec.startswith("phone:"):
        return HttpFrameSource(spec[6:])
    if spec.startswith("file:"):
        # a recording is processed as fast as the pipeline can chew it, not at capture fps
        return FileSource(spec[5:], loop=kw.get("loop", False), realtime=kw.get("realtime", False))
    if spec.startswith("window:"):
        return WindowSource(spec[7:])
    raise ValueError(f"unknown --source '{spec}' (camera | phone:<ip> | http://... | file:<path> | window:<title>)")
