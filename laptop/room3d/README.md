# Live 3D room reconstruction — laptop only

The phone is **just an RGB camera**. Nothing is added to the SecondSense app. `scrcpy` injects a
temporary server over `adb`, streams the **back camera** as H.264, and the laptop does
everything: monocular metric depth → visual odometry → TSDF fusion → a 3D room that fills in
live as you pan.

```
scrcpy --video-source=camera  ──►  PyAV H.264 decode  ──►  Depth-Anything-V2-Metric-Indoor
   (no app, no phone code)                                          │
                                                                    ▼
   Open3D window: room mesh grows + camera trail   ◄──  TSDF fusion  ◄──  RGBD odometry
```

No IMU, no metric scale from the phone, no loop closure — it drifts on fast motion. Fine for a
2–3 minute slow scan of one room, which is the demo.

## Setup (Windows 11)

1. **scrcpy ≥ 2.4** on `PATH` — <https://github.com/Genymobile/scrcpy/releases> (unzip, add the
   folder to PATH). Check: `scrcpy --version`.
2. **adb**: `adb devices` must list the phone as `device` (unlock it, accept the USB-debugging
   prompt). The SecondSense app does **not** need to be running.
3. **Python 3.10 or 3.11**, fresh venv:
   ```
   python -m venv .venv && .venv\Scripts\activate
   pip install -r laptop/room3d/requirements.txt
   ```
   GPU is optional — install a CUDA `torch` wheel for ~15 FPS; CPU-only runs at ~2 FPS.
4. First run downloads the depth model (~100 MB) from Hugging Face.

## Run

```
cd laptop
set PYTHONPATH=%CD%
..\venv\Scripts\python -u -m room3d.app --source phone:<PHONE-IP> --seconds 180
```

Default output is one **coloured point cloud of the room**, built the "photo mode" way and
stitched: Depth-Anything-V2-Small → normalise → remap to 0.5–4 m → stride back-projection →
per-frame **scale-lock** (keeps every frame's cloud the same size) → **ICP** onto the running
cloud → merge. `S` saves `room-<ts>.ply`, `R` resets, `Q` quits.

Two-step for a clean full sweep (recommended — decouples capture from CPU speed):
```
..\venv\Scripts\python -m room3d.record --source phone:<PHONE-IP> --out sweep.mp4 --seconds 120
..\venv\Scripts\python -m room3d.app    --source file:sweep.mp4
```

`--tsdf` switches to metric depth + RGBD odometry + a fused **mesh** (better geometry, much
heavier — wants a GPU). `--source camera` uses scrcpy (blocked on some vivo/oppo ROMs).

An **Open3D window** (the 3D room) and a small **OpenCV window** (RGB + depth + status) open.
Focus the Open3D window for the keys:

| key | action |
|-----|--------|
| `S` | save `room_mesh-<ts>.ply` + `room-<ts>.ply` |
| `R` | wipe the map and restart |
| `Q` | quit (also auto-saves) |

**How to scan:** hold the phone steady, then do a **slow translating pan** — drift sideways or
walk a little as you turn. A pure pivot-in-place has no baseline and the odometry will freeze
(the status shows `TRACKING LOST`); just add some sideways motion and it recovers.

### Useful flags

| flag | default | notes |
|------|---------|-------|
| `--gpu` | off | frame-to-model SLAM via `o3d.t.pipelines.slam` — needs a **CUDA-enabled Open3D** wheel; much less drift |
| `--source file:scan.mkv` | — | replay a recording (`scrcpy --video-source=camera --record=scan.mkv`) — dev/offline, no phone |
| `--focal <px>` | `0.9·width` | real focal length if you know it; removes cylindrical shear |
| `--fov <deg>` | — | alternative to `--focal` |
| `--depth-max <m>` | `3.0` | cull distant low-confidence depth (stops "hallway" Z-blowup) |
| `--voxel <m>` | `0.02` | smaller = finer + heavier |
| `--max-width <px>` | `640` | working resolution; drop to `480` on a weak GPU/CPU |
| `--blur-min <v>` | `35` | Laplacian-variance floor; frames below it are skipped ("coast") |
| `--no-preview` | — | hide the OpenCV window |

## Troubleshooting

| symptom | fix |
|---|---|
| `scrcpy never connected` | phone locked / USB-debugging prompt not accepted; `adb kill-server && adb devices` |
| `--video-source=camera` unsupported | scrcpy too old — need ≥ 2.4 |
| room funnels / stretches away | wrong focal — pass `--focal` or `--fov`; lower `--depth-max` |
| walls duplicated / ghosted | pan slower; the depth EMA + median-lock need gentle motion |
| freezes when you stop moving | expected (no baseline) — add slight translation |
| CUDA illegal memory access with `--gpu` | raise `--max-blocks`, or the block guard already froze integration (`MAP FULL`) |
| ~2 FPS | CPU-only depth — install CUDA `torch`, or `--max-width 480` |

## Not done (v2 ideas)

- One-time **Depth Pro** pass on frame 0 for a real focal length (research §Intrinsics).
- **TensorRT** engine for the depth model (sub-15 ms).
- Loop closure / pose-graph for larger-than-one-room scans.
- Reuse the phone's own ARCore `RoomScanActivity` output instead of laptop odometry (more
  accurate, but needs a small phone change — deliberately avoided here).
