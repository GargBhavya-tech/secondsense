"""
SecondSense — laptop-side live 3D room reconstruction (demo only; nothing runs on the phone).

    python -m room3d.app --source phone:10.156.105.9 --seconds 180
    python -m room3d.app --source file:sweep.mp4
    python -m room3d.app --source file:sweep.mp4 --tsdf        # heavier metric-depth + TSDF mesh

Default output = the "photo mode" look, run continuously and stitched into one room:
  Depth-Anything-V2-Small -> normalise -> remap to a guessed 0.5-4 m range -> stride
  back-projection (fx=fy=0.9*W, camera ~1.5 m up) -> per-frame scale-lock so the clouds stay
  the same size -> ICP-align each frame onto the accumulated cloud -> merge + voxel-downsample.
An Open3D window shows the coloured room point cloud growing. Keys: S save .ply · R reset · Q quit

--tsdf switches to the metric-depth + RGBD-odometry + TSDF-mesh path (see reconstruct.py) —
better geometry, much heavier; needs a GPU to be pleasant.

Sweep tip: slow, steady, keep translating (small sideways drift as you turn); overlap heavily.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import open3d as o3d

from .depth import RelDepth, backproject, blur_score, central_median_depth
from .ingest import open_source


def _first_frame(src, timeout=25.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = src.read()
        if f is not None:
            return f
        time.sleep(0.1)
    raise RuntimeError("no frames within timeout — is the phone app running on the same Wi-Fi?")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="camera", help="phone:<ip> | http://... | file:<path> | camera")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--device", default=None, help="depth model device: cuda | cpu (default auto)")
    ap.add_argument("--max-width", type=int, default=480, help="pipeline working width")
    ap.add_argument("--stride", type=int, default=4, help="back-project every Nth pixel")
    ap.add_argument("--voxel", type=float, default=0.03, help="point-cloud merge/downsample voxel (m)")
    ap.add_argument("--depth-min", type=float, default=0.5)
    ap.add_argument("--depth-max", type=float, default=4.0)
    ap.add_argument("--cam-height", type=float, default=1.5)
    ap.add_argument("--fx-scale", type=float, default=0.9)
    ap.add_argument("--icp-dist", type=float, default=0.15, help="ICP max correspondence dist (m)")
    ap.add_argument("--max-points", type=int, default=1_500_000)
    ap.add_argument("--blur-min", type=float, default=18.0)
    ap.add_argument("--every", type=int, default=1, help="process 1 of every N source frames")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N s (0 = until Q)")
    ap.add_argument("--tsdf", action="store_true", help="use metric depth + RGBD odometry + TSDF mesh instead")
    ap.add_argument("--gpu", action="store_true", help="(with --tsdf) frame-to-model SLAM, CUDA Open3D")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    if args.tsdf:
        return _run_tsdf(args)

    print("[app] opening source:", args.source, flush=True)
    src = open_source(args.source, adb_serial=args.serial)
    try:
        _first_frame(src)
    except Exception:
        src.close()
        raise

    depth = RelDepth(device=args.device)
    print(f"[app] depth (relative) on {depth.device}", flush=True)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("SecondSense — Live Room", 1220, 820)
    world = o3d.geometry.PointCloud()
    vis.add_geometry(world)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
    st = {"quit": False, "reset": False, "save": False}
    vis.register_key_callback(ord("Q"), lambda _v: st.update(quit=True) or False)
    vis.register_key_callback(ord("R"), lambda _v: st.update(reset=True) or False)
    vis.register_key_callback(ord("S"), lambda _v: st.update(save=True) or False)

    pose = np.eye(4)
    ref_median = 0.0                     # running reference for the per-frame scale-lock
    t0, t_log = time.time(), 0.0
    n, n_reg, skip, first_view = 0, 0, 0, False
    print("[app] scanning — sweep slowly.  S save · R reset · Q quit", flush=True)
    try:
        while not st["quit"]:
            if args.seconds and time.time() - t0 > args.seconds:
                break
            raw = src.read()
            if raw is None:
                if args.source.startswith("file:"):
                    break
                vis.poll_events(); vis.update_renderer(); time.sleep(0.01); continue
            skip = (skip + 1) % max(1, args.every)
            if skip != 0:
                continue

            s = min(1.0, args.max_width / raw.shape[1])
            bgr = cv2.resize(raw, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else raw
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            blurry = blur_score(gray) < args.blur_min
            if not blurry:
                # scale-lock: bring this frame's central-median depth back to the reference,
                # so relative-depth frames are the same size and ICP can actually stitch them.
                raw_d = depth.infer_metricish(bgr, args.depth_min, args.depth_max, scale=1.0)
                med = central_median_depth(raw_d)
                sc = 1.0 if ref_median <= 0 or med <= 0 else float(np.clip(ref_median / med, 0.5, 2.0))
                dm = raw_d * sc
                if ref_median <= 0:
                    ref_median = central_median_depth(dm)
                else:
                    ref_median = 0.9 * ref_median + 0.1 * central_median_depth(dm)

                cur = backproject(bgr, dm, args.stride, args.fx_scale, args.cam_height,
                                  args.depth_min, args.depth_max).voxel_down_sample(args.voxel)
                if len(cur.points) > 50:
                    if len(world.points) == 0:
                        world.points, world.colors = cur.points, cur.colors
                    else:
                        reg = o3d.pipelines.registration.registration_icp(
                            cur, world, args.icp_dist, pose,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
                        if reg.fitness > 0.3 and np.isfinite(reg.transformation).all():
                            pose = reg.transformation
                            n_reg += 1
                        cur.transform(pose)
                        merged = (world + cur).voxel_down_sample(args.voxel)
                        if len(merged.points) > args.max_points:
                            merged = merged.random_down_sample(args.max_points / len(merged.points))
                        world.points, world.colors = merged.points, merged.colors
                    vis.update_geometry(world)
                    if not first_view:
                        vis.reset_view_point(True); first_view = True
                n += 1

            if st["reset"]:
                world.clear(); vis.update_geometry(world)
                pose = np.eye(4); ref_median = 0.0; n = n_reg = 0
                st["reset"] = False
                print("[app] reset", flush=True)
            if st["save"]:
                _save_pcd(world); st["save"] = False

            vis.poll_events(); vis.update_renderer()

            if time.time() - t_log > 3.0:
                t_log = time.time()
                tag = "blurry-hold" if blurry else "scanning"
                print(f"[app] {time.time()-t0:5.0f}s  frames={n} aligned={n_reg} "
                      f"points={len(world.points)}  {tag}", flush=True)

            if not args.no_preview:
                cv2.putText(bgr, f"frames {n}  aligned {n_reg}  pts {len(world.points)}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("SecondSense capture", bgr)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    st["quit"] = True
    finally:
        _save_pcd(world)
        src.close()
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("[app] done", flush=True)


def _save_pcd(pc) -> None:
    ts = time.strftime("%Y%m%d-%H%M%S")
    try:
        clean, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0) if len(pc.points) > 200 else (pc, None)
        o3d.io.write_point_cloud(f"room-{ts}.ply", clean)
        print(f"[app] saved room-{ts}.ply ({len(clean.points)} pts)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[app] save failed: {e}", flush=True)


# ------------------------------------------------------------------------------------------
# --tsdf : the heavier metric-depth + RGBD-odometry + TSDF-mesh path (reconstruct.py)
# ------------------------------------------------------------------------------------------
def _run_tsdf(args) -> None:
    import math
    from .depth import MetricDepth, TemporalDepthFilter
    from .reconstruct import make_reconstructor

    print("[app] --tsdf: metric depth + RGBD odometry + TSDF", flush=True)
    src = open_source(args.source, adb_serial=args.serial)
    try:
        f0 = _first_frame(src)
    except Exception:
        src.close()
        raise
    s = min(1.0, args.max_width / f0.shape[1])
    W, H = int(round(f0.shape[1] * s)), int(round(f0.shape[0] * s))
    fx = args.fx_scale * W
    intr = (W, H, fx, fx, W / 2.0, H / 2.0)
    depth = MetricDepth(device=args.device)
    tfilt = TemporalDepthFilter()
    recon = make_reconstructor(intr, voxel=max(0.01, args.voxel * 0.7),
                               depth_max=min(args.depth_max, 3.0), use_gpu=args.gpu)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("SecondSense — Live Room (TSDF)", 1220, 820)
    live = o3d.geometry.TriangleMesh()
    vis.add_geometry(live)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
    st = {"quit": False}
    vis.register_key_callback(ord("Q"), lambda _v: st.update(quit=True) or False)

    t0, t_log, n, n_ok = time.time(), 0.0, 0, 0
    try:
        while not st["quit"]:
            if args.seconds and time.time() - t0 > args.seconds:
                break
            raw = src.read()
            if raw is None:
                if args.source.startswith("file:"):
                    break
                vis.poll_events(); vis.update_renderer(); time.sleep(0.01); continue
            bgr = cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA) if s < 1 else raw
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if blur_score(gray) < args.blur_min:
                vis.poll_events(); vis.update_renderer(); continue
            d = tfilt.apply(gray, depth.infer(bgr))
            d = np.where((d > 0.05) & (d < min(args.depth_max, 3.0)), d, 0.0).astype(np.float32)
            n += 1
            if recon.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), d):
                n_ok += 1
            try:
                m = recon.extract_mesh()
                if m is not None and len(m.vertices):
                    if not m.has_vertex_normals():
                        m.compute_vertex_normals()
                    live.vertices, live.triangles = m.vertices, m.triangles
                    live.vertex_colors, live.vertex_normals = m.vertex_colors, m.vertex_normals
                    vis.update_geometry(live)
            except Exception:  # noqa: BLE001
                pass
            vis.poll_events(); vis.update_renderer()
            if time.time() - t_log > 3.0:
                t_log = time.time()
                print(f"[app] {time.time()-t0:5.0f}s  frames={n} tracked={n_ok}", flush=True)
    finally:
        ts = time.strftime("%Y%m%d-%H%M%S")
        try:
            o3d.io.write_triangle_mesh(f"room_mesh-{ts}.ply", recon.extract_mesh())
            o3d.io.write_point_cloud(f"room-{ts}.ply", recon.extract_pcd())
            print(f"[app] saved room_mesh-{ts}.ply / room-{ts}.ply", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[app] save failed: {e}", flush=True)
        src.close()
        vis.destroy_window()
        print("[app] done", flush=True)


if __name__ == "__main__":
    main()
