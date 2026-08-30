"""
Odometry + dense volumetric fusion for the 3D room demo, per the research:

  --gpu   TensorReconstructor  — o3d.t.pipelines.slam frame-to-MODEL tracking against a sparse
          VoxelBlockGrid (spatial hash). Raycasts a noise-free depth/normal map from the
          accumulated TSDF and aligns each new frame to THAT, which bounds local drift. Needs a
          CUDA-enabled Open3D wheel.

  default LegacyReconstructor — o3d.pipelines.odometry.compute_rgbd_odometry (Park 2017 hybrid
          photometric+geometric) frame-to-frame, integrated into a ScalableTSDFVolume. Pure
          CPU, installs anywhere, drifts a little more. Fine for a single slow room scan.

Both expose the same tiny surface used by app.py:
  process(rgb_uint8, depth_m_float32) -> bool     # False == odometry rejected, map "coasts"
  extract_mesh() / extract_pcd()
  .pose (4x4 cam->world), .trajectory, .blocks_used, .capacity, reset()
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

Intr = Tuple[int, int, float, float, float, float]  # w, h, fx, fy, cx, cy


def make_reconstructor(intr: Intr, voxel: float = 0.02, depth_max: float = 3.0,
                       use_gpu: bool = False, max_blocks: int = 50_000):
    if use_gpu:
        try:
            if o3d.core.cuda.is_available():
                return TensorReconstructor(intr, voxel, depth_max, max_blocks)
            print("[reconstruct] --gpu asked but Open3D has no CUDA; using CPU legacy path.")
        except Exception as e:  # noqa: BLE001
            print(f"[reconstruct] tensor path unavailable ({e}); using CPU legacy path.")
    return LegacyReconstructor(intr, voxel, depth_max)


# --------------------------------------------------------------------------------------------
class LegacyReconstructor:
    def __init__(self, intr: Intr, voxel: float, depth_max: float):
        w, h, fx, fy, cx, cy = intr
        self._pin = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        self.depth_max = depth_max
        self._voxel = voxel
        self._vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=3.0 * voxel,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self._prev: Optional[o3d.geometry.RGBDImage] = None
        self.pose = np.eye(4)
        self.trajectory: List[np.ndarray] = [self.pose.copy()]
        self._jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        self._opt = o3d.pipelines.odometry.OdometryOption()
        self._opt.depth_diff_max = 0.15         # tolerate a blurry pan (research)
        for name, val in (("depth_min", 0.1), ("depth_max", depth_max)):
            try:
                setattr(self._opt, name, val)
            except AttributeError:
                pass
        self.blocks_used = 0
        self.capacity = 0

    def _rgbd(self, rgb: np.ndarray, depth_m: np.ndarray) -> o3d.geometry.RGBDImage:
        col = o3d.geometry.Image(np.ascontiguousarray(rgb))
        dep = o3d.geometry.Image(np.ascontiguousarray(depth_m.astype(np.float32)))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            col, dep, depth_scale=1.0, depth_trunc=self.depth_max, convert_rgb_to_intensity=False)

    def process(self, rgb: np.ndarray, depth_m: np.ndarray) -> bool:
        rgbd = self._rgbd(rgb, depth_m)
        ok = True
        if self._prev is not None:
            try:
                success, T_prev_cur, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                    rgbd, self._prev, self._pin, np.eye(4), self._jac, self._opt)
            except Exception:  # noqa: BLE001
                success, T_prev_cur = False, np.eye(4)
            if success and np.isfinite(T_prev_cur).all():
                self.pose = self.pose @ T_prev_cur
            else:
                ok = False
        self._prev = rgbd
        if ok:
            self._vol.integrate(rgbd, self._pin, np.linalg.inv(self.pose))
            self.trajectory.append(self.pose.copy())
        return ok

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        m = self._vol.extract_triangle_mesh()
        m.compute_vertex_normals()
        return m

    def extract_pcd(self) -> o3d.geometry.PointCloud:
        return self._vol.extract_point_cloud()

    def reset(self) -> None:
        self.__init__((self._pin.width, self._pin.height,
                       self._pin.get_focal_length()[0], self._pin.get_focal_length()[1],
                       self._pin.get_principal_point()[0], self._pin.get_principal_point()[1]),
                      voxel=self._voxel, depth_max=self.depth_max)


# --------------------------------------------------------------------------------------------
class TensorReconstructor:
    def __init__(self, intr: Intr, voxel: float, depth_max: float, max_blocks: int):
        from open3d.t.pipelines import slam

        w, h, fx, fy, cx, cy = intr
        self._dev = o3d.core.Device("CUDA:0")
        self._intr = o3d.core.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3d.core.float64)
        self._w, self._h = w, h
        self.depth_max = depth_max
        self._slam = slam
        self._voxel = voxel
        self.capacity = max_blocks
        self.blocks_used = 0
        self._model = slam.Model(voxel, 16, max_blocks, o3d.core.Tensor(np.eye(4)), self._dev)
        self._input = slam.Frame(h, w, self._intr, self._dev)
        self._raycast = slam.Frame(h, w, self._intr, self._dev)
        self._i = 0
        self.pose = np.eye(4)
        self.trajectory: List[np.ndarray] = [self.pose.copy()]

    def process(self, rgb: np.ndarray, depth_m: np.ndarray) -> bool:
        col = o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(rgb))).to(self._dev)
        dep = o3d.t.geometry.Image(
            o3d.core.Tensor(np.ascontiguousarray(depth_m.astype(np.float32)))).to(self._dev)
        self._input.set_data_from_image("color", col)
        self._input.set_data_from_image("depth", dep)

        ok = True
        if self._i > 0:
            try:
                res = self._model.track_frame_to_model(
                    self._input, self._raycast, 1.0, float(self.depth_max))
                T = res.transformation.cpu().numpy()
                if np.isfinite(T).all():
                    self.pose = self.pose @ T
                else:
                    ok = False
            except Exception:  # noqa: BLE001
                ok = False

        if ok:
            self._model.update_frame_pose(self._i, o3d.core.Tensor(self.pose))
            self._model.integrate(self._input, 1.0, float(self.depth_max))
            self._model.synthesize_model_frame(self._raycast, 1.0, 0.1, float(self.depth_max))
            self.trajectory.append(self.pose.copy())
            self._i += 1
            try:
                self.blocks_used = int(self._model.voxel_grid.hashmap().size())
            except Exception:  # noqa: BLE001
                pass
        return ok

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        m = self._model.extract_trianglemesh()
        return m.to_legacy() if hasattr(m, "to_legacy") else m

    def extract_pcd(self) -> o3d.geometry.PointCloud:
        p = self._model.extract_pointcloud()
        return p.to_legacy() if hasattr(p, "to_legacy") else p

    def reset(self) -> None:
        self._model = self._slam.Model(self._voxel, 16, self.capacity,
                                       o3d.core.Tensor(np.eye(4)), self._dev)
        self._i = 0
        self.pose = np.eye(4)
        self.trajectory = [self.pose.copy()]
        self.blocks_used = 0
