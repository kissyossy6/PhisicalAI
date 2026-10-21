"""MuJoCo tabletop environment: Franka Panda + fruits / toy blocks + plate / toy box.

Provides
  * RGB-D rendering of the camera that Gemini Robotics-ER sees
  * back-projection of a normalized [y, x] point (0-1000) to a 3D table point
  * depth-based refinement of the grasp center / yaw around that point
  * damped-least-squares IK and simple motion primitives (move, grasp, release)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

SCENE_XML = os.path.join(os.path.dirname(__file__), "scene.xml")

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
TCP_OFFSET = np.array([0.0, 0.0, 0.105])  # hand frame -> fingertip center
PARK_Q = np.array([0.0, -0.9, 0.0, -2.75, 0.0, 1.9, 0.785])  # tucked back, out of the camera view
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0
CONTROL_DT = 0.02  # 50 Hz
VIDEO_DT = 1 / 25

ITEM_BODIES = ["apple", "orange", "lemon", "red_block", "blue_block", "green_block"]
CONTAINERS = {"plate": 0.0, "toybox": 0.0}


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def top_down_rotation(yaw: float) -> np.ndarray:
    """Hand frame rotation with the gripper pointing straight down."""
    return _rot_z(yaw) @ np.diag([1.0, -1.0, -1.0])


def min_jerk(t: float) -> float:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


@dataclass
class GraspTarget:
    pos: np.ndarray  # grasp center in world (m)
    yaw: float
    top_z: float
    n_points: int


class TidyEnv:
    def __init__(self, width: int = 800, height: int = 600, video_size=(1280, 720)):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.width, self.height = width, height

        self.rgb_renderer = mujoco.Renderer(self.model, height, width)
        self.depth_renderer = mujoco.Renderer(self.model, height, width)
        self.depth_renderer.enable_depth_rendering()
        self.video_renderer = mujoco.Renderer(self.model, video_size[1], video_size[0])

        self.arm_qadr = np.array([self.model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_dadr = np.array([self.model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.q_lo = self.model.jnt_range[[self.model.joint(j).id for j in ARM_JOINTS], 0]
        self.q_hi = self.model.jnt_range[[self.model.joint(j).id for j in ARM_JOINTS], 1]
        self.hand_id = self.model.body("hand").id

        self.frame_callback: Callable[[TidyEnv], None] | None = None
        self._since_frame = 0.0
        self.sim_time = 0.0
        self._yaw = 0.0
        self.reset()

    # ------------------------------------------------------------------ basics
    def reset(self, seed: int | None = None):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[self.arm_qadr] = PARK_Q
        self.data.ctrl[:7] = PARK_Q
        self.data.ctrl[7] = GRIPPER_OPEN
        if seed is not None:
            self.scatter_items(np.random.default_rng(seed))
        mujoco.mj_forward(self.model, self.data)
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

    def scatter_items(self, rng: np.random.Generator):
        """Randomly place the items in the free middle area of the table."""
        placed: list[np.ndarray] = []
        for name in ITEM_BODIES:
            body = self.model.body(name)
            qadr = self.model.jnt_qposadr[body.jntadr[0]]
            for _ in range(200):
                xy = np.array([rng.uniform(0.3, 0.68), rng.uniform(-0.2, 0.2)])
                if all(np.linalg.norm(xy - p) > 0.085 for p in placed):
                    break
            placed.append(xy)
            yaw = rng.uniform(-np.pi, np.pi)
            self.data.qpos[qadr : qadr + 2] = xy
            self.data.qpos[qadr + 3 : qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def set_body_pos(self, name: str, xy, yaw: float = 0.0):
        """Teleport an item (used to simulate a human messing up the table)."""
        body = self.model.body(name)
        qadr = self.model.jnt_qposadr[body.jntadr[0]]
        dadr = self.model.jnt_dofadr[body.jntadr[0]]
        self.data.qpos[qadr : qadr + 3] = [xy[0], xy[1], 0.05]
        self.data.qpos[qadr + 3 : qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[dadr : dadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------------- rendering
    def render_rgb(self, camera: str = "gemini") -> np.ndarray:
        self.rgb_renderer.update_scene(self.data, camera=camera)
        return self.rgb_renderer.render().copy()

    def render_depth(self, camera: str = "gemini") -> np.ndarray:
        self.depth_renderer.update_scene(self.data, camera=camera)
        return self.depth_renderer.render().copy()

    def render_video(self, camera: str = "cinema") -> np.ndarray:
        self.video_renderer.update_scene(self.data, camera=camera)
        return self.video_renderer.render().copy()

    def _camera(self, camera: str):
        cam = self.data.cam(camera)
        fovy = np.deg2rad(self.model.cam(camera).fovy[0])
        f = 0.5 * self.height / np.tan(fovy / 2)
        return cam.xpos.copy(), cam.xmat.reshape(3, 3).copy(), f

    def pixels_to_world(self, u: np.ndarray, v: np.ndarray, depth: np.ndarray, camera="gemini") -> np.ndarray:
        pos, rot, f = self._camera(camera)
        cx, cy = self.width / 2, self.height / 2
        z = depth
        p_cam = np.stack([(u - cx) / f * z, -(v - cy) / f * z, -z], axis=-1)
        return pos + p_cam @ rot.T

    def world_to_point(self, p: np.ndarray, camera="gemini") -> list[int]:
        """World point -> normalized [y, x] (0-1000), the format Gemini uses."""
        pos, rot, f = self._camera(camera)
        pc = rot.T @ (p - pos)
        u = self.width / 2 + f * pc[0] / -pc[2]
        v = self.height / 2 - f * pc[1] / -pc[2]
        return [int(round(v / self.height * 1000)), int(round(u / self.width * 1000))]

    def point_to_world(self, point_yx, depth: np.ndarray, camera="gemini") -> np.ndarray:
        v = np.clip(point_yx[0] / 1000 * self.height, 0, self.height - 1)
        u = np.clip(point_yx[1] / 1000 * self.width, 0, self.width - 1)
        # median depth in a small window is robust to edge pixels
        r = 2
        vi, ui = int(v), int(u)
        patch = depth[max(vi - r, 0) : vi + r + 1, max(ui - r, 0) : ui + r + 1]
        return self.pixels_to_world(np.array(u), np.array(v), np.array(np.median(patch)), camera)

    def refine_grasp(self, point_yx, depth: np.ndarray, radius: float = 0.05) -> GraspTarget:
        """Gemini gives *which* object (a 2D point). Depth tells us *exactly where*.

        Back-project the whole depth image, keep the points that stick out of the
        table near the Gemini point, and take their centroid / top height. The yaw
        is the angle of the minimum-area XY bounding rectangle (works for cubes).
        """
        seed = self.point_to_world(point_yx, depth)
        vv, uu = np.mgrid[0 : self.height, 0 : self.width]
        pts = self.pixels_to_world(uu.ravel(), vv.ravel(), depth.ravel())
        d_xy = np.linalg.norm(pts[:, :2] - seed[:2], axis=1)
        mask = (d_xy < radius) & (pts[:, 2] > 0.012) & (pts[:, 2] < 0.15)
        obj = pts[mask]
        if len(obj) < 30:
            return GraspTarget(seed, 0.0, max(seed[2], 0.03), len(obj))
        # keep the cluster around the highest density near the seed
        top_z = np.percentile(obj[:, 2], 90)
        upper = obj[obj[:, 2] > top_z * 0.45]
        center = upper[:, :2].mean(axis=0)
        best_yaw, best_area = 0.0, np.inf
        xy = upper[:, :2] - center
        for yaw in np.deg2rad(np.arange(0, 90, 3)):
            r = xy @ _rot_z(yaw)[:2, :2]
            area = np.ptp(r[:, 0]) * np.ptp(r[:, 1])
            if area < best_area:
                best_area, best_yaw = area, yaw
        if best_yaw > np.pi / 4:
            best_yaw -= np.pi / 2
        # the oblique view sees the front faces, which biases the centroid toward
        # the camera; use the top surface only for the XY center
        top = obj[obj[:, 2] > top_z - 0.008]
        if len(top) > 20:
            center = top[:, :2].mean(axis=0)
        grasp_z = max(top_z * 0.5, 0.014)
        return GraspTarget(np.array([center[0], center[1], grasp_z]), float(best_yaw), float(top_z), len(obj))

    # ---------------------------------------------------------------------- IK
    def tcp_pose(self, data=None):
        data = data or self.data
        rot = data.xmat[self.hand_id].reshape(3, 3)
        return data.xpos[self.hand_id] + rot @ TCP_OFFSET, rot.copy()

    def solve_ik(self, pos, rot, q_init, iters=60, tol=1e-4) -> np.ndarray:
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        q = q_init.copy()
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iters):
            d.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            cur_p, cur_r = self.tcp_pose(d)
            err_p = pos - cur_p
            r_err = rot @ cur_r.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, r_err.ravel())
            err_r = np.zeros(3)
            mujoco.mju_quat2Vel(err_r, quat, 1.0)
            err = np.concatenate([err_p, err_r])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jac(self.model, d, jacp, jacr, cur_p, self.hand_id)
            J = np.vstack([jacp[:, self.arm_dadr], jacr[:, self.arm_dadr]])
            lam = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
            # small pull toward a comfortable posture in the null space
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (0.05 * (PARK_Q_NULL - q))
            q = np.clip(q + dq, self.q_lo + 1e-3, self.q_hi - 1e-3)
        return q

    # --------------------------------------------------------------- stepping
    def step(self, duration: float):
        n = int(round(duration / self.model.opt.timestep))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.model.opt.timestep
            self._since_frame += self.model.opt.timestep
            if self._since_frame >= VIDEO_DT - 1e-9:
                self._since_frame -= VIDEO_DT
                if self.frame_callback:
                    self.frame_callback(self)

    def wait(self, duration: float):
        self.step(duration)

    # -------------------------------------------------------------- primitives
    def move_joints(self, q_goal: np.ndarray, duration: float = 2.0):
        q0 = self.data.ctrl[:7].copy()
        steps = max(int(duration / CONTROL_DT), 1)
        for i in range(1, steps + 1):
            self.data.ctrl[:7] = q0 + (q_goal - q0) * min_jerk(i / steps)
            self.step(CONTROL_DT)

    def move_tcp(self, pos, yaw: float, duration: float = 1.5):
        """Straight-line Cartesian motion of the fingertip center (top-down)."""
        p0, r0 = self.tcp_pose()
        rot_goal = top_down_rotation(yaw)
        q0 = quat_from_mat(r0)
        q1 = quat_from_mat(rot_goal)
        steps = max(int(duration / CONTROL_DT), 1)
        q = self.data.ctrl[:7].copy()
        self._yaw = yaw
        for i in range(1, steps + 1):
            s = min_jerk(i / steps)
            p = p0 + (np.asarray(pos) - p0) * s
            quat = slerp(q0, q1, s)
            r = np.zeros(9)
            mujoco.mju_quat2Mat(r, quat)
            q = self.solve_ik(p, r.reshape(3, 3), q, iters=15)
            self.data.ctrl[:7] = q
            self.step(CONTROL_DT)
        self.wait(0.15)

    def gripper(self, value: float, duration: float = 0.6):
        self.data.ctrl[7] = value
        self.wait(duration)

    def park(self, duration: float = 1.6):
        p, _ = self.tcp_pose()
        if p[2] < SAFE_Z - 0.02 and np.linalg.norm(self.data.ctrl[:7] - PARK_Q) > 0.1:
            self.move_tcp([p[0], p[1], SAFE_Z], self._yaw, 0.8)
        self.move_joints(PARK_Q, duration)

    def go_above(self, xy, yaw: float, duration: float = 1.4):
        """From anywhere to a safe height above xy, without sweeping the table."""
        if np.linalg.norm(self.data.ctrl[:7] - PARK_Q) < 0.1:
            q_home = self.solve_ik(HOME_POS, top_down_rotation(0.0), PARK_Q_NULL, iters=200)
            self.move_joints(q_home, 1.2)
            self._yaw = 0.0
        self.move_tcp([xy[0], xy[1], SAFE_Z], yaw, duration)

    def pick(self, grasp: GraspTarget) -> bool:
        g = grasp.pos
        self.gripper(GRIPPER_OPEN, 0.1)
        self.go_above(g[:2], grasp.yaw)
        self.move_tcp([g[0], g[1], g[2] + 0.06], grasp.yaw, 0.8)
        self.move_tcp(g, grasp.yaw, 0.6)
        self.gripper(GRIPPER_CLOSED, 0.6)
        self.move_tcp([g[0], g[1], SAFE_Z], grasp.yaw, 1.0)
        return self.holding()

    def place(self, place_xy, place_z: float):
        self.move_tcp([place_xy[0], place_xy[1], SAFE_Z], self._yaw, 1.4)
        self.move_tcp([place_xy[0], place_xy[1], place_z + 0.035], self._yaw, 0.8)
        self.gripper(GRIPPER_OPEN, 0.5)
        self.move_tcp([place_xy[0], place_xy[1], SAFE_Z], self._yaw, 0.6)

    def pick_and_place(self, grasp: GraspTarget, place_xy, place_z: float) -> bool:
        ok = self.pick(grasp)
        if ok:
            self.place(place_xy, place_z)
        else:
            self.gripper(GRIPPER_OPEN, 0.3)
        return ok

    def holding(self) -> bool:
        """Gripper fully closed means nothing between the fingers."""
        width = self.data.qpos[7] + self.data.qpos[8]
        return width > 0.01


SAFE_Z = 0.22
HOME_POS = np.array([0.45, 0.0, 0.35])
PARK_Q_NULL = np.array([0.0, 0.2, 0.0, -2.0, 0.0, 2.2, 0.785])


def quat_from_mat(r: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, r.ravel())
    return q


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)
