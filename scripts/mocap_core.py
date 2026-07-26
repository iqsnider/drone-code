"""
mocap_core.py -- geometry + pose estimation for the 2-camera IR mocap rig.

Pure NumPy (no OpenCV/hardware), so the whole estimation chain can be unit
tested off-hardware. Conventions used everywhere in this project:

WORLD frame  : X = right (camera0 -> camera1 direction), Y = forward (from the
               cameras into the working volume), Z = up. Origin on the floor,
               directly beneath the desired hover point. Right handed (X x Y = Z).
BODY frame   : X = forward, Y = left, Z = up (FLU). Origin at the drone CoM.
CAMERA frame : OpenCV convention -- X right, Y down, Z forward (into scene).

A drone pose is (R_bw, p) where R_bw maps BODY vectors to WORLD and p is the CoM
position in WORLD:   world_point = R_bw @ body_point + p.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------
def look_at_R(C, T, world_up=(0.0, 0.0, 1.0)):
    """Rotation world->camera (OpenCV frame) for a camera at C looking at T.

    Rows of the returned R are the camera axes (x_right, y_down, z_forward)
    expressed in world coordinates.
    """
    C = np.asarray(C, float); T = np.asarray(T, float)
    up = np.asarray(world_up, float)
    z = T - C
    z /= np.linalg.norm(z)                       # forward (+Z cam) in world
    if abs(np.dot(z, up)) > 0.999:               # looking near-straight up/down
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up); x /= np.linalg.norm(x)  # right (+X cam)
    y = np.cross(z, x)                           # down  (+Y cam); z x x = y
    return np.vstack([x, y, z])


def K_from_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def rodrigues(rvec):
    """Rotation vector -> 3x3 matrix (pure NumPy; matches cv2.Rodrigues)."""
    r = np.asarray(rvec, float).reshape(3)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


class CameraModel:
    """Pinhole camera (no distortion) with a full world pose."""
    def __init__(self, K, R_wc, C):
        self.K = np.asarray(K, float)            # 3x3 intrinsics
        self.R = np.asarray(R_wc, float)         # 3x3 world->camera
        self.C = np.asarray(C, float)            # camera centre in world
        self.t = -self.R @ self.C                # translation world->camera
        self.P = self.K @ np.hstack([self.R, self.t[:, None]])   # 3x4

    @classmethod
    def from_look_at(cls, K, C, target, up=(0, 0, 1)):
        return cls(K, look_at_R(C, target, up), C)

    @classmethod
    def from_rvec_tvec(cls, K, rvec, tvec):
        """Build from an OpenCV solvePnP result. Convention: X_cam = R@X_world +
        tvec with R = Rodrigues(rvec) (world->camera), so the camera centre in
        world is C = -R^T @ tvec. Captures full orientation incl. lens roll."""
        R = rodrigues(rvec)
        C = -R.T @ np.asarray(tvec, float).reshape(3)
        return cls(K, R, C)

    def project(self, Xw):
        """World point(s) -> pixel(s). Xw shape (3,) or (N,3)."""
        Xw = np.atleast_2d(np.asarray(Xw, float))
        Xc = (self.R @ Xw.T).T + self.t          # into camera frame
        uv = (self.K @ Xc.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        return uv if uv.shape[0] > 1 else uv[0]

    def in_front(self, Xw):
        Xc = self.R @ (np.asarray(Xw, float) - self.C)
        return Xc[2] > 0


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------
def triangulate(cam0, cam1, uv0, uv1):
    """Linear DLT triangulation of one correspondence from two cameras."""
    P0, P1 = cam0.P, cam1.P
    u0, v0 = uv0; u1, v1 = uv1
    A = np.stack([
        u0 * P0[2] - P0[0],
        v0 * P0[2] - P0[1],
        u1 * P1[2] - P1[0],
        v1 * P1[2] - P1[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def reprojection_error(cam0, cam1, uv0, uv1, Xw):
    e0 = np.linalg.norm(cam0.project(Xw) - np.asarray(uv0))
    e1 = np.linalg.norm(cam1.project(Xw) - np.asarray(uv1))
    return 0.5 * (e0 + e1)


# ---------------------------------------------------------------------------
# Stereo correspondence (which blob in cam1 matches which in cam0)
# ---------------------------------------------------------------------------
from itertools import permutations


def match_stereo(cam0, cam1, pts0, pts1):
    """Return (Xs, perm, err): triangulated world points for the best pairing
    of pts0[i] <-> pts1[perm[i]], plus mean reprojection error. Expects exactly
    3 points in each list (brute force over the 6 permutations)."""
    pts0 = np.asarray(pts0, float); pts1 = np.asarray(pts1, float)
    n = len(pts0)
    best = None
    for perm in permutations(range(n)):
        Xs = np.array([triangulate(cam0, cam1, pts0[i], pts1[perm[i]])
                       for i in range(n)])
        err = np.mean([reprojection_error(cam0, cam1, pts0[i], pts1[perm[i]], Xs[i])
                       for i in range(n)])
        if best is None or err < best[2]:
            best = (Xs, perm, err)
    return best


# ---------------------------------------------------------------------------
# Rigid pose from labelled point correspondences (Kabsch / Umeyama, no scale)
# ---------------------------------------------------------------------------
def kabsch(B, W):
    """Best rigid transform mapping BODY points B -> WORLD points W.
    Returns (R_bw, p, rmsd) with  W ~= (R_bw @ B.T).T + p."""
    B = np.asarray(B, float); W = np.asarray(W, float)
    bc = B.mean(0); wc = W.mean(0)
    H = (B - bc).T @ (W - wc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    p = wc - R @ bc
    resid = W - ((R @ B.T).T + p)
    rmsd = np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))
    return R, p, rmsd


def identify_and_pose(world_pts, led_body):
    """Given 3 unlabelled triangulated WORLD points and the known BODY-frame LED
    positions, find the labelling + rigid pose that best fits (min RMSD).
    Returns (R_bw, p, rmsd, order) where world_pts[order[k]] <-> led_body[k]."""
    world_pts = np.asarray(world_pts, float)
    led_body = np.asarray(led_body, float)
    best = None
    for perm in permutations(range(len(led_body))):
        W = world_pts[list(perm)]
        R, p, rmsd = kabsch(led_body, W)
        if best is None or rmsd < best[2]:
            best = (R, p, rmsd, perm)
    return best


def yaw_from_R(R_bw):
    """Heading (rad) = angle of body-forward (+X) in the world XY-plane,
    measured from world +X toward world +Y."""
    fwd = R_bw[:, 0]
    return np.arctan2(fwd[1], fwd[0])


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# State filter: 3D alpha-beta (g-h) tracker with outlier gating + yaw low-pass
# ---------------------------------------------------------------------------
class StateFilter:
    def __init__(self, alpha=0.5, beta=0.05, gate_m=0.30):
        self.alpha = alpha; self.beta = beta; self.gate = gate_m
        self.x = None; self.v = np.zeros(3)
        self._sc = None  # (sin, cos) of yaw, low-passed
        self.valid = False; self.misses = 0

    def predict(self, dt):
        if self.x is not None:
            self.x = self.x + self.v * dt

    def update(self, meas, yaw, dt):
        meas = np.asarray(meas, float)
        if self.x is None:
            self.x = meas.copy(); self.v[:] = 0
            self._sc = np.array([np.sin(yaw), np.cos(yaw)])
            self.valid = True; self.misses = 0
            return True
        r = meas - self.x
        if np.linalg.norm(r) > self.gate:        # reject implausible jump
            self.misses += 1
            self.valid = self.misses < 5
            return False
        self.x = self.x + self.alpha * r
        self.v = self.v + self.beta * r / max(dt, 1e-3)
        a = 0.3
        self._sc = (1 - a) * self._sc + a * np.array([np.sin(yaw), np.cos(yaw)])
        self.misses = 0; self.valid = True
        return True

    @property
    def yaw(self):
        return 0.0 if self._sc is None else np.arctan2(self._sc[0], self._sc[1])


# ---------------------------------------------------------------------------
# Hover controller: world position/yaw -> (roll_deg, pitch_deg, yaw_norm, throttle)
# ---------------------------------------------------------------------------
class HoverController:
    """ANGLE-mode outer loop. Betaflight does the inner attitude stabilisation;
    we command lean angles (deg) and throttle (0..1). All sign_* are unknown for
    a given airframe/BF setup and MUST be verified at low authority before flight.
    """
    def __init__(self, g):
        self.kp_xy = g["kp_xy"]; self.kd_xy = g["kd_xy"]; self.ki_xy = g["ki_xy"]
        self.kp_z = g["kp_z"];  self.kd_z = g["kd_z"];  self.ki_z = g["ki_z"]
        self.kp_yaw = g["kp_yaw"]
        self.hover_ff = g["hover_ff"]
        self.max_tilt = g["max_tilt_deg"]
        self.thr_cap = g["throttle_cap"]
        self.i_xy_lim = g.get("i_xy_limit", 3.0)
        self.i_z_lim = g.get("i_z_limit", 0.15)
        self.s_roll = g.get("sign_roll", 1.0)
        self.s_pitch = g.get("sign_pitch", 1.0)
        self.s_yaw = g.get("sign_yaw", 1.0)
        self.reset()

    def reset(self):
        self.ix = self.iy = self.iz = 0.0

    def __call__(self, pos, vel, yaw, sp, yaw_sp, dt, integrate=True):
        x, y, z = pos; vx, vy, vz = vel
        xs, ys, zs = sp
        ex, ey, ez = xs - x, ys - y, zs - z
        if integrate:
            self.ix = np.clip(self.ix + ex * dt, -self.i_xy_lim, self.i_xy_lim)
            self.iy = np.clip(self.iy + ey * dt, -self.i_xy_lim, self.i_xy_lim)
            self.iz = np.clip(self.iz + ez * dt, -self.i_z_lim, self.i_z_lim)
        # world-frame horizontal demand (PD + I)
        ax = self.kp_xy * ex - self.kd_xy * vx + self.ki_xy * self.ix
        ay = self.kp_xy * ey - self.kd_xy * vy + self.ki_xy * self.iy
        # rotate world demand into body frame (forward=+X, left=+Y at heading yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        a_fwd = ax * c + ay * s
        a_left = -ax * s + ay * c
        pitch = np.clip(self.s_pitch * a_fwd, -self.max_tilt, self.max_tilt)
        roll = np.clip(self.s_roll * (-a_left), -self.max_tilt, self.max_tilt)  # right = -left
        # altitude
        thr = self.hover_ff + self.kp_z * ez - self.kd_z * vz + self.ki_z * self.iz
        thr = float(np.clip(thr, 0.0, self.thr_cap))
        # yaw (rate command)
        eyaw = wrap_pi(yaw_sp - yaw)
        yaw_cmd = float(np.clip(self.s_yaw * self.kp_yaw * eyaw, -1.0, 1.0))
        return roll, pitch, yaw_cmd, thr
