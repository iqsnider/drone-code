"""Live 6-DoF pose of the drone from the two-camera IR rig.

The world frame is the one the ArUco calibration established: origin at the
marker centre, +Z up. Each camera's pose in it comes from the "extrinsics"
block of its config.

Per frame: threshold both images to IR blobs, pair the three blobs across the
two views, triangulate them, then fit the known body-frame LED triangle onto
those three world points. The fit gives position and full orientation, and its
residual (rmsd) is the honest quality signal -- a bad pairing or a stray
reflection shows up as a large rmsd rather than a plausible-looking wrong pose.
"""
import json
import time
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np

from calibration.camera_setup import (CAMERA_FILES, detect_centroids,
                                      open_cameras, read_grays)
from calibration import camera_ids
from drone.state_estimation import PoseEKF

N_LEDS = 3


# ---------------------------------------------------------------------------
# camera model
# ---------------------------------------------------------------------------
class CameraModel:
    """Pinhole camera with a known world pose.

    Points are undistorted before use, so the projection matrix itself carries
    no distortion terms.
    """

    def __init__(self, cfg):
        self.K = np.array([[cfg["fx"], 0.0, cfg["cx"]],
                           [0.0, cfg["fy"], cfg["cy"]],
                           [0.0, 0.0, 1.0]], float)
        self.dist = np.array(cfg["dist"], float).reshape(-1, 1)
        ex = cfg["extrinsics"]
        self.R = cv2.Rodrigues(np.array(ex["rvec"], float))[0]   # world -> cam
        self.t = np.array(ex["tvec"], float)
        self.C = -self.R.T @ self.t                              # centre in world
        self.P = self.K @ np.hstack([self.R, self.t[:, None]])

    def undistort(self, pts):
        """Pixel points -> ideal pinhole pixel points for this camera."""
        pts = np.asarray(pts, float).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)

    def project(self, Xw):
        Xw = np.atleast_2d(np.asarray(Xw, float))
        uv = (self.P @ np.hstack([Xw, np.ones((len(Xw), 1))]).T).T
        return uv[:, :2] / uv[:, 2:3]


# ---------------------------------------------------------------------------
# triangulation + stereo pairing
# ---------------------------------------------------------------------------
def triangulate(cam0, cam1, uv0, uv1):
    """Linear DLT triangulation of one correspondence."""
    P0, P1 = cam0.P, cam1.P
    (u0, v0), (u1, v1) = uv0, uv1
    A = np.stack([u0 * P0[2] - P0[0],
                  v0 * P0[2] - P0[1],
                  u1 * P1[2] - P1[0],
                  v1 * P1[2] - P1[1]])
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3]


def match_stereo(cam0, cam1, pts0, pts1):
    """Pair the blobs across views by lowest reprojection error.

    Returns (world_points, reproj_px) for the best of the 3! pairings.
    """
    best = None
    for perm in permutations(range(len(pts0))):
        Xs = np.array([triangulate(cam0, cam1, pts0[i], pts1[perm[i]])
                       for i in range(len(pts0))])
        err = 0.0
        for i in range(len(pts0)):
            err += np.linalg.norm(cam0.project(Xs[i])[0] - pts0[i])
            err += np.linalg.norm(cam1.project(Xs[i])[0] - pts1[perm[i]])
        err /= 2 * len(pts0)
        if best is None or err < best[1]:
            best = (Xs, err)
    return best


# ---------------------------------------------------------------------------
# rigid fit of the known LED triangle
# ---------------------------------------------------------------------------
def kabsch(B, W):
    """Rigid transform taking BODY points B onto WORLD points W.

    Returns (R_bw, p, rmsd) with W ~= (R_bw @ B.T).T + p.
    """
    bc, wc = B.mean(0), W.mean(0)
    U, _, Vt = np.linalg.svd((B - bc).T @ (W - wc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    p = wc - R @ bc
    resid = W - ((R @ B.T).T + p)
    return R, p, float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))


def identify_and_pose(world_pts, led_body):
    """Label the unordered world points against the LED geometry, then fit.

    Returns (R_bw, p, rmsd) for the labelling with the smallest residual.
    """
    best = None
    for perm in permutations(range(len(led_body))):
        R, p, rmsd = kabsch(led_body, world_pts[list(perm)])
        if best is None or rmsd < best[2]:
            best = (R, p, rmsd)
    return best


def rpy_deg(R_bw):
    """Roll/pitch/yaw in degrees, ZYX convention, from a body->world rotation."""
    yaw = np.arctan2(R_bw[1, 0], R_bw[0, 0])
    pitch = np.arctan2(-R_bw[2, 0], np.hypot(R_bw[2, 1], R_bw[2, 2]))
    roll = np.arctan2(R_bw[2, 1], R_bw[2, 2])
    return [float(np.degrees(a)) for a in (roll, pitch, yaw)]


# ---------------------------------------------------------------------------
# the rig
# ---------------------------------------------------------------------------
class PoseTracker:
    """Owns the cameras and turns each pair of frames into a pose."""

    def __init__(self, cfgdir):
        cfgdir = Path(cfgdir)
        self.cam_cfg = [json.loads((cfgdir / n).read_text()) for n in CAMERA_FILES]
        self.drone = json.loads((cfgdir / "drone.json").read_text())
        self.models = [CameraModel(c) for c in self.cam_cfg]
        self.led_body = np.asarray(self.drone["led_body"], float)
        # Threshold for a raw fit being usable as an EKF seed. Its only job is
        # to exclude a garbage frame -- a mis-paired stereo match puts rmsd in
        # the hundreds of millimetres, while an honest fit sits near the error
        # in led_body itself. It is deliberately NOT control.reject_rmsd_m:
        # that is the control layer's quality bar, and any error in the
        # measured LED geometry puts a floor under rmsd that no frame can beat.
        filt = self.drone.get("filter") or {}
        self.init_rmsd = float(filt.get("init_rmsd_m", 0.05))
        self.ekf = PoseEKF(self.led_body, filt)
        self.cam = None
        self._last_t = None

    def open(self):
        self.cam = open_cameras(self.cam_cfg, camera_ids.resolve_indices(self.cam_cfg))

    def close(self):
        if self.cam is not None:
            self.cam.end()
            self.cam = None

    def estimate(self, pts0, pts1):
        """Pose from two lists of pixel centroids, or None if under-determined."""
        if len(pts0) < N_LEDS or len(pts1) < N_LEDS:
            return None
        u0 = self.models[0].undistort(pts0[:N_LEDS])
        u1 = self.models[1].undistort(pts1[:N_LEDS])
        world, reproj = match_stereo(self.models[0], self.models[1], u0, u1)
        R, p, rmsd = identify_and_pose(world, self.led_body)
        return {"pos": p.tolist(), "R": R.tolist(), "rpy": rpy_deg(R),
                "leds": world.tolist(), "rmsd": float(rmsd),
                "reproj": float(reproj), "ok": bool(rmsd <= self.init_rmsd)}

    def read(self):
        """Grab both cameras, estimate, and step the filter.

        Returns {"raw": pose or None, "est": filter state or None, ...}. The
        filter is fed the triangulated points rather than the raw pose, because
        it does its own labelling and can therefore recover from a frame the
        raw fit got wrong.
        """
        grays = read_grays(self.cam)
        dets = [detect_centroids(g, c, max_n=N_LEDS)[0]
                for g, c in zip(grays, self.cam_cfg)]
        raw = self.estimate(dets[0], dets[1])

        now = time.time()
        dt = (now - self._last_t) if self._last_t is not None else 0.0
        self._last_t = now
        self.ekf.predict(dt)

        if raw is not None:
            if self.ekf.valid:
                raw["accepted"] = self.ekf.update(np.asarray(raw["leds"], float))
            elif raw["ok"]:
                # Only a fit the rig itself trusts is allowed to seed the filter;
                # a bad seed would otherwise gate every later good frame out.
                self.ekf.initialize(raw["pos"], np.asarray(raw.pop("R"), float))
                raw["accepted"] = True
            else:
                raw["accepted"] = False
            raw.pop("R", None)                  # not needed on the wire

        est = self.ekf.state()
        est["rpy"] = rpy_deg(np.asarray(est.pop("R"), float))
        return {"t": now, "dt": dt, "blobs": [len(d) for d in dets],
                "cams": [m.C.tolist() for m in self.models],
                "raw": raw, "est": est if est["valid"] else None}
