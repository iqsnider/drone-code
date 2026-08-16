import base64
import json
import time
from itertools import combinations, permutations
from pathlib import Path

import cv2
import numpy as np

from calibration.camera_setup import detect_centroids, open_cameras, read_grays
from calibration import camera_ids
from drone.state_estimation import PoseEKF

N_LEDS = 3
MIN_CAMS = 2                 # a 3D point needs at least two lines of sight

PREVIEW_SIZE = (320, 240)    # panel-sized, so the wire carries no wasted pixels
PREVIEW_QUALITY = 55


class CameraModel:

    def __init__(self, cfg):
        self.K = np.array([[cfg["fx"], 0, cfg["cx"]],
                           [0, cfg["fy"], cfg["cy"]],
                           [0, 0, 1]], float)
        self.dist = np.array(cfg["dist"], float).reshape(-1, 1)
        ex = cfg["extrinsics"]
        self.R = cv2.Rodrigues(np.array(ex["rvec"], float))[0]   # world -> cam
        self.t = np.array(ex["tvec"], float)
        self.C = -self.R.T @ self.t                              # center in world
        self.P = self.K @ np.hstack([self.R, self.t[:, None]])

    def undistort(self, pts):
        """Pixel points -> ideal pinhole pixel points for this camera."""
        pts = np.asarray(pts, float).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)

    def project(self, Xw):
        Xw = np.atleast_2d(np.asarray(Xw, float))
        uv = (self.P @ np.hstack([Xw, np.ones((len(Xw), 1))]).T).T
        return uv[:, :2] / uv[:, 2:3]


def triangulate(models, uvs):
    """
    Least-squares 3D point from N >= 2 views (homogeneous DLT).

    Each view contributes two rows saying "the ray through this pixel passes
    through X"; the null space of the stack is the point that best satisfies
    all of them at once.
    """
    A = np.empty((2 * len(models), 4))
    for i, (m, (u, v)) in enumerate(zip(models, uvs)):
        A[2 * i] = u * m.P[2] - m.P[0]
        A[2 * i + 1] = v * m.P[2] - m.P[1]
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3]


def _best_perm(cost):
    """Assignment that minimises cost[i, perm[i]], brute forced over N_LEDS!."""
    n = len(cost)
    return min(permutations(range(n)),
               key=lambda p: sum(cost[i, p[i]] for i in range(n)))


def match_views(models, uvs):
    """
    Label every camera's blobs consistently, then triangulate from all of them.

    The first view fixes the LED labelling. The second is matched to it by
    brute force -- the only pairing we have no 3D prior for -- and the rest are
    matched by reprojecting the resulting points, which costs one projection
    per view instead of a joint search over every view at once.
    """
    ref = uvs[0]
    n = len(ref)

    # bootstrap 3D points from the first pair
    best = None
    for perm in permutations(range(n)):
        pair = [models[0], models[1]]
        Xs = np.array([triangulate(pair, [ref[i], uvs[1][perm[i]]])
                       for i in range(n)])
        err = sum(np.linalg.norm(models[0].project(Xs[i])[0] - ref[i]) +
                  np.linalg.norm(models[1].project(Xs[i])[0] - uvs[1][perm[i]])
                  for i in range(n))
        if best is None or err < best[1]:
            best = (Xs, err, list(perm))
    Xs, _, perm1 = best

    # order the remaining views against those points
    ordered = [ref, uvs[1][perm1]]
    for m, uv in zip(models[2:], uvs[2:]):
        pred = m.project(Xs)
        cost = np.linalg.norm(pred[:, None, :] - uv[None, :, :], axis=2)
        ordered.append(uv[list(_best_perm(cost))])

    # re-triangulate each LED using every view that saw it
    world = np.array([triangulate(models, [o[k] for o in ordered])
                      for k in range(n)])

    err = sum(np.linalg.norm(m.project(world[k])[0] - o[k])
              for m, o in zip(models, ordered) for k in range(n))
    return world, err / (n * len(models))


def kabsch(B, W):
    """
    rigid fit of the led triangle
    """
    bc, wc = B.mean(0), W.mean(0)
    U, _, Vt = np.linalg.svd((B - bc).T @ (W - wc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    p = wc - R @ bc
    resid = W - ((R @ B.T).T + p)
    return R, p, float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))


def identify_and_pose(world_pts, led_body):
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


class PoseTracker:
    def __init__(self, cfgdir, preview=False):
        # preview costs a JPEG encode per camera per frame, so the flight loops
        # leave it off and only the viewer asks for it
        self.preview = preview
        cfgdir = Path(cfgdir)
        names = camera_ids.camera_files(cfgdir)
        self.cam_cfg = [json.loads((cfgdir / n).read_text()) for n in names]

        uncalibrated = [n for n, c in zip(names, self.cam_cfg)
                        if "extrinsics" not in c]
        if uncalibrated:
            raise SystemExit(
                "no extrinsics for " + ", ".join(uncalibrated) +
                " -- run scripts/run_camera_pose_calibration.py first")

        self.drone = json.loads((cfgdir / "drone.json").read_text())
        self.models = [CameraModel(c) for c in self.cam_cfg]

        # pairwise camera separations, used to pick the stereo pair that
        # bootstraps LED correspondence each frame
        centres = np.array([m.C for m in self.models])
        self.baseline = np.linalg.norm(centres[:, None, :] - centres[None, :, :],
                                       axis=2)
        self.led_body = np.asarray(self.drone["led_body"], float)

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

    def estimate(self, dets):
        """
        Pose from one list of pixel centroids per camera, or None if
        under-determined. Cameras that do not see all the LEDs sit this frame
        out, so any MIN_CAMS of them are enough to stay tracking.
        """
        use = [i for i, d in enumerate(dets) if len(d) >= N_LEDS]
        if len(use) < MIN_CAMS:
            return None

        # match_views bootstraps correspondence from the first two cameras it is
        # given, so lead with the widest-separated pair. Across a short baseline
        # several LED pairings reproject almost equally well and the wrong one
        # wins, triangulating ghost points -- on this rig that doubled the
        # apparent size of the LED triangle on ~80% of frames.
        a, b = max(combinations(use, 2), key=lambda p: self.baseline[p])
        order = [a, b] + [i for i in use if i not in (a, b)]

        models = [self.models[i] for i in order]
        uvs = [self.models[i].undistort(dets[i][:N_LEDS]) for i in order]
        world, reproj = match_views(models, uvs)
        R, p, rmsd = identify_and_pose(world, self.led_body)

        return {"pos": p.tolist(), "R": R.tolist(), "rpy": rpy_deg(R),
                "leds": world.tolist(), "rmsd": float(rmsd),
                "reproj": float(reproj), "used": sorted(order),
                "ok": bool(rmsd <= self.init_rmsd)}

    def _preview(self, gray, pts):
        """One camera's frame as base64 JPEG, with its detected blobs ringed.

        Resized before drawing so the markers stay crisp at panel size rather
        than being softened by the downscale.
        """
        g = np.asarray(gray)
        sx = PREVIEW_SIZE[0] / g.shape[1]
        sy = PREVIEW_SIZE[1] / g.shape[0]
        vis = cv2.cvtColor(cv2.resize(g, PREVIEW_SIZE, interpolation=cv2.INTER_AREA),
                           cv2.COLOR_GRAY2BGR)
        for (x, y) in pts:
            cv2.circle(vis, (int(x * sx), int(y * sy)), 7, (0, 255, 0), 1)
        _, buf = cv2.imencode(".jpg", vis,
                              [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
        return base64.b64encode(buf).decode()

    def read(self):
        grays = read_grays(self.cam)
        dets = [detect_centroids(g, c, max_n=N_LEDS)[0]
                for g, c in zip(grays, self.cam_cfg)]
        raw = self.estimate(dets)

        now = time.time()
        dt = (now - self._last_t) if self._last_t is not None else 0.0
        self._last_t = now
        self.ekf.predict(dt)

        if raw is not None:
            if self.ekf.valid:
                raw["accepted"] = self.ekf.update(np.asarray(raw["leds"], float))
            elif raw["ok"]:

                self.ekf.initialize(raw["pos"], np.asarray(raw.pop("R"), float))
                raw["accepted"] = True
            else:
                raw["accepted"] = False
            raw.pop("R", None)                  # not needed on the wire

        est = self.ekf.state()
        est["rpy"] = rpy_deg(np.asarray(est.pop("R"), float))
        out = {"t": now, "dt": dt, "blobs": [len(d) for d in dets],
               "cams": [m.C.tolist() for m in self.models],
               "raw": raw, "est": est if est["valid"] else None}
        if self.preview:
            out["views"] = [self._preview(g, d) for g, d in zip(grays, dets)]
        return out
