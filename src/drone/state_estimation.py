"""Error-state EKF over the drone's pose, driven by the triangulated LEDs.

Why a filter at all: a brief occlusion, or one LED passing in front of another,
makes the stereo pairing pick the wrong correspondence. The resulting three
world points are not a valid LED triangle, and the rigid fit to them can land
anywhere -- including flipped. Fed straight to a controller that is a wild
manoeuvre, so the estimate has to survive the bad frames rather than follow them.

Three things do that here:

  association  the LED labelling is chosen against the filter's *prediction*
               rather than by best fit alone, so a plausible-but-wrong labelling
               is not free to win
  gating       each measurement's Mahalanobis distance is tested against the
               innovation covariance, so a frame that disagrees with the model
               by more than its own uncertainty allows is dropped, not absorbed
  coasting     with no usable measurement the filter propagates on its motion
               model, and gives up (valid -> False) once it has extrapolated for
               longer than `max_coast_s` and the estimate is no longer trustworthy

State is the nominal (p, v, R, omega) with a 12-dimensional error state
[dp, dv, dtheta, domega]; the rotation error is applied on the right,
R_true = R_nom @ Exp(dtheta), which keeps the measurement Jacobian simple and
avoids any Euler-angle singularity. Motion model is constant velocity and
constant body rate -- the drone is not telling us its inputs, so acceleration is
treated as process noise.
"""
from itertools import permutations

import cv2
import numpy as np

N_ERR = 12                      # dp(3) dv(3) dtheta(3) domega(3)
_I3 = np.eye(3)


def skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def exp_so3(w):
    """Rotation vector -> rotation matrix."""
    return cv2.Rodrigues(np.asarray(w, float))[0]


def orthonormalise(R):
    """Pull a matrix back onto SO(3) after repeated small updates."""
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:                    # guard against a reflection
        U[:, -1] *= -1
        R = U @ Vt
    return R


class PoseEKF:
    """Constant-velocity pose filter measuring the three LED world positions."""

    def __init__(self, led_body, cfg=None):
        cfg = cfg or {}
        self.b = np.asarray(led_body, float)
        self.n_leds = len(self.b)
        # Triangulation noise, per LED per axis. The rig's own residuals are the
        # right scale here, not the camera's pixel noise.
        self.meas_std = float(cfg.get("meas_noise_m", 0.004))
        self.accel_pn = float(cfg.get("accel_pn", 0.15))      # m/s^2
        self.alpha_pn = float(cfg.get("ang_accel_pn", 20.0))  # rad/s^2
        # chi-square 0.999 quantile for 3*n_leds degrees of freedom
        self.gate_chi2 = float(cfg.get("gate_chi2", 27.88))
        self.max_coast_s = float(cfg.get("max_coast_s", 0.5))
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.w = np.zeros(3)
        self.P = np.eye(N_ERR)
        self.valid = False
        self.age = 0.0              # seconds since the last accepted update
        self.n_accepted = 0
        self.n_rejected = 0

    def initialize(self, pos, R):
        """Seed from a trusted raw pose; velocity and rate start at zero."""
        self.p = np.asarray(pos, float).copy()
        self.R = orthonormalise(np.asarray(R, float))
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.P = np.diag(np.concatenate([
            np.full(3, 0.01 ** 2),      # position, m
            np.full(3, 0.50 ** 2),      # velocity, m/s -- unknown, stay loose
            np.full(3, 0.10 ** 2),      # attitude, rad
            np.full(3, 1.00 ** 2),      # body rate, rad/s
        ]))
        self.valid = True
        self.age = 0.0

    # -- prediction --------------------------------------------------------
    def predict(self, dt):
        if not self.valid or dt <= 0.0:
            return
        dR = exp_so3(self.w * dt)
        self.p = self.p + self.v * dt
        self.R = self.R @ dR

        F = np.eye(N_ERR)
        F[0:3, 3:6] = dt * _I3
        F[6:9, 6:9] = dR.T                      # right-error convention
        F[6:9, 9:12] = dt * _I3
        self.P = F @ self.P @ F.T + self._Q(dt)

        self.age += dt
        if self.age > self.max_coast_s:
            self.valid = False                  # extrapolated too far to trust

    def _Q(self, dt):
        """White-acceleration process noise, discretised over dt."""
        Q = np.zeros((N_ERR, N_ERR))
        for base, pn in ((0, self.accel_pn), (6, self.alpha_pn)):
            s = pn ** 2
            Q[base:base + 3, base:base + 3] = s * dt ** 3 / 3.0 * _I3
            Q[base:base + 3, base + 3:base + 6] = s * dt ** 2 / 2.0 * _I3
            Q[base + 3:base + 6, base:base + 3] = s * dt ** 2 / 2.0 * _I3
            Q[base + 3:base + 6, base + 3:base + 6] = s * dt * _I3
        return Q

    # -- measurement -------------------------------------------------------
    def associate(self, world_pts):
        """Order the unlabelled world points to match led_body, via prediction.

        The raw fit picks the labelling with the smallest residual, which on a
        near-isoceles LED triangle can prefer the wrong one. Here the prediction
        breaks the tie instead.
        """
        pred = (self.R @ self.b.T).T + self.p
        best = None
        for perm in permutations(range(self.n_leds)):
            d = float(np.sum((world_pts[list(perm)] - pred) ** 2))
            if best is None or d < best[1]:
                best = (list(perm), d)
        return best[0]

    def update(self, world_pts):
        """Fold in three unlabelled LED world points. True if accepted."""
        if not self.valid:
            return False
        world_pts = np.asarray(world_pts, float)
        if len(world_pts) != self.n_leds:
            return False

        z = world_pts[self.associate(world_pts)].reshape(-1)
        h = ((self.R @ self.b.T).T + self.p).reshape(-1)
        y = z - h

        n = 3 * self.n_leds
        H = np.zeros((n, N_ERR))
        for k in range(self.n_leds):
            H[3 * k:3 * k + 3, 0:3] = _I3
            H[3 * k:3 * k + 3, 6:9] = -self.R @ skew(self.b[k])
        Rm = (self.meas_std ** 2) * np.eye(n)

        S = H @ self.P @ H.T + Rm
        md2 = float(y @ np.linalg.solve(S, y))
        if md2 > self.gate_chi2:                # disagrees beyond its own noise
            self.n_rejected += 1
            return False

        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.R = orthonormalise(self.R @ exp_so3(dx[6:9]))
        self.w = self.w + dx[9:12]

        A = np.eye(N_ERR) - K @ H               # Joseph form, stays symmetric
        self.P = A @ self.P @ A.T + K @ Rm @ K.T
        self.age = 0.0
        self.n_accepted += 1
        return True

    # -- output ------------------------------------------------------------
    def state(self):
        return {
            "pos": self.p.tolist(),
            "vel": self.v.tolist(),
            "R": self.R.tolist(),
            "omega": self.w.tolist(),
            "valid": bool(self.valid),
            "age": float(self.age),
            "coasting": bool(self.valid and self.age > 1e-3),
            "pos_std": np.sqrt(np.diag(self.P)[0:3]).tolist(),
            "accepted": int(self.n_accepted),
            "rejected": int(self.n_rejected),
        }
