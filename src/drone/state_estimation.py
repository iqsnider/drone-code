from itertools import permutations

import cv2
import numpy as np

N_ERR = 12
_I3 = np.eye(3)


def skew(v):
    S = np.array([[0, -v[2], v[1]],
                 [v[2], 0, -v[0]],
                 [-v[1], v[0], 0]])

    return S


def exp_so3(w):
    R = cv2.Rodrigues(np.asarray(w, float))[0]

    return R


def orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


class PoseEKF:

    def __init__(self, led_body, cfg=None):
        cfg = cfg or {}
        self.b = np.asarray(led_body, float)
        self.n_leds = len(self.b)

        self.meas_std = float(cfg.get("meas_noise_m", 0.004))
        self.accel_pn = float(cfg.get("accel_pn", 0.15))      # m/s^2
        self.alpha_pn = float(cfg.get("ang_accel_pn", 20))    # rad/s^2

        self.gate_chi2 = float(cfg.get("gate_chi2", 27.88))
        self.max_coast_s = float(cfg.get("max_coast_s", 0.5))
        self.reset()

    def reset(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.w = np.zeros(3)
        self.P = np.eye(N_ERR)
        self.valid = False
        self.age = 0
        self.n_accepted = 0
        self.n_rejected = 0

    def initialize(self, pos, R):
        self.p = np.asarray(pos, float).copy()
        self.R = orthonormalize(np.asarray(R, float))
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.P = np.diag(np.concatenate([
            np.full(3, 0.01**2),
            np.full(3, 0.5**2),
            np.full(3, 0.1**2),
            np.full(3, 1**2),
        ]))
        self.valid = True
        self.age = 0

    def predict(self, dt):
        if not self.valid or dt <= 0:
            return
        dR = exp_so3(self.w * dt)
        self.p = self.p + self.v * dt
        self.R = self.R @ dR

        F = np.eye(N_ERR)
        F[0:3, 3:6] = dt * _I3
        F[6:9, 6:9] = dR.T
        F[6:9, 9:12] = dt * _I3
        self.P = F @ self.P @ F.T + self._Q(dt)

        self.age += dt
        if self.age > self.max_coast_s:
            self.valid = False

    def _Q(self, dt):
        Q = np.zeros((N_ERR, N_ERR))
        for base, pn in ((0, self.accel_pn), (6, self.alpha_pn)):
            s = pn ** 2
            Q[base:base + 3, base:base + 3] = s * dt**3 / 3 * _I3
            Q[base:base + 3, base + 3:base + 6] = s * dt**2 / 2 * _I3
            Q[base + 3:base + 6, base:base + 3] = s * dt**2 / 2 * _I3
            Q[base + 3:base + 6, base + 3:base + 6] = s * dt * _I3

        return Q

    def associate(self, world_pts):
        pred = (self.R @ self.b.T).T + self.p
        best = None
        for perm in permutations(range(self.n_leds)):
            d = float(np.sum((world_pts[list(perm)] - pred) ** 2))
            if best is None or d < best[1]:
                best = (list(perm), d)
        order = best[0]

        return order

    def update(self, world_pts):
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
        if md2 > self.gate_chi2:
            self.n_rejected += 1
            return False

        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.R = orthonormalize(self.R @ exp_so3(dx[6:9]))
        self.w = self.w + dx[9:12]

        A = np.eye(N_ERR) - K @ H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T
        self.age = 0
        self.n_accepted += 1

        return True

    def state(self):
        s = {"pos": self.p.tolist(),
             "vel": self.v.tolist(),
             "R": self.R.tolist(),
             "omega": self.w.tolist(),
             "valid": bool(self.valid),
             "age": float(self.age),
             "coasting": bool(self.valid and self.age > 1e-3),
             "pos_std": np.sqrt(np.diag(self.P)[0:3]).tolist(),
             "accepted": int(self.n_accepted),
             "rejected": int(self.n_rejected)}

        return s
