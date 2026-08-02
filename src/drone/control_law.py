import numpy as np

G = 9.80665

def expm(M, terms=18):
    M = np.asarray(M, float)
    norm = np.abs(M).sum(axis=1).max()
    s = max(0, int(np.ceil(np.log2(norm))) + 1) if norm > 0 else 0
    Ms = M / (2.0 ** s)
    E = np.eye(len(M))
    T = np.eye(len(M))
    for k in range(1, terms + 1):
        T = T @ Ms / k
        E = E + T
    for _ in range(s):
        E = E @ E
    return E


def discretize(A, B, dt):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * dt
    M[:n, n:] = B * dt
    E = expm(M)
    return E[:n, :n], E[:n, n:]


def dlqr(A, B, Q, R, iters=10000, tol=1e-14):
    P = np.asarray(Q, float).copy()
    K = None
    for _ in range(iters):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P_next = A.T @ P @ (A - B @ K) + Q
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    return K, P


def axis_system(tau, dt):
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, -1 / tau]])
    B = np.array([[0], [0], [0], [1 / tau]])
    return discretize(A, B, dt)


def _axis_gain(tau, dt, int_max, pos_max, vel_max, acc_max, effort):
    Ad, Bd = axis_system(tau, dt)
    Q = np.diag([1 / int_max ** 2, 1 / pos_max ** 2,
                 1 / vel_max ** 2, 1 / acc_max ** 2])
    R = np.array([[(effort ** 2) / acc_max ** 2]])
    K, _ = dlqr(Ad, Bd, Q, R)
    return K.ravel(), Ad, Bd


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LQRHover:
    def __init__(self, drone_cfg, dt):
        g = drone_cfg["gains"]
        q = g["lqr"]
        self.dt = float(dt)

        self.hover_ff = float(g["hover_ff"])
        self.max_tilt = float(g["max_tilt_deg"])
        self.thr_cap = float(g["throttle_cap"])
        self.thr_max = float(q["thr_max"])
        self.kp_yaw = float(g["kp_yaw"])
        self.s_roll = float(g.get("sign_roll", 1.0))
        self.s_pitch = float(g.get("sign_pitch", 1.0))
        self.s_yaw = float(g.get("sign_yaw", 1.0))

        lat = float(q["latency_s"])
        self.tau_xy = float(q["tau_att_s"]) + lat
        self.tau_z = float(q["tau_thr_s"]) + lat

        self.acc_max_xy = G * np.tan(np.radians(self.max_tilt))
        self.thr_to_acc = G / self.hover_ff          # accel per unit throttle
        self.acc_max_z = self.thr_to_acc * self.thr_max

        self.Kxy, self.Ad_xy, self.Bd_xy = _axis_gain(
            self.tau_xy, dt, q["int_max_ms"], q["pos_max_m"], q["vel_max_ms"],
            self.acc_max_xy, q["effort"])
        self.Kz, self.Ad_z, self.Bd_z = _axis_gain(
            self.tau_z, dt, q["z_int_max_ms"], q["z_pos_max_m"],
            q["z_vel_max_ms"], self.acc_max_z, q["z_effort"])

        trim_acc_xy = min(G * np.tan(np.radians(q["trim_authority_deg"])),
                          self.acc_max_xy)
        trim_acc_z = min(q["trim_authority_thr"] * self.thr_to_acc,
                         self.acc_max_z)
        self.i_xy_lim = float(trim_acc_xy / self.Kxy[0])
        self.i_z_lim = float(trim_acc_z / self.Kz[0])
        self.trim_thr = float(trim_acc_z / self.thr_to_acc)

        self.trim_tau = float(q.get("trim_tau_s", 3))
        self.ff_lo, self.ff_hi = 0.05, self.thr_cap - 0.02
        self.reset()

    def _learn_hover_ff(self, dt):
        trim = -self.Kz[0] * self.i[2] / self.thr_to_acc
        new_ff = float(np.clip(self.hover_ff + trim * (dt / self.trim_tau),
                               self.ff_lo, self.ff_hi))
        moved = new_ff - self.hover_ff
        self.hover_ff = new_ff
        self.i[2] += moved * self.thr_to_acc / self.Kz[0]

    def reset(self):
        self.i = np.zeros(3)
        self.a = np.zeros(3)

    def __call__(self, pos, vel, yaw, setpoint, yaw_sp, dt=None, integrate=True):
        dt = self.dt if dt is None else dt
        pos = np.asarray(pos, float)
        vel = np.asarray(vel, float)
        err = pos - np.asarray(setpoint, float)

        if integrate:
            lim = np.array([self.i_xy_lim, self.i_xy_lim, self.i_z_lim])
            self.i = np.clip(self.i + err * dt, -lim, lim)
            self._learn_hover_ff(dt)

        acc = np.array([
            -float(self.Kxy @ [self.i[0], err[0], vel[0], self.a[0]]),
            -float(self.Kxy @ [self.i[1], err[1], vel[1], self.a[1]]),
            -float(self.Kz @ [self.i[2], err[2], vel[2], self.a[2]]),
        ])
        lim = np.array([self.acc_max_xy, self.acc_max_xy, self.acc_max_z])
        acc = np.clip(acc, -lim, lim)

        tau = np.array([self.tau_xy, self.tau_xy, self.tau_z])
        self.a = self.a + (acc - self.a) * (dt / tau)

        c, s = np.cos(yaw), np.sin(yaw)
        a_fwd = acc[0] * c + acc[1] * s
        a_left = -acc[0] * s + acc[1] * c

        pitch = float(np.clip(np.degrees(np.arctan2(self.s_pitch * a_fwd, G)),
                              -self.max_tilt, self.max_tilt))
        roll = float(np.clip(np.degrees(np.arctan2(self.s_roll * -a_left, G)),
                             -self.max_tilt, self.max_tilt))

        tilt = np.radians(np.hypot(roll, pitch))
        thr = self.hover_ff + acc[2] / self.thr_to_acc
        thr /= max(np.cos(tilt), 0.5)
        thr = float(np.clip(thr, self.hover_ff - self.thr_max,
                            self.hover_ff + self.thr_max))
        thr = float(np.clip(thr, 0, self.thr_cap))

        return roll, pitch, 0, thr

    def closed_loop_poles(self):
        return (np.linalg.eigvals(self.Ad_xy - self.Bd_xy @ self.Kxy[None, :]),
                np.linalg.eigvals(self.Ad_z - self.Bd_z @ self.Kz[None, :]))
