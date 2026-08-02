"""LQR hover control law: mocap state -> ANGLE-mode stick commands.

Betaflight closes the attitude loop; this closes position and velocity around
it. Each horizontal axis is modelled independently as

    d/dt [ integral, position, velocity, acceleration ]
        = [ position, velocity, acceleration, (u - acceleration)/tau ]

so the state carries the fact that a commanded lean does not arrive instantly:
the fourth state is the acceleration actually being delivered, chasing the
commanded one through the attitude loop's lag. Ignoring that lag is what makes
naive position loops on a multirotor oscillate -- the controller keeps piling on
correction for a lean that is already on its way. Transport delay is folded into
the same time constant, which is coarse but errs towards damping.

Vertical is the same structure, driven by throttle instead of lean.

Weights come from Bryson's rule: each state is divided by the largest value it
should reach in normal operation, so the numbers in drone.json stay physical
("I do not want to be more than pos_max_m off station") rather than abstract
gains. `effort` then scales the control penalty -- larger is gentler.

Gains are solved once at construction by iterating the discrete Riccati
equation, so there is no SciPy dependency and nothing to tune at runtime.
"""
import numpy as np

G = 9.80665


# ---------------------------------------------------------------------------
# small linear-algebra helpers (numpy only)
# ---------------------------------------------------------------------------
def expm(M, terms=18):
    """Matrix exponential by scaling and squaring with a Taylor series."""
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
    """Zero-order-hold discretisation via the block-matrix exponential."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * dt
    M[:n, n:] = B * dt
    E = expm(M)
    return E[:n, :n], E[:n, n:]


def dlqr(A, B, Q, R, iters=10000, tol=1e-14):
    """Discrete-time LQR gain, by iterating the Riccati difference equation.

    Returns (K, P) with u = -K x. Iteration rather than a direct solver keeps
    this free of SciPy; the systems here are 4x4 and converge in a few hundred
    steps.
    """
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
    """[integral, pos, vel, accel] driven by a lagged acceleration command."""
    A = np.array([[0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, -1.0 / tau]])
    B = np.array([[0.0], [0.0], [0.0], [1.0 / tau]])
    return discretize(A, B, dt)


def _axis_gain(tau, dt, int_max, pos_max, vel_max, acc_max, effort):
    Ad, Bd = axis_system(tau, dt)
    Q = np.diag([1.0 / int_max ** 2, 1.0 / pos_max ** 2,
                 1.0 / vel_max ** 2, 1.0 / acc_max ** 2])
    R = np.array([[(effort ** 2) / acc_max ** 2]])
    K, _ = dlqr(Ad, Bd, Q, R)
    return K.ravel(), Ad, Bd


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
class LQRHover:
    """Position/velocity hold. Gains are solved for the `dt` given here.

    Returns (roll_deg, pitch_deg, yaw_norm, throttle) for the ESP32 bridge.
    """

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

        # Lag each loop sees: attitude (or throttle) response plus transport
        # delay, lumped into one time constant.
        lat = float(q["latency_s"])
        self.tau_xy = float(q["tau_att_s"]) + lat
        self.tau_z = float(q["tau_thr_s"]) + lat

        # Authority limits set both the command clamp and the Bryson weight.
        self.acc_max_xy = G * np.tan(np.radians(self.max_tilt))
        self.thr_to_acc = G / self.hover_ff          # accel per unit throttle
        self.acc_max_z = self.thr_to_acc * self.thr_max

        self.Kxy, self.Ad_xy, self.Bd_xy = _axis_gain(
            self.tau_xy, dt, q["int_max_ms"], q["pos_max_m"], q["vel_max_ms"],
            self.acc_max_xy, q["effort"])
        self.Kz, self.Ad_z, self.Bd_z = _axis_gain(
            self.tau_z, dt, q["z_int_max_ms"], q["z_pos_max_m"],
            q["z_vel_max_ms"], self.acc_max_z, q["z_effort"])

        # How far the integrator may wind is a question about trim authority,
        # not about the cost weight, so it comes from trim_authority_* rather
        # than from int_max_ms (which is only the Bryson normaliser). Sized in
        # acceleration and divided by the integral gain, this is exactly the
        # standing error the loop can null: mostly a hover_ff that is off, since
        # that number is a guess until it is flown and measured.
        trim_acc_xy = min(G * np.tan(np.radians(q["trim_authority_deg"])),
                          self.acc_max_xy)
        trim_acc_z = min(q["trim_authority_thr"] * self.thr_to_acc,
                         self.acc_max_z)
        self.i_xy_lim = float(trim_acc_xy / self.Kxy[0])
        self.i_z_lim = float(trim_acc_z / self.Kz[0])
        self.trim_thr = float(trim_acc_z / self.thr_to_acc)

        # hover_ff is the one number that is genuinely hard to measure by hand,
        # so the loop learns it: whatever standing throttle the integrator ends
        # up holding IS the feed-forward error, and moving it across leaves the
        # instantaneous command untouched while the model gets better. It also
        # tracks the battery sagging, which raises the hover throttle over a
        # flight. thr_to_acc is deliberately NOT recomputed -- the loop gains
        # were solved for it, and it must not shift underneath them.
        self.trim_tau = float(q.get("trim_tau_s", 3.0))
        self.ff_lo, self.ff_hi = 0.05, self.thr_cap - 0.02
        self.reset()

    def _learn_hover_ff(self, dt):
        trim = -self.Kz[0] * self.i[2] / self.thr_to_acc      # throttle units
        new_ff = float(np.clip(self.hover_ff + trim * (dt / self.trim_tau),
                               self.ff_lo, self.ff_hi))
        moved = new_ff - self.hover_ff
        self.hover_ff = new_ff
        self.i[2] += moved * self.thr_to_acc / self.Kz[0]     # keep thr the same

    def reset(self):
        self.i = np.zeros(3)        # integral of position error, world axes
        self.a = np.zeros(3)        # internal model of delivered acceleration

    def __call__(self, pos, vel, yaw, setpoint, yaw_sp, dt=None, integrate=True):
        dt = self.dt if dt is None else dt
        pos = np.asarray(pos, float)
        vel = np.asarray(vel, float)
        err = pos - np.asarray(setpoint, float)

        if integrate:
            lim = np.array([self.i_xy_lim, self.i_xy_lim, self.i_z_lim])
            self.i = np.clip(self.i + err * dt, -lim, lim)
            self._learn_hover_ff(dt)

        # desired world accelerations, u = -K [integral, pos, vel, accel]
        acc = np.array([
            -float(self.Kxy @ [self.i[0], err[0], vel[0], self.a[0]]),
            -float(self.Kxy @ [self.i[1], err[1], vel[1], self.a[1]]),
            -float(self.Kz @ [self.i[2], err[2], vel[2], self.a[2]]),
        ])
        lim = np.array([self.acc_max_xy, self.acc_max_xy, self.acc_max_z])
        acc = np.clip(acc, -lim, lim)

        # advance the internal lag model with what we are about to command
        tau = np.array([self.tau_xy, self.tau_xy, self.tau_z])
        self.a = self.a + (acc - self.a) * (dt / tau)

        # world -> body (yaw only; body X forward, Y left)
        c, s = np.cos(yaw), np.sin(yaw)
        a_fwd = acc[0] * c + acc[1] * s
        a_left = -acc[0] * s + acc[1] * c

        # lean that delivers that acceleration
        pitch = float(np.clip(np.degrees(np.arctan2(self.s_pitch * a_fwd, G)),
                              -self.max_tilt, self.max_tilt))
        roll = float(np.clip(np.degrees(np.arctan2(self.s_roll * -a_left, G)),
                             -self.max_tilt, self.max_tilt))

        # throttle: hover feed-forward plus vertical demand, divided by the
        # cosine of the lean so tilting does not quietly cost altitude
        tilt = np.radians(np.hypot(roll, pitch))
        thr = self.hover_ff + acc[2] / self.thr_to_acc
        thr /= max(np.cos(tilt), 0.5)
        thr = float(np.clip(thr, self.hover_ff - self.thr_max,
                            self.hover_ff + self.thr_max))
        thr = float(np.clip(thr, 0.0, self.thr_cap))

        # Yaw is left to Betaflight. Closing heading from here means a rate
        # command sent through the whole vision -> EKF -> wifi -> FC path, and
        # the loop gain is the stick scaled by the FC's own yaw-rate setting,
        # which nothing on this side knows. With that gain and ~100 ms of round
        # trip the heading loop oscillates. A zero yaw stick is not "no
        # control": Betaflight holds heading on its gyro, which has none of
        # that delay and does it far better. Where the drone points does not
        # affect station keeping either way, because the demand above is
        # rotated into the body frame by the *measured* yaw.
        return roll, pitch, 0.0, thr

    def closed_loop_poles(self):
        """Discrete closed-loop eigenvalues per axis; |lambda| < 1 is stable."""
        return (np.linalg.eigvals(self.Ad_xy - self.Bd_xy @ self.Kxy[None, :]),
                np.linalg.eigvals(self.Ad_z - self.Bd_z @ self.Kz[None, :]))
