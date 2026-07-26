"""
hover.py -- autonomous IR-mocap hover for the 2x PS3-Eye rig + ESP32/MSP drone.

Pipeline:  2 IR cameras -> blob detection -> stereo triangulation -> rigid-body
pose (Kabsch) -> filter -> cascaded ANGLE-mode controller -> UDP CmdPacket to the
ESP32 (same wire format as teleop.py). Betaflight does the inner-loop stabilising.

MODES (python hover.py --mode ...):
  selftest   pure-math synthetic round trip. No hardware. Run this first.
  track      live cameras + pose estimate on screen, NO commands sent. Bring-up.
  fly        full closed loop. Sends commands. Keep your TX as the hardware kill.

  --write-config   (re)write default JSON configs into ./config and exit.
  --config DIR     config directory (default: ./config)
  --view           in fly mode, also open the camera windows

SAFETY: this commands a real aircraft. Bring up in order -- selftest, then track
(verify the 3D point sits where the drone is), then fly with props OFF to watch
the commanded roll/pitch/throttle react correctly, then tethered, then free.
The autonomous loop starts DISARMED; you must arm (z) and then engage (e), and
engage is refused unless tracking is valid. Keep the transmitter bound as an
independent kill path -- this software is not a substitute for it.

Depends: numpy (always); opencv-python, pseyepy, pygame (track/fly only, imported
lazily so selftest/write-config run with just numpy).
"""
import argparse
import json
import math
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

import mocap_core as mc

# ---------------------------------------------------------------------------
# Wire formats -- must match main.cpp / teleop.py exactly
# ---------------------------------------------------------------------------
CMD_FMT = "<IBffffB"           # seq, flags, roll, pitch, yaw, throttle, angleMode
TELEM_FMT = "<IBffffHI"        # seq, state, roll, pitch, yaw, throttle, loopMax, armFlags
TELEM_SIZE = struct.calcsize(TELEM_FMT)
FLAG_ARM = 0x01
CMD_PORT = 9000
TELEM_PORT = 9001

ARM_FLAG_NAMES = [
    "NO_GYRO", "FAILSAFE", "RX_FAILSAFE", "BAD_RX_RECOVERY", "BOXFAILSAFE",
    "RUNAWAY_TAKEOFF", "CRASH_DETECTED", "THROTTLE", "ANGLE", "BOOT_GRACE_TIME",
    "NOPREARM", "LOAD", "CALIBRATING", "CLI", "CMS_MENU", "BST", "MSP",
    "PARALYZE", "GPS", "RESC", "RPMFILTER", "REBOOT_REQUIRED", "DSHOT_BITBANG",
    "ACC_CALIB", "MOTOR_PROTO", "ARM_SWITCH",
]


def decode_arm_flags(flags):
    if flags == 0:
        return "READY"
    if flags == 0xFFFFFFFF:
        return "(no status yet)"
    names = [ARM_FLAG_NAMES[i] if i < len(ARM_FLAG_NAMES) else "BIT%d" % i
             for i in range(32) if flags & (1 << i)]
    return ", ".join(names) if names else "0x%08X" % flags


# ---------------------------------------------------------------------------
# Default configuration  (--write-config emits these as JSON)
# ---------------------------------------------------------------------------
# Nominal PS3-Eye pinhole at 640x480, ~56 deg horizontal FOV:
#   fx = (640/2) / tan(56/2 deg) ~= 601.  cx,cy = image centre.
# These are NOMINAL -- replace fx,fy,cx,cy with your own intrinsics calibration
# when you run it (you said you have a good method). Distortion is ignored here.
_FX = _FY = 601.0
_CX, _CY = 320.0, 240.0

# Rig geometry (metres, WORLD frame: X=cam0->cam1, Y=into volume, Z=up, origin on
# the floor beneath the hover point). ALL PLACEHOLDERS -- measure and replace.
_BASELINE = 1.0       # horizontal separation between the two cameras   (a)
_HEIGHT = 2.0         # camera height above the floor                   (z)
_STANDOFF = 1.5       # how far back (-Y) the cameras sit from the hover column
_HOVER = [0.0, 0.0, 1.0]   # look target == desired hover point

DEFAULT_CAMERA = {
    "index": 0,
    "resolution": "large",           # "large"=640x480, "small"=320x240
    "fps": 60,
    # --- detection (from your IR tuner; press 'p' there and paste) ---
    "exposure": 5, "gain": 10, "thresh": 40,
    "min_area": 2, "max_area": 300, "min_circ": 0.60, "blur_ksize": 5,
    # --- intrinsics (nominal; replace with your calibration) ---
    "fx": _FX, "fy": _FY, "cx": _CX, "cy": _CY,
    # --- extrinsics: camera centre + look-at target in WORLD metres ---
    "position": [-_BASELINE / 2, -_STANDOFF, _HEIGHT],
    "look_at": _HOVER,
    "up": [0.0, 0.0, 1.0],
}

DEFAULT_DRONE = {
    # 3 IR LEDs, positions in BODY frame (metres, from CoM). FLU: X fwd, Y left,
    # Z up. Make them a DISTINCTLY SCALENE, non-collinear triangle so the labels
    # are unambiguous -- measure yours with calipers and replace these.
    "led_body": [
        [0.150, 0.000, 0.000],
        [-0.150, 0.000, 0.000],
        [-0.043, 0.105, 0.020],
    ],
    "mass_kg": 0.5,
    "setpoint": [0.0, 0.0, 1.0],     # world hover target; z is the 1 m goal
    "gains": {
        "kp_xy": 6.0, "kd_xy": 3.5, "ki_xy": 1.2,     # deg-tilt per m / per (m/s) / per m*s
        "kp_z": 0.6, "kd_z": 0.25, "ki_z": 0.4,       # throttle per m / per (m/s) / per m*s
        "kp_yaw": 1.5,                                 # yaw-cmd per rad
        "hover_ff": 0.42,       # feed-forward throttle that ~holds weight (MEASURE)
        "max_tilt_deg": 12.0,   # outer-loop tilt clamp (well under the 25 deg link clamp)
        "throttle_cap": 0.60,   # hard ceiling for early tests -- raise deliberately
        "i_xy_limit": 3.0, "i_z_limit": 0.15,
        # sign conventions are airframe/BF specific -- VERIFY each at low authority
        "sign_roll": 1.0, "sign_pitch": 1.0, "sign_yaw": 1.0,
    },
    "filter": {"alpha": 0.5, "beta": 0.05, "gate_m": 0.30},
    "control": {
        "climb_rate": 0.4,      # m/s the z setpoint ramps toward 1 m on engage
        "loss_grace_s": 0.25,   # tracking may drop this long before we react
        "loss_ramp_s": 1.0,     # then throttle ramps to 0 over this, and disarms
        "reject_rmsd_m": 0.015, # Kabsch fit worse than this -> frame rejected
        "send_hz": 100,
    },
    "esp_ip": "192.168.4.1",
}


def write_default_config(cfgdir: Path):
    cfgdir.mkdir(parents=True, exist_ok=True)
    cam0 = dict(DEFAULT_CAMERA, index=0,
                position=[-_BASELINE / 2, -_STANDOFF, _HEIGHT])
    cam1 = dict(DEFAULT_CAMERA, index=1,
                position=[+_BASELINE / 2, -_STANDOFF, _HEIGHT])
    for name, data in [("camera0.json", cam0), ("camera1.json", cam1),
                       ("drone.json", DEFAULT_DRONE)]:
        (cfgdir / name).write_text(json.dumps(data, indent=2))
        print("wrote", cfgdir / name)


# ---------------------------------------------------------------------------
# Config loading -> camera models + controller pieces
# ---------------------------------------------------------------------------
class Rig:
    def __init__(self, cfgdir: Path):
        self.cam_cfg = [json.loads((cfgdir / f"camera{i}.json").read_text())
                        for i in (0, 1)]
        self.drone = json.loads((cfgdir / "drone.json").read_text())
        self.models = []
        for c in self.cam_cfg:
            K = mc.K_from_intrinsics(c["fx"], c["fy"], c["cx"], c["cy"])
            ex = c.get("extrinsics", c)          # allow nested or flat
            if "rvec" in ex and "tvec" in ex:    # solvePnP result (full pose incl roll)
                self.models.append(mc.CameraModel.from_rvec_tvec(K, ex["rvec"], ex["tvec"]))
            else:                                 # manual position + aim (level cameras)
                self.models.append(mc.CameraModel.from_look_at(
                    K, c["position"], c["look_at"], c.get("up", [0, 0, 1])))
        self.led_body = np.asarray(self.drone["led_body"], float)
        self._check_constellation()

    def _check_constellation(self):
        b = self.led_body
        d = sorted([np.linalg.norm(b[i] - b[j]) for i, j in ((0, 1), (0, 2), (1, 2))])
        spread = (d[1] - d[0], d[2] - d[1])
        if min(spread) < 0.02:
            print(f"WARNING: LED edge lengths {[round(float(x),3) for x in d]} m are close; "
                  "labelling may be ambiguous. Make the triangle more scalene.")

    def estimate(self, pts0, pts1):
        """pts0/pts1: pixel centroids (each a list of (x,y)). Needs >=3 in each.
        Returns dict or None."""
        if len(pts0) < 3 or len(pts1) < 3:
            return None
        Xs, perm, rerr = mc.match_stereo(self.models[0], self.models[1],
                                         pts0[:3], pts1[:3])
        R, p, rmsd, order = mc.identify_and_pose(Xs, self.led_body)
        return {"pos": p, "R": R, "yaw": mc.yaw_from_R(R),
                "rmsd": rmsd, "reproj": rerr, "world_leds": Xs}


# ---------------------------------------------------------------------------
# Detection (OpenCV) -- returns brightest valid IR centroids, sub-pixel
# ---------------------------------------------------------------------------
def detect_centroids(gray, cfg, max_n=3):
    import cv2
    k = cfg["blur_ksize"]
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    _, mask = cv2.threshold(blur, cfg["thresh"], 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg["min_area"] or area > cfg["max_area"]:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        if 4.0 * np.pi * area / (perim * perim) < cfg["min_circ"]:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(c)
        cmask = np.zeros((h, w), np.uint8)
        cv2.drawContours(cmask, [c - [x, y]], -1, 255, -1)
        inten = cv2.mean(gray[y:y + h, x:x + w], mask=cmask)[0]
        out.append((cx, cy, inten))
    out.sort(key=lambda t: -t[2])                 # brightest first
    return [(cx, cy) for cx, cy, _ in out[:max_n]], mask


# ---------------------------------------------------------------------------
# Camera capture (pseyepy)
# ---------------------------------------------------------------------------
def open_cameras(rig: Rig):
    from pseyepy import Camera
    import numpy as _np
    res_map = {"large": Camera.RES_LARGE, "small": Camera.RES_SMALL}
    res = res_map[rig.cam_cfg[0]["resolution"]]
    fps = rig.cam_cfg[0]["fps"]
    ids = [c["index"] for c in rig.cam_cfg]
    cam = Camera(ids, fps=fps, resolution=res, colour=False)

    def _set_list(name, idx, value):
        try:
            cur = getattr(cam, name)
            if isinstance(cur, (list, tuple, _np.ndarray)):
                cur = list(cur)
                while len(cur) <= idx:
                    cur.append(value)
                cur[idx] = value
                setattr(cam, name, cur)
            else:
                setattr(cam, name, value)
        except Exception:
            pass

    for i, c in enumerate(rig.cam_cfg):
        _set_list("exposure", i, c["exposure"])
        _set_list("gain", i, c["gain"])
    return cam


def read_grays(cam):
    import cv2
    frames, _ = cam.read()
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    grays = []
    for f in frames:
        f = np.asarray(f)
        if f.ndim == 3:
            f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        if f.dtype != np.uint8:
            f = cv2.convertScaleAbs(f)
        grays.append(f)
    return grays


# ---------------------------------------------------------------------------
# UDP link to the ESP32
# ---------------------------------------------------------------------------
class Link:
    def __init__(self, esp_ip):
        self.esp = (esp_ip, CMD_PORT)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", TELEM_PORT))
        self.rx.setblocking(False)
        self.seq = 0
        self.telem = None

    def send(self, arm, roll, pitch, yaw, throttle, angle_mode=1):
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        flags = FLAG_ARM if arm else 0
        self.tx.sendto(struct.pack(CMD_FMT, self.seq, flags,
                                   float(roll), float(pitch), float(yaw),
                                   float(throttle), angle_mode), self.esp)

    def poll(self):
        try:
            while True:
                data, _ = self.rx.recvfrom(64)
                if len(data) >= TELEM_SIZE:
                    self.telem = struct.unpack(TELEM_FMT, data[:TELEM_SIZE])
        except (BlockingIOError, OSError):
            pass
        return self.telem

    def disarm_burst(self, n=10):
        for _ in range(n):
            self.send(False, 0, 0, 0, 0)
            time.sleep(0.005)


# ---------------------------------------------------------------------------
# MODE: selftest -- synthetic round trip (numpy only)
# ---------------------------------------------------------------------------
def mode_selftest():
    rng = np.random.default_rng(0)

    def euler(r, p, y):
        cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p),
                                  np.cos(y), np.sin(y))
        return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @
                np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @
                np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))

    K = mc.K_from_intrinsics(_FX, _FY, _CX, _CY)
    c0 = mc.CameraModel.from_look_at(K, [-_BASELINE/2, -_STANDOFF, _HEIGHT], _HOVER)
    c1 = mc.CameraModel.from_look_at(K, [+_BASELINE/2, -_STANDOFF, _HEIGHT], _HOVER)
    led = np.array(DEFAULT_DRONE["led_body"])
    pos_e, yaw_e, rej = [], [], 0
    N = 3000
    for _ in range(N):
        pt = np.array([rng.uniform(-.4, .4), rng.uniform(-.4, .4), rng.uniform(.6, 1.4)])
        Rt = euler(rng.uniform(-.25, .25), rng.uniform(-.25, .25), rng.uniform(-np.pi, np.pi))
        yaw = math.atan2(Rt[1, 0], Rt[0, 0])
        world = (Rt @ led.T).T + pt
        uv0 = np.array([c0.project(w) for w in world]) + rng.normal(0, 0.3, (3, 2))
        uv1 = np.array([c1.project(w) for w in world]) + rng.normal(0, 0.3, (3, 2))
        uv1 = uv1[rng.permutation(3)]
        Xs, _, _ = mc.match_stereo(c0, c1, uv0, uv1)
        R, p, rmsd, _ = mc.identify_and_pose(Xs, led)
        if rmsd > 0.015:
            rej += 1
            continue
        pos_e.append(np.linalg.norm(p - pt))
        yaw_e.append(abs(mc.wrap_pi(mc.yaw_from_R(R) - yaw)))
    pos_e = np.array(pos_e) * 1000
    yaw_e = np.degrees(yaw_e)
    print(f"selftest: {N} synthetic frames @0.3px noise, default rig+constellation")
    print(f"  position err mm  p50={np.percentile(pos_e,50):.2f} "
          f"p99={np.percentile(pos_e,99):.2f} max={pos_e.max():.2f}")
    print(f"  yaw err deg      p50={np.percentile(yaw_e,50):.3f} "
          f"p99={np.percentile(yaw_e,99):.3f} max={yaw_e.max():.2f}")
    print(f"  frames rejected by rmsd gate: {rej}/{N}")
    ok = np.percentile(pos_e, 99) < 15 and np.percentile(yaw_e, 99) < 3
    print("  RESULT:", "OK" if ok else "CHECK GEOMETRY")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# MODE: track -- live pose estimate, no commands
# ---------------------------------------------------------------------------
def mode_track(rig: Rig):
    import cv2
    cam = open_cameras(rig)
    print("track mode: move the LED rig in the shared view. q to quit.")
    filt = mc.StateFilter(**rig.drone["filter"])
    last = time.perf_counter()
    try:
        while True:
            grays = read_grays(cam)
            now = time.perf_counter(); dt = now - last; last = now
            dets, masks = [], []
            for i, g in enumerate(grays):
                pts, mask = detect_centroids(g, rig.cam_cfg[i])
                dets.append(pts); masks.append(mask)
            est = rig.estimate(dets[0], dets[1]) if len(dets) == 2 else None
            filt.predict(dt)
            if est and est["rmsd"] <= rig.drone["control"]["reject_rmsd_m"]:
                filt.update(est["pos"], est["yaw"], dt)

            vis = []
            for i, g in enumerate(grays):
                d = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                for (x, y) in dets[i]:
                    cv2.circle(d, (int(x), int(y)), 8, (0, 255, 0), 1)
                cv2.putText(d, f"cam{i} n={len(dets[i])}", (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                vis.append(d)
            montage = np.hstack(vis) if len(vis) == 2 else vis[0]
            if est:
                p = filt.x if filt.x is not None else est["pos"]
                txt = (f"pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})m "
                       f"yaw={np.degrees(filt.yaw):+.0f} "
                       f"rmsd={est['rmsd']*1000:.1f}mm reproj={est['reproj']:.2f}px")
                color = (0, 255, 0) if est["rmsd"] <= rig.drone["control"]["reject_rmsd_m"] else (0, 0, 255)
            else:
                txt = "no pose (need 3 blobs in BOTH cameras)"
                color = (0, 0, 255)
            cv2.putText(montage, txt, (6, montage.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            cv2.imshow("track (q=quit)", montage)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cam.end()
        cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------------------
# MODE: fly -- full closed loop with safety state machine + pygame HUD
# ---------------------------------------------------------------------------
def mode_fly(rig: Rig, view=False):
    import pygame
    import cv2

    d = rig.drone
    ctrl = d["control"]
    controller = mc.HoverController(d["gains"])
    filt = mc.StateFilter(**d["filter"])
    link = Link(d["esp_ip"])
    cam = open_cameras(rig)

    pygame.init()
    screen = pygame.display.set_mode((560, 420))
    pygame.display.set_caption("autonomous hover -- z=arm e=engage space=CUT esc=quit")
    font = pygame.font.SysFont("menlo,consolas,monospace", 17)
    big = pygame.font.SysFont("menlo,consolas,monospace", 30, bold=True)

    ARMED = ENGAGED = False
    prev_z = prev_e = False
    setpoint = np.array(d["setpoint"], float)
    z_sp = setpoint[2]
    yaw_sp = 0.0
    last = time.perf_counter()
    last_valid = last
    loss_t0 = None

    def hud(lines, banner, color):
        screen.fill((22, 22, 26))
        screen.blit(big.render(banner, True, color), (20, 18))
        y = 64
        for ln in lines:
            screen.blit(font.render(ln, True, (215, 215, 215)), (20, y))
            y += font.get_linesize() + 6
        screen.blit(font.render("z arm/disarm   e engage   space CUT   esc quit",
                    True, (130, 130, 140)), (20, 420 - 30))
        pygame.display.flip()

    running = True
    try:
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    running = False
            keys = pygame.key.get_pressed()
            now = time.perf_counter(); dt = now - last; last = now
            dt = min(max(dt, 1e-3), 0.1)

            # --- vision ---
            grays = read_grays(cam)
            dets = [detect_centroids(g, rig.cam_cfg[i])[0] for i, g in enumerate(grays)]
            est = rig.estimate(dets[0], dets[1]) if len(dets) == 2 else None
            filt.predict(dt)
            good = bool(est and est["rmsd"] <= ctrl["reject_rmsd_m"])
            if good:
                filt.update(est["pos"], est["yaw"], dt)
            tracking = filt.valid and (now - last_valid) < 5.0
            if good and filt.valid:
                last_valid = now

            # --- edge-triggered keys ---
            z_now = keys[pygame.K_z]
            if z_now and not prev_z:
                ARMED = not ARMED
                if not ARMED:
                    ENGAGED = False
                controller.reset()
            prev_z = z_now
            e_now = keys[pygame.K_e]
            if e_now and not prev_e:
                if not ENGAGED:
                    if ARMED and tracking:          # engage only with valid track
                        ENGAGED = True
                        setpoint[0], setpoint[1] = filt.x[0], filt.x[1]
                        z_sp = filt.x[2]
                        yaw_sp = filt.yaw
                        controller.reset()
                else:
                    ENGAGED = False
            prev_e = e_now
            if keys[pygame.K_SPACE]:                # panic
                ARMED = ENGAGED = False

            # --- command synthesis ---
            roll = pitch = yaw_cmd = 0.0
            throttle = 0.0
            if ARMED and ENGAGED:
                stale = now - last_valid
                if stale < ctrl["loss_grace_s"] and filt.x is not None:
                    loss_t0 = None
                    z_sp = min(z_sp + ctrl["climb_rate"] * dt, setpoint[2]) \
                        if z_sp < setpoint[2] else \
                        max(z_sp - ctrl["climb_rate"] * dt, setpoint[2])
                    sp = np.array([setpoint[0], setpoint[1], z_sp])
                    roll, pitch, yaw_cmd, throttle = controller(
                        filt.x, filt.v, filt.yaw, sp, yaw_sp, dt, integrate=True)
                else:
                    # tracking lost: level out, ramp throttle down, then disarm
                    if loss_t0 is None:
                        loss_t0 = now
                    frac = (now - loss_t0) / ctrl["loss_ramp_s"]
                    if frac >= 1.0:
                        ARMED = ENGAGED = False
                    else:
                        _, _, _, hold = controller(
                            filt.x if filt.x is not None else np.array([0, 0, z_sp]),
                            filt.v, filt.yaw, np.array([setpoint[0], setpoint[1], z_sp]),
                            yaw_sp, dt, integrate=False)
                        throttle = hold * (1.0 - frac)
                        roll = pitch = yaw_cmd = 0.0

            link.send(ARMED, roll, pitch, yaw_cmd, throttle)
            telem = link.poll()

            # --- HUD ---
            if ENGAGED:
                banner, color = "ENGAGED", (40, 200, 90)
            elif ARMED:
                banner, color = "ARMED (idle)", (230, 180, 40)
            else:
                banner, color = "DISARMED", (150, 150, 150)
            p = filt.x if filt.x is not None else np.zeros(3)
            lines = [
                "tracking : %s%s" % ("VALID" if tracking else "LOST",
                                     "" if good else "  (coasting)"),
                "pos  m   : %+.2f %+.2f %+.2f" % (p[0], p[1], p[2]),
                "yaw  deg : %+.0f" % np.degrees(filt.yaw),
                "sp   m   : %+.2f %+.2f %+.2f" % (setpoint[0], setpoint[1], z_sp),
                "cmd      : roll %+5.1f  pitch %+5.1f  yaw %+.2f" % (roll, pitch, yaw_cmd),
                "throttle : %.2f  (cap %.2f)" % (throttle, controller.thr_cap),
            ]
            if est:
                lines.append("fit rmsd : %.1f mm" % (est["rmsd"] * 1000))
            if telem:
                lines.append("FC state : %d   loop %d us" % (telem[1], telem[6]))
                lines.append("FC arming: %s" % decode_arm_flags(telem[7]))
            else:
                lines.append("FC       : (no telemetry -- check wifi/link)")
            hud(lines, banner, color)

            if view:
                mont = np.hstack([cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) for g in grays]) \
                    if len(grays) == 2 else cv2.cvtColor(grays[0], cv2.COLOR_GRAY2BGR)
                for i, g in enumerate(grays):
                    for (x, y) in dets[i]:
                        cv2.circle(mont, (int(x) + i * grays[0].shape[1], int(y)),
                                   7, (0, 255, 0), 1)
                cv2.imshow("cameras", mont)
                cv2.waitKey(1)
    finally:
        link.disarm_burst()
        cam.end()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        pygame.quit()
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["selftest", "track", "fly"], default="selftest")
    ap.add_argument("--config", default="config", type=Path)
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    if args.write_config:
        write_default_config(args.config)
        return 0
    if args.mode == "selftest":
        return mode_selftest()

    if not (args.config / "camera0.json").exists():
        print(f"no config in {args.config}/ -- run:  python hover.py --write-config")
        return 2
    rig = Rig(args.config)
    if args.mode == "track":
        return mode_track(rig)
    if args.mode == "fly":
        print("FLY MODE. Props off for first bring-up. TX bound as kill path.")
        return mode_fly(rig, view=args.view)


if __name__ == "__main__":
    sys.exit(main())
