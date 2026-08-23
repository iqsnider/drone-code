import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from drone.pose import PoseTracker
from drone.control_law import LQRHover
from drone.link import Link, decode_arm_flags

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config"

def main(
    config: Path = DEFAULT_CONFIG,
    height: Optional[float] = typer.Option(None, "--height", help="override hover height in meters"),
):
    """
    LQR hover: hold the drone at a setpoint using the camera rig for feedback.
    """
    import pygame

    tracker = PoseTracker(config)
    d = tracker.drone
    ctl = d["control"]

    target = np.array(d["setpoint"], float)
    if height is not None:
        target[2] = height
    dt = 1 / float(tracker.cam_cfg[0]["fps"])
    controller = LQRHover(d, dt)
    link = Link(d["esp_ip"])

    tracker.open()
    pygame.init()
    screen = pygame.display.set_mode((620, 470))
    pygame.display.set_caption("LQR hover (z arm, e engage, space CUT, esc quit)")
    font = pygame.font.SysFont("menlo,consolas,monospace", 17)
    big = pygame.font.SysFont("menlo,consolas,monospace", 30, bold=True)

    armed = engaged = False
    prev_z = prev_e = False
    sp = target.copy()          # ramped setpoint, moves toward target
    yaw_sp = 0
    loss_t0 = None
    last_thr = controller.hover_ff      # what the loss ramp eases down from
    last = time.perf_counter()
    cut_reason = ""

    def hud(lines, banner, color):
        screen.fill((22, 22, 26))
        screen.blit(big.render(banner, True, color), (20, 16))
        y = 62
        for ln in lines:
            screen.blit(font.render(ln, True, (215, 215, 215)), (20, y))
            y += font.get_linesize() + 5
        screen.blit(font.render("z arm/disarm   e engage   space CUT   esc quit",
                                True, (130, 130, 140)), (20, 470 - 28))
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

            now = time.perf_counter()
            step = min(max(now - last, 1e-3), 0.1)
            last = now

            out = tracker.read()
            est = out["est"]
            tracking = est is not None and est["age"] < ctl["loss_grace_s"]

            # edge-triggered keys
            z_now = keys[pygame.K_z]
            if z_now and not prev_z:
                armed = not armed
                if not armed:
                    engaged = False
                controller.reset()
            prev_z = z_now

            e_now = keys[pygame.K_e]
            if e_now and not prev_e:
                if engaged:
                    engaged = False
                elif armed and tracking:
                    engaged = True
                    cut_reason = ""
                    sp = np.array(est["pos"], float)     # start where it is
                    yaw_sp = np.radians(est["rpy"][2])   # hold current heading
                    controller.reset()
            prev_e = e_now

            if keys[pygame.K_SPACE]:
                if armed or engaged:
                    cut_reason = "manual CUT"
                armed = engaged = False

            roll = pitch = yaw_cmd = 0
            throttle = 0
            if armed and engaged:
                if tracking:
                    loss_t0 = None
                    pos = np.array(est["pos"], float)
                    # walk the setpoint to the target so engaging never
                    # applies a step the loop has to chase
                    delta = target - sp
                    span = np.linalg.norm(delta)
                    move = ctl["climb_rate"] * step
                    sp = target.copy() if span <= move else sp + delta / span * move
                    roll, pitch, yaw_cmd, throttle = controller(
                        pos, est["vel"], np.radians(est["rpy"][2]),
                        sp, yaw_sp, step, integrate=True)
                    last_thr = throttle
                else:
                    # tracking lost: level the aircraft, ease the throttle off,
                    # then disarm. Better a controlled sink than a blind hover.
                    if loss_t0 is None:
                        loss_t0 = now
                    frac = (now - loss_t0) / ctl["loss_ramp_s"]
                    if frac >= 1:
                        armed = engaged = False
                        cut_reason = "TRACKING LOST"
                    else:
                        # ease off from whatever was last commanded, so losing
                        # tracking mid-climb is not itself a throttle step
                        throttle = last_thr * (1 - frac)

            link.send(armed, roll, pitch, yaw_cmd, throttle)
            telem = link.poll()

            # HUD
            if engaged:
                banner, color = "ENGAGED", (40, 200, 90)
            elif armed:
                banner, color = "ARMED (idle)", (230, 180, 40)
            else:
                banner, color = "DISARMED", (150, 150, 150)

            lines = []
            if est:
                p, v = est["pos"], est["vel"]
                lines += [
                    "track    : %s%s" % ("VALID" if tracking else "LOST",
                                         "  (coasting)" if est["coasting"] else ""),
                    "pos  m   : %+.3f %+.3f %+.3f" % tuple(p),
                    "vel  m/s : %+.3f %+.3f %+.3f" % tuple(v),
                    "rpy  deg : %+.1f %+.1f %+.1f" % tuple(est["rpy"]),
                ]
            else:
                lines.append("track    : NO ESTIMATE  (blobs %s)"
                             % " ".join("%d" % b for b in out["blobs"]))
            lines += [
                "target m : %+.3f %+.3f %+.3f" % tuple(target),
                "setpt  m : %+.3f %+.3f %+.3f" % tuple(sp),
                "cmd      : roll %+5.1f  pitch %+5.1f  yaw %+.2f" % (roll, pitch, yaw_cmd),
                "throttle : %.3f   (hover_ff %.2f, cap %.2f)"
                % (throttle, controller.hover_ff, controller.thr_cap),
            ]
            if cut_reason:
                lines.append("last cut : %s" % cut_reason)
            if telem:
                lines.append("FC       : state %d  loop %d us  loss %.1f%%"
                             % (telem[1], telem[6], telem[8] / 10))
                lines.append("FC arming: %s" % decode_arm_flags(telem[7]))
            else:
                lines.append("FC       : no telemetry, check wifi to the ESP32")
            hud(lines, banner, color)
    finally:
        link.disarm_burst()
        tracker.close()
        pygame.quit()


if __name__ == "__main__":
    typer.run(main)
