"""Measure the hover throttle by ramping up until the drone just leaves the ground."""

import argparse
import json
import shutil
import time
from datetime import date
from pathlib import Path

import numpy as np

from drone.control_law import G
from drone.link import Link, decode_arm_flags
from drone.pose import PoseTracker

RAMP_START = 0.08
RAMP_RATE = 0.03
LIFT_HEIGHT = 0.03
LIFT_SPEED = 0.05
REST_SECONDS = 1
RAMP_DOWN_S = 0.6
TIMEOUT_S = 40


def crossing_throttle(thr_detect, vz, rate):
    hover = float(thr_detect)
    for _ in range(50):
        tau = np.sqrt(max(2*vz*hover / (rate*G), 0))
        hover = float(thr_detect) - rate * tau
        if hover <= 1e-3:
            return float(thr_detect)
    return hover


def apply_hover_ff(cfgdir, value):
    path = Path(cfgdir) / "drone.json"
    shutil.copy2(path, path.with_suffix(".json.bak"))
    d = json.loads(path.read_text())
    old = d["gains"]["hover_ff"]
    d["gains"]["hover_ff"] = round(float(value), 4)
    d.setdefault("_trim_source", {}).update(
        {"hover_ff": round(float(value), 4), "method": "ramp to liftoff",
         "on": str(date.today())})
    path.write_text(json.dumps(d, indent=2) + "\n")
    return old, path


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[2] / "config")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure but do not write drone.json")
    args = ap.parse_args()

    import pygame

    tracker = PoseTracker(args.config)
    d = tracker.drone
    cap = float(d["gains"]["throttle_cap"])
    link = Link(d["esp_ip"])

    tracker.open()
    pygame.init()
    screen = pygame.display.set_mode((620, 400))
    pygame.display.set_caption("hover trim -- z start  space CUT  esc quit")
    font = pygame.font.SysFont("menlo,consolas,monospace", 17)
    big = pygame.font.SysFont("menlo,consolas,monospace", 30, bold=True)

    armed = False
    prev_z = False
    thr = 0
    rest = []
    rest_z = None
    result = None
    cut = ""
    t_arm = None
    last = time.perf_counter()

    def hud(lines, banner, color):
        screen.fill((22, 22, 26))
        screen.blit(big.render(banner, True, color), (20, 16))
        y = 62
        for ln in lines:
            screen.blit(font.render(ln, True, (215, 215, 215)), (20, y))
            y += font.get_linesize() + 5
        screen.blit(font.render("z start   space CUT   esc quit", True,
                                (130, 130, 140)), (20, 400 - 28))
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

            z_now = keys[pygame.K_z]
            if z_now and not prev_z and result is None:
                if est is None:
                    cut = "no tracking -- cannot detect liftoff"
                elif rest_z is None:
                    cut = "still measuring resting height"
                else:
                    armed = True
                    thr = RAMP_START
                    t_arm = now
                    cut = ""
            prev_z = z_now

            if keys[pygame.K_SPACE]:
                if armed:
                    cut = "manual CUT"
                armed = False
                thr = 0

            if not armed and result is None and est is not None:
                rest.append(est["pos"][2])
                if len(rest) > REST_SECONDS * 60:
                    rest.pop(0)
                    rest_z = float(np.median(rest))

            if armed:
                if est is None:
                    armed = False
                    thr = 0
                    cut = "TRACKING LOST"
                else:
                    z, vz = est["pos"][2], est["vel"][2]
                    if z > rest_z + LIFT_HEIGHT and vz > LIFT_SPEED:
                        result = crossing_throttle(thr, vz, RAMP_RATE)
                        armed = False
                    elif thr >= cap:
                        armed = False
                        thr = 0
                        cut = f"reached throttle_cap {cap} without lifting"
                    elif now - t_arm > TIMEOUT_S:
                        armed = False
                        thr = 0
                        cut = "timed out"
                    else:
                        thr = min(thr + RAMP_RATE * step, cap)

            if not armed and thr > 0:
                thr = max(0, thr - step / RAMP_DOWN_S)

            link.send(armed or thr > 0, 0, 0, 0, thr)
            telem = link.poll()

            if result is not None and thr <= 0:
                running = False

            banner, color = (("RAMPING", (40, 200, 90)) if armed else
                              ("MEASURED", (40, 200, 90)) if result else
                              ("READY", (150, 150, 150)))
            lines = [
                "throttle : %.4f   (cap %.2f)" % (thr, cap),
                "rest z   : %s" % ("%.4f m" % rest_z if rest_z is not None
                                   else "measuring..."),
            ]
            if est:
                lines += ["z        : %+.4f m" % est["pos"][2],
                          "vz       : %+.4f m/s" % est["vel"][2]]
            else:
                lines.append("z        : NO TRACKING")
            if result is not None:
                lines.append("hover_ff : %.4f  <-- measured" % result)
            if cut:
                lines.append("note     : %s" % cut)
            if telem:
                lines.append("FC arming: %s" % decode_arm_flags(telem[7]))
            hud(lines, banner, color)
    finally:
        link.disarm_burst()
        tracker.close()
        pygame.quit()

    if result is None:
        print("no measurement taken." + (f" ({cut})" if cut else ""))
        return 1
    print(f"\nhover_ff measured at {result:.4f}")
    old, path = apply_hover_ff(args.config, result)
    print(f"hover_ff {old} -> {result:.4f}  written to {path}"
          f"  (backup at {path.name}.bak)")


if __name__ == "__main__":
    main()
