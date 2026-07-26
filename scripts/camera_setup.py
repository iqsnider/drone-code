"""
camera_setup.py -- live camera tracking test + tuner for the 2x PS3-Eye rig.

Shows each camera feed with detected IR blobs AND a live 3D view of the rig
(camera positions/frustums, the world origin, the hover setpoint, and the
recovered drone pose). Trackbars adjust each camera's detection settings; press
's' to write them straight back into the camera JSONs. No commands are ever sent
to the drone -- this is purely for calibrating detection and sanity-checking the
geometry before you fly.

    python camera_setup.py                 # uses ./config
    python camera_setup.py --config DIR

The 3D view is drawn into its own OpenCV window ("rig 3D") so it works even when
matplotlib has no interactive GUI backend. Orbit it with the keys below.

KEYS (focus any of the OpenCV windows):
    s        save current trackbar values into config/camera0.json + camera1.json
    m        toggle raw / binary-mask view
    r        clear the drone position trail
    i / k    tilt the 3D view up / down
    j / l    orbit the 3D view left / right
    q        quit

REFERENCE POINT: the world origin (0,0,0) is the floor point directly beneath
your hover spot. X points camera0->camera1, Y points from the cameras into the
room, Z is up. Everything the tracker reports is relative to that point; it is
defined by the position/look_at values in the camera JSONs, so set those to your
measured layout. The tool marks the origin, the setpoint, and the cameras so you
can see whether the geometry you typed matches reality.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import mocap_core as mc
from hover import Rig, detect_centroids, read_grays, open_cameras


# ---------------------------------------------------------------------------
# 3D scene drawing (no cv2/hardware -- unit-testable headless)
# ---------------------------------------------------------------------------
def frustum_corners(model, depth, w, h):
    """Four image-corner rays back-projected to `depth` metres, in world coords."""
    Kinv = np.linalg.inv(model.K)
    pts = []
    for (u, v) in [(0, 0), (w, 0), (w, h), (0, h)]:
        d = Kinv @ np.array([u, v, 1.0])
        d = d / d[2] * depth                       # point at given depth (cam z)
        pts.append(model.R.T @ d + model.C)        # camera -> world
    return np.array(pts)


def _set_equal_3d(ax, pts):
    pts = np.asarray(pts)
    lo = pts.min(0); hi = pts.max(0)
    ctr = (lo + hi) / 2
    r = max((hi - lo).max() / 2, 0.5)
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(max(0, ctr[2] - r), ctr[2] + r)
    ax.set_box_aspect((1, 1, 1))


def draw_scene(ax, rig, pose=None, trail=None):
    """pose: dict with pos(3), R(3x3), world_leds(3,3) or None."""
    ax.clear()
    res_w = 640 if rig.cam_cfg[0]["resolution"] == "large" else 320
    res_h = 480 if rig.cam_cfg[0]["resolution"] == "large" else 240
    setpoint = np.array(rig.drone["setpoint"], float)

    bounds = [np.zeros(3), setpoint]
    # floor grid at z=0
    gmin, gmax = -1.5, 1.5
    for x in np.linspace(gmin, gmax, 7):
        ax.plot([x, x], [gmin, gmax], [0, 0], color="0.85", lw=0.6, zorder=0)
    for y in np.linspace(gmin, gmax, 7):
        ax.plot([gmin, gmax], [y, y], [0, 0], color="0.85", lw=0.6, zorder=0)

    # world axes at the origin (the reference point)
    L = 0.5
    ax.quiver(0, 0, 0, L, 0, 0, color="r", lw=2)
    ax.quiver(0, 0, 0, 0, L, 0, color="g", lw=2)
    ax.quiver(0, 0, 0, 0, 0, L, color="b", lw=2)
    ax.scatter([0], [0], [0], color="k", s=40)
    ax.text(0, 0, -0.08, "origin", color="k", fontsize=8, ha="center")

    # hover setpoint
    ax.scatter(*setpoint, color="orange", marker="*", s=160, zorder=5)
    ax.text(setpoint[0], setpoint[1], setpoint[2] + 0.08, "setpoint",
            color="orange", fontsize=8, ha="center")
    ax.plot([setpoint[0], setpoint[0]], [setpoint[1], setpoint[1]],
            [0, setpoint[2]], color="orange", lw=0.6, ls=":")

    # cameras + frustums
    for i, (m, c) in enumerate(zip(rig.models, rig.cam_cfg)):
        C = m.C
        bounds.append(C)
        ax.scatter(*C, color="navy", s=50, marker="s")
        ax.text(C[0], C[1], C[2] + 0.1, f"cam{i}", color="navy", fontsize=9, ha="center")
        fc = frustum_corners(m, depth=np.linalg.norm(C - setpoint), w=res_w, h=res_h)
        for corner in fc:
            ax.plot([C[0], corner[0]], [C[1], corner[1]], [C[2], corner[2]],
                    color="steelblue", lw=0.7, alpha=0.7)
        loop = np.vstack([fc, fc[0]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="steelblue", lw=0.7, alpha=0.7)

    # drone pose
    if pose is not None:
        p = np.asarray(pose["pos"], float)
        bounds.append(p)
        leds = np.asarray(pose["world_leds"], float)
        tri = np.vstack([leds, leds[0]])
        ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], color="crimson", lw=1.2)
        ax.scatter(leds[:, 0], leds[:, 1], leds[:, 2], color="red", s=30)
        ax.scatter(*p, color="crimson", s=60, marker="o")
        R = np.asarray(pose["R"], float)
        for vec, col in zip(R.T, ("darkred", "darkgreen", "darkblue")):
            ax.quiver(p[0], p[1], p[2], *(0.25 * vec), color=col, lw=2)
        # drop line to floor
        ax.plot([p[0], p[0]], [p[1], p[1]], [0, p[2]], color="crimson", lw=0.5, ls=":")

    if trail is not None and len(trail) > 1:
        t = np.array(trail)
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color="crimson", lw=1.0, alpha=0.5)

    _set_equal_3d(ax, np.array(bounds))
    ax.set_xlabel("X  (cam0->cam1)")
    ax.set_ylabel("Y  (into room)")
    ax.set_zlabel("Z  (up)")


# ---------------------------------------------------------------------------
# Trackbar plumbing (cv2)
# ---------------------------------------------------------------------------
_TRACKBARS = [  # (label, cfg key, max, scale)  scale converts trackbar<->cfg
    ("exposure", "exposure", 255, 1),
    ("gain", "gain", 63, 1),
    ("threshold", "thresh", 255, 1),
    ("min area", "min_area", 500, 1),
    ("max area", "max_area", 5000, 1),
    ("min circ %", "min_circ", 100, 0.01),
]


def make_windows(rig):
    import cv2
    for i, c in enumerate(rig.cam_cfg):
        win = f"cam{i}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        for label, key, maxv, scale in _TRACKBARS:
            init = int(round(c[key] / scale))
            cv2.createTrackbar(label, win, min(init, maxv), maxv, lambda v: None)


def read_trackbars(rig):
    import cv2
    vals = []
    for i in range(len(rig.cam_cfg)):
        win = f"cam{i}"
        d = {}
        for label, key, maxv, scale in _TRACKBARS:
            d[key] = cv2.getTrackbarPos(label, win) * scale
        d["max_area"] = max(d["max_area"], d["min_area"] + 1)
        vals.append(d)
    return vals


def push_camera_params(cam, idx, exposure, gain):
    """Live exposure/gain update, tolerant of pseyepy's per-camera list attrs."""
    for name, value in (("exposure", exposure), ("gain", gain)):
        try:
            cur = getattr(cam, name)
            if isinstance(cur, (list, tuple, np.ndarray)):
                cur = list(cur)
                while len(cur) <= idx:
                    cur.append(value)
                cur[idx] = int(value)
                setattr(cam, name, cur)
            else:
                setattr(cam, name, int(value))
        except Exception:
            pass


def save_configs(rig, cfgdir, tvals):
    for i, c in enumerate(rig.cam_cfg):
        for key in ("exposure", "gain", "thresh", "min_area", "max_area", "min_circ"):
            c[key] = (round(tvals[i][key], 3) if key == "min_circ"
                      else int(tvals[i][key]))
        (cfgdir / f"camera{i}.json").write_text(json.dumps(c, indent=2))
    print(f"saved detection settings -> {cfgdir}/camera0.json, camera1.json")


# ---------------------------------------------------------------------------
def run(cfgdir):
    import cv2
    import matplotlib
    matplotlib.use("Agg")                       # render offscreen, show via cv2
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    rig = Rig(cfgdir)
    cam = open_cameras(rig)
    make_windows(rig)

    # Offscreen 3D figure; we blit it into an OpenCV window each update so it
    # works regardless of whether matplotlib has an interactive GUI backend.
    fig = Figure(figsize=(6.5, 5.5), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    view = {"elev": 22.0, "azim": -60.0}        # orbit with i/k/j/l keys
    cv2.namedWindow("rig 3D", cv2.WINDOW_NORMAL)

    filt = mc.StateFilter(**rig.drone["filter"])
    reject = rig.drone["control"]["reject_rmsd_m"]
    trail = []
    last = time.perf_counter()
    last_3d = 0.0
    show_mask = False
    last_push = [(None, None)] * len(rig.cam_cfg)
    last_pose = None

    print("camera_setup: tune detection, watch the 3D pose.")
    print("  s=save  m=mask  r=trail  i/k=tilt  j/l=orbit  q=quit")
    try:
        while True:
            tvals = read_trackbars(rig)
            for i in range(len(rig.cam_cfg)):
                want = (int(tvals[i]["exposure"]), int(tvals[i]["gain"]))
                if want != last_push[i]:
                    push_camera_params(cam, i, *want)
                    last_push[i] = want

            grays = read_grays(cam)
            now = time.perf_counter(); dt = now - last; last = now

            dets, masks = [], []
            for i, g in enumerate(grays):
                cfg = dict(rig.cam_cfg[i], **tvals[i])   # live values override
                pts, mask = detect_centroids(g, cfg)
                dets.append(pts); masks.append(mask)

            est = rig.estimate(dets[0], dets[1]) if len(dets) == 2 else None
            filt.predict(dt)
            good = bool(est and est["rmsd"] <= reject)
            if good:
                filt.update(est["pos"], est["yaw"], dt)
                last_pose = {"pos": filt.x.copy(), "R": est["R"],
                             "world_leds": est["world_leds"]}
                trail.append(filt.x.copy())
                if len(trail) > 200:
                    trail.pop(0)

            # camera windows
            for i, g in enumerate(grays):
                disp = cv2.cvtColor(masks[i] if show_mask else g, cv2.COLOR_GRAY2BGR)
                for (x, y) in dets[i]:
                    cv2.circle(disp, (int(x), int(y)), 8, (0, 255, 0), 1)
                    cv2.drawMarker(disp, (int(x), int(y)), (0, 255, 0),
                                   cv2.MARKER_CROSS, 12, 1)
                cv2.putText(disp, f"cam{i}  blobs={len(dets[i])}  "
                            f"exp={int(tvals[i]['exposure'])} gain={int(tvals[i]['gain'])}",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if est:
                    col = (0, 255, 0) if good else (0, 0, 255)
                    cv2.putText(disp, f"rmsd={est['rmsd']*1000:.1f}mm", (6, disp.shape[0]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                cv2.imshow(f"cam{i}", disp)

            # 3D view -- render offscreen at ~10 Hz and blit into a cv2 window
            if now - last_3d > 0.1:
                last_3d = now
                ax.view_init(elev=view["elev"], azim=view["azim"])
                draw_scene(ax, rig, pose=last_pose if (good or last_pose) else None,
                           trail=trail)
                if last_pose is not None:
                    p = last_pose["pos"]
                    ax.set_title(f"pos ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}) m   "
                                 f"yaw {np.degrees(filt.yaw):+.0f}"
                                 f"{'' if good else '  [stale]'}", fontsize=9)
                else:
                    ax.set_title("no pose -- need 3 blobs in BOTH cameras", fontsize=9)
                canvas.draw()
                buf = np.asarray(canvas.buffer_rgba())
                cv2.imshow("rig 3D", cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == ord("r"):
                trail.clear()
            elif key == ord("s"):
                save_configs(rig, cfgdir, tvals)
            elif key == ord("j"):
                view["azim"] -= 8
            elif key == ord("l"):
                view["azim"] += 8
            elif key == ord("i"):
                view["elev"] = min(89, view["elev"] + 8)
            elif key == ord("k"):
                view["elev"] = max(-89, view["elev"] - 8)
    finally:
        cam.end()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config", type=Path)
    args = ap.parse_args()
    if not (args.config / "camera0.json").exists():
        print(f"no config in {args.config}/ -- run:  python hover.py --write-config")
        return 2
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
