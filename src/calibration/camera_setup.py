"""Tune the IR blob detection settings for each camera.

Shows a live view per camera with trackbars for exposure, gain, threshold and
the blob size/shape limits, and circles every blob that currently passes. Tune
until exactly the drone's IR LEDs are circled and nothing else, then press 's'
to write the values back into config/cameraN.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibration import camera_ids

CAMERA_FILES = ["camera0.json", "camera1.json"]

# (trackbar label, config key, trackbar max, trackbar -> config scale)
TRACKBARS = [
    ("exposure", "exposure", 255, 1),
    ("gain", "gain", 63, 1),
    ("threshold", "thresh", 255, 1),
    ("min area", "min_area", 500, 1),
    ("max area", "max_area", 5000, 1),
    ("min circ %", "min_circ", 100, 0.01),
]


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def detect_centroids(gray, cfg, max_n=3):
    """Brightest blobs passing the size/shape gates, as sub-pixel centroids.

    Returns (points, mask) where points is [(x, y), ...] brightest first.
    """
    k = int(cfg["blur_ksize"]) | 1               # GaussianBlur needs an odd size
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    _, mask = cv2.threshold(blur, int(cfg["thresh"]), 255, cv2.THRESH_BINARY)
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

    out.sort(key=lambda t: -t[2])                # brightest first
    return [(cx, cy) for cx, cy, _ in out[:max_n]], mask


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
def open_cameras(cfgs, indices):
    from pseyepy import Camera
    large = str(cfgs[0]["resolution"]).lower().startswith("l")
    cam = Camera(list(indices),
                 fps=int(cfgs[0]["fps"]),
                 resolution=Camera.RES_LARGE if large else Camera.RES_SMALL,
                 colour=False)
    for i, c in enumerate(cfgs):
        cam.exposure[i] = int(c["exposure"])
        cam.gain[i] = int(c["gain"])
    return cam


def read_grays(cam):
    frames, _ = cam.read(squeeze=False)
    return [np.asarray(f) for f in frames]


# ---------------------------------------------------------------------------
# trackbars
# ---------------------------------------------------------------------------
def make_windows(cfgs):
    for i, c in enumerate(cfgs):
        win = f"cam{i}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        for label, key, maxv, scale in TRACKBARS:
            init = int(round(c[key] / scale))
            cv2.createTrackbar(label, win, min(init, maxv), maxv, lambda v: None)


def read_trackbars(n):
    vals = []
    for i in range(n):
        win = f"cam{i}"
        d = {key: cv2.getTrackbarPos(label, win) * scale
             for label, key, _maxv, scale in TRACKBARS}
        d["max_area"] = max(d["max_area"], d["min_area"] + 1)
        vals.append(d)
    return vals


def save_configs(cfgs, cfgdir, tvals):
    for i, c in enumerate(cfgs):
        for _label, key, _maxv, _scale in TRACKBARS:
            c[key] = round(tvals[i][key], 3) if key == "min_circ" else int(tvals[i][key])
        path = cfgdir / CAMERA_FILES[i]
        path.write_text(json.dumps(c, indent=2) + "\n")
        print(f"  saved -> {path}")


# ---------------------------------------------------------------------------
def run(cfgdir):
    cfgs = [json.loads((cfgdir / n).read_text()) for n in CAMERA_FILES]
    indices = camera_ids.resolve_indices(cfgs)
    cam = open_cameras(cfgs, indices)
    make_windows(cfgs)

    show_mask = False
    pushed = [(None, None)] * len(cfgs)

    print("\ntune until only the drone's LEDs are circled.")
    print("  s = save   m = toggle mask view   q = quit")
    try:
        while True:
            tvals = read_trackbars(len(cfgs))

            # exposure/gain live on the sensor, so only write them on a change
            for i in range(len(cfgs)):
                want = (int(tvals[i]["exposure"]), int(tvals[i]["gain"]))
                if want != pushed[i]:
                    cam.exposure[i], cam.gain[i] = want
                    pushed[i] = want

            for i, gray in enumerate(read_grays(cam)):
                cfg = dict(cfgs[i], **tvals[i])          # live values win
                pts, mask = detect_centroids(gray, cfg)

                disp = cv2.cvtColor(mask if show_mask else gray, cv2.COLOR_GRAY2BGR)
                for (x, y) in pts:
                    cv2.circle(disp, (int(x), int(y)), 8, (0, 255, 0), 1)
                    cv2.drawMarker(disp, (int(x), int(y)), (0, 255, 0),
                                   cv2.MARKER_CROSS, 12, 1)
                colour = (0, 255, 0) if len(pts) == 3 else (0, 255, 255)
                cv2.putText(disp, f"cam{i}  blobs={len(pts)}/3  "
                            f"exp={int(tvals[i]['exposure'])} "
                            f"gain={int(tvals[i]['gain'])}",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
                cv2.imshow(f"cam{i}", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == ord("s"):
                save_configs(cfgs, cfgdir, tvals)
    finally:
        cam.end()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[2] / "config",
                    help="directory holding camera0.json and camera1.json")
    args = ap.parse_args()

    missing = [n for n in CAMERA_FILES if not (args.config / n).is_file()]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)} in {args.config}")

    print(f"config dir: {args.config}")
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
