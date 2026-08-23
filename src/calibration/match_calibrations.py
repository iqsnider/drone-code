import json
from pathlib import Path

import cv2
import numpy as np
from pseyepy import Camera, cam_count

from calibration import camera_ids

PROJECT_DIR = Path(__file__).resolve().parents[2]
CALIB_DIR = PROJECT_DIR / "calibrate-ps3eyes" / "src"
CONFIG_DIR = PROJECT_DIR / "config"

# Bright settings so the room is visible and a hand over the lens is obvious.
# auto_gain MUST stay off: it would brighten the covered camera back up and
# erase the very signal this tool detects.
EXPOSURE = 200
GAIN = 63
FPS = 30

COVER_RATIO = 0.45      # covered when brightness drops below this much of baseline
CLEAR_RATIO = 0.70      # and every other camera is still above this
HOLD_FRAMES = 5         # consecutive frames before accepting, ignores a passing shadow
BASELINE_FRAMES = 30

# Used only when a camera has no previously tuned settings to inherit.
DETECTION_DEFAULTS = {
    "resolution": "large",
    "fps": 30,
    "exposure": 35,
    "gain": 32,
    "thresh": 38,
    "min_area": 3,
    "max_area": 300,
    "min_circ": 0.6,
    "blur_ksize": 5,
}


def load_calibrations():
    """
    Intrinsics from the submodule, keyed by the device index they came from.
    """
    cals = {}
    for path in sorted(CALIB_DIR.glob("camera[0-9].json")):
        data = json.loads(path.read_text())
        cals[int(data["camera_index"])] = (path.name, data)

    return cals


def existing_by_port():
    """
    Detection settings from the current configs, keyed by usb_port.

    Exposure and thresholds belong to a physical camera, not to a slot, so they
    follow the port rather than the file name when slots get reshuffled.
    """
    out = {}
    for name in camera_ids.camera_files(CONFIG_DIR):
        cfg = json.loads((CONFIG_DIR / name).read_text())
        port = cfg.get("usb_port")
        if port:
            out[port] = {k: cfg[k] for k in DETECTION_DEFAULTS if k in cfg}

    return out


def frame_means(cam):
    frames, _ = cam.read(squeeze=False)
    frames = [np.asarray(f) for f in frames]
    means = np.array([float(f.mean()) for f in frames])

    return frames, means


def tile(frames, cols=2):
    rows = int(np.ceil(len(frames) / cols))
    blank = np.zeros_like(frames[0])
    padded = list(frames) + [blank] * (rows * cols - len(frames))
    grid = np.vstack([np.hstack(padded[r * cols:(r + 1) * cols])
                      for r in range(rows)])

    return grid


def draw(frames, ratios, ports, assigned, slot, n_slots):
    """
    Preview tiles: one per device, showing how dark it currently is.
    """
    tiles = []
    for i, (f, r) in enumerate(zip(frames, ratios)):
        vis = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        covered = r < COVER_RATIO
        taken = i in assigned
        color = (120, 120, 120) if taken else ((0, 255, 0) if covered else (0, 180, 255))
        cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), color, 3)
        cv2.putText(vis, f"dev {i}  {ports.get(i, '?')}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        label = f"-> {assigned[i]}" if taken else f"{r * 100:3.0f}% lit"
        cv2.putText(vis, label, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        tiles.append(vis)

    grid = tile(tiles)
    banner = np.zeros((34, grid.shape[1], 3), np.uint8)
    cv2.putText(banner, f"cover the camera you want as camera{slot}  "
                        f"({slot + 1}/{n_slots})   esc to abort",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    preview = np.vstack([banner, grid])

    return preview


def pick_covered(ratios, taken):
    """
    Device index that is unambiguously covered, or None.

    Requires one free camera to be clearly dark *and* every other free camera
    to be clearly lit, so a light being switched off does not read as a cover.
    """
    free = [i for i in range(len(ratios)) if i not in taken]
    if not free:
        return None
    dark = min(free, key=lambda i: ratios[i])
    if ratios[dark] >= COVER_RATIO:
        return None
    if any(ratios[i] <= CLEAR_RATIO for i in free if i != dark):
        return None
    return dark


def run():
    cals = load_calibrations()
    inherited = existing_by_port()

    n = cam_count()

    cam = Camera(list(range(n)), fps=FPS, resolution=Camera.RES_SMALL,
                 colour=False, auto_gain=False, gain=GAIN, exposure=EXPOSURE)
    try:
        ports = {i: camera_ids.port_path(i) for i in range(n)}

        print(f"\n{n} cameras, {len(cals)} calibrations")
        for i in range(n):
            print(f"  device {i}  USB port {ports[i]}  <- {cals[i][0]}")

        for _ in range(10):
            frame_means(cam)                       # let the sensors settle
        base = np.zeros(n)
        for _ in range(BASELINE_FRAMES):
            base += frame_means(cam)[1]
        base /= BASELINE_FRAMES

        print("\nCover one camera at a time with your hand. Hold until it "
              "locks, then uncover.\n")
        cv2.namedWindow("match calibrations", cv2.WINDOW_NORMAL)

        assigned = {}                              # device index -> config name
        for slot in range(n):
            name = f"camera{slot}.json"
            held, candidate = 0, None
            while True:
                frames, means = frame_means(cam)
                ratios = means / base
                vis = draw(frames, ratios, ports, assigned, slot, n)
                cv2.imshow("match calibrations", vis)
                cv2.waitKey(1)

                found = pick_covered(ratios, set(assigned))
                held = held + 1 if (found is not None and found == candidate) else 0
                candidate = found
                if candidate is not None and held >= HOLD_FRAMES:
                    assigned[candidate] = name
                    print(f"  {name}: device {candidate}, USB port "
                          f"{ports[candidate]}, calibration {cals[candidate][0]}")
                    break

            while True:                            # wait for the hand to come off
                _, means = frame_means(cam)
                if (means / base)[candidate] > CLEAR_RATIO:
                    break
                cv2.waitKey(1)
    finally:
        cam.end()
        cv2.destroyAllWindows()

    write_configs(assigned, ports, cals, inherited)


def write_configs(assigned, ports, cals, inherited):
    print()
    for dev, name in sorted(assigned.items(), key=lambda kv: kv[1]):
        _, cal = cals[dev]
        port = ports[dev]
        mtx = cal["mtx"]
        dist = cal["dist"]

        cfg = dict(DETECTION_DEFAULTS)
        cfg.update(inherited.get(port, {}))
        cfg["index"] = dev
        cfg["usb_port"] = port
        cfg["fx"], cfg["fy"] = mtx[0][0], mtx[1][1]
        cfg["cx"], cfg["cy"] = mtx[0][2], mtx[1][2]
        cfg["dist"] = list(dist[0]) if isinstance(dist[0], list) else list(dist)
        cfg["_calib"] = {k: cal[k] for k in
                         ("sensor", "lens", "image_width", "image_height",
                          "reproj_error_px", "n_views", "square_length_m")
                         if k in cal}

        path = CONFIG_DIR / name
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        inh = "kept tuned settings" if port in inherited else "default settings"
        print(f"  wrote {path}  ({inh})")

    print("\nIntrinsics changed, so any stored extrinsics are stale and were "
          "not carried over.\nNext:")
    print("  uv run python scripts/run_camera_settings_calibration.py")
    print("  uv run python scripts/run_camera_pose_calibration.py")


def main():
    run()


if __name__ == "__main__":
    main()
