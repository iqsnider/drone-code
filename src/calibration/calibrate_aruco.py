import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from calibration import camera_ids

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_DIR / "config"
CAMERA_FILES = ["camera0.json", "camera1.json"]

MARKER_ID = 241
MARKER_EDGE_LENGTH = 0.15 # [m]
ARUCO_DICT_NAME = "DICT_4X4_250"

DETECT_EXPOSURE = 255
DETECT_GAIN = 63
DETECT_FPS = 15

SAMPLES = 60
MIN_SAMPLES = 10
WARMUP_FRAMES = 15
TIMEOUT_S = 45

BACKEND = "auto"
SHOW_PREVIEW = True
UPDATE_LOOKAT = True
WRITE_BACKUP = True


def resolve_config_dir():
    for c in [CONFIG_DIR, Path.cwd() / "config"]:
        if c.is_dir() and all((c / n).is_file() for n in CAMERA_FILES):
            return c
    raise SystemExit("could not find " + " and ".join(CAMERA_FILES)
                     + f" in {CONFIG_DIR} or {Path.cwd() / 'config'}")


def load_marker_spec():
    """Marker id / edge length / dictionary from markers.json, if present."""
    global MARKER_ID, MARKER_EDGE_LENGTH, ARUCO_DICT_NAME
    path = PROJECT_DIR / "markers.json"
    if not path.is_file():
        print(f"note: no {path}, using built-in marker defaults")
        return
    spec = json.loads(path.read_text())
    MARKER_ID = int(spec["origin_id"])
    MARKER_EDGE_LENGTH = float(spec["marker_length_m"])
    ARUCO_DICT_NAME = spec["dictionary"]


def _tune_params(p):
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 50
    p.cornerRefinementMinAccuracy = 0.01
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 43
    p.adaptiveThreshWinSizeStep = 8
    return p


def make_detect_fn():
    """
    Returns detect(gray) -> (corners, ids).
    """
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
    det = cv2.aruco.ArucoDetector(d, _tune_params(cv2.aruco.DetectorParameters()))
    return lambda gray: det.detectMarkers(gray)[:2]


class PseyepyGrabber:
    def __init__(self, cfg, index):
        from pseyepy import Camera
        self.native_exposure = int(cfg.get("exposure", 60))
        self.native_gain = int(cfg.get("gain", 40))
        large = str(cfg.get("resolution", "large")).lower().startswith("l")
        self.cam = Camera(
            index,
            fps=DETECT_FPS,
            resolution=Camera.RES_LARGE if large else Camera.RES_SMALL,
            colour=False,
            gain=DETECT_GAIN,
            exposure=DETECT_EXPOSURE)

    def read_gray(self):
        frame, _ts = self.cam.read()
        return np.asarray(frame, dtype=np.uint8)

    def close(self):
        try:
            self.cam.exposure = self.native_exposure
            self.cam.gain = self.native_gain
        except Exception:
            pass
        try:
            self.cam.end()
        except Exception:
            pass


class CvGrabber:

    def __init__(self, cfg, index):
        self.native_exposure = float(cfg.get("exposure", 60))
        self.native_gain = float(cfg.get("gain", 40))
        large = str(cfg.get("resolution", "large")).lower().startswith("l")
        w, h = (640, 480) if large else (320, 240)
        self.cap = cv2.VideoCapture(int(index))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, DETECT_FPS)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 == manual on V4L2
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(DETECT_EXPOSURE))
        self.cap.set(cv2.CAP_PROP_GAIN, float(DETECT_GAIN))

    def read_gray(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("frame grab failed")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def close(self):
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.native_exposure)
        self.cap.set(cv2.CAP_PROP_GAIN, self.native_gain)
        self.cap.release()


def open_camera(cfg, index):
    if BACKEND in ("auto", "pseyepy"):
        return PseyepyGrabber(cfg, index)
    return CvGrabber(cfg, index)

def marker_object_points():
    h = MARKER_EDGE_LENGTH / 2
    return np.array([[-h, h, 0],
                     [h, h, 0],
                     [h, -h, 0],
                     [-h, -h, 0]], dtype=np.float64)


def solve_pose(obj_pts, img_pts, K, dist):
    _, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    errs = np.asarray(errs).ravel()
    order = np.argsort(errs)
    best = order[0]
    ambiguity = (float(errs[order[1]] / max(errs[best], 1e-9))
                 if len(order) > 1 else float("inf"))
    return rvecs[best].reshape(3), tvecs[best].reshape(3), ambiguity


def reproj_rms(obj_pts, img_pts, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - img_pts.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def camera_center(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ tvec.reshape(3, 1)).ravel()


def optical_axis(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return (R.T @ np.array([0.0, 0.0, 1.0])).ravel()


def look_at_point(center, forward):
    if abs(forward[2]) > 1e-6:
        t = -center[2] / forward[2]
        if t > 0:
            return center + t * forward
    return center + float(np.linalg.norm(center)) * forward


def locate_camera(cfg_path, detect, index):
    cfg = json.loads(Path(cfg_path).read_text())
    K = np.array([[cfg["fx"], 0, cfg["cx"]],
                  [0, cfg["fy"], cfg["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(cfg.get("dist", cfg.get("distortion", [0, 0, 0, 0, 0])),
                    dtype=np.float64).reshape(-1, 1)

    port = cfg.get("usb_port")
    print(f"\n=== {cfg_path}  (device index {index}"
          f"{', USB port ' + port if port else ''}) ===")
    print(f"  exposure {cfg.get('exposure')} -> {DETECT_EXPOSURE}, "
          f"gain {cfg.get('gain')} -> {DETECT_GAIN}, "
          f"fps {cfg.get('fps')} -> {DETECT_FPS}  (temporary)")

    grab = open_camera(cfg, index)
    obj_pts = marker_object_points()

    samples = []
    seen_frames = 0
    t0 = time.time()
    win = f"{Path(cfg_path).stem} - marker {MARKER_ID}"
    try:
        while len(samples) < SAMPLES:
            if time.time() - t0 > TIMEOUT_S:
                break
            try:
                gray = grab.read_gray()
            except RuntimeError:
                continue
            seen_frames += 1

            corners, ids = detect(gray)
            hit = None
            if ids is not None:
                ids = np.asarray(ids).ravel()
                idx = np.where(ids == MARKER_ID)[0]
                if len(idx):
                    hit = corners[idx[0]].reshape(4, 2).astype(np.float64)
                    if seen_frames > WARMUP_FRAMES:
                        samples.append(hit)

            if SHOW_PREVIEW:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if hit is not None:
                    cv2.aruco.drawDetectedMarkers(
                        vis, [hit.reshape(1, 4, 2).astype(np.float32)],
                        np.array([[MARKER_ID]]))
                cv2.putText(vis, f"{len(samples)}/{SAMPLES}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if hit is not None else (0, 0, 255), 2)
                cv2.imshow(win, vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    raise KeyboardInterrupt("aborted by user")
    finally:
        grab.close()
        if SHOW_PREVIEW:
            cv2.destroyWindow(win)

    if len(samples) < MIN_SAMPLES:
        if DETECT_EXPOSURE >= 255 and DETECT_GAIN >= 63:
            hint = ("Already at max gain/exposure, so add light rather than "
                    "turning knobs -- and if the lenses have IR-pass filters, "
                    "printed ink needs an IR-rich lamp or the filter removed")
        else:
            hint = (f"Raise --exposure / --gain (now {DETECT_EXPOSURE}/"
                    f"{DETECT_GAIN}, max 255/63) or lower --fps (now "
                    f"{DETECT_FPS}) for a longer exposure")
        raise RuntimeError(
            f"only {len(samples)} detections of marker {MARKER_ID} in "
            f"{seen_frames} frames. {hint}. Also check focus, and that the "
            "whole marker plus its white border is in view."
        )

    stack = np.stack(samples)
    img_pts = np.median(stack, axis=0)
    jitter = float(np.mean(np.std(stack, axis=0)))

    rvec, tvec, ambiguity = solve_pose(obj_pts, img_pts, K, dist)
    rms = reproj_rms(obj_pts, img_pts, rvec, tvec, K, dist)
    center = camera_center(rvec, tvec)
    fwd = optical_axis(rvec)

    print(f"  detections     : {len(samples)} / {seen_frames} frames "
          f"(corner jitter {jitter:.3f} px)")
    print(f"  reproj RMS     : {rms:.3f} px")
    print(f"  position (m)   : [{center[0]:+.4f}, {center[1]:+.4f}, {center[2]:+.4f}]")
    print(f"  distance       : {np.linalg.norm(tvec):.3f} m to marker center")
    if ambiguity < 3:
        print(f"  ** WARNING: planar pose ambiguity (error ratio {ambiguity:.2f}). "
              "View the marker more obliquely or from closer up.")
    if rms > 1:
        print("  ** WARNING: high reprojection error. fx/fy/cx/cy look like "
              "placeholders and there are no distortion coefficients -- run an "
              "intrinsic calibration for real accuracy.")

    cfg["extrinsics"] = {
        "rvec": [round(float(v), 6) for v in rvec],
        "tvec": [round(float(v), 6) for v in tvec],
        "_position_m": [round(float(v), 4) for v in center],
        "_reproj_rms_px": round(rms, 3),
    }
    if UPDATE_LOOKAT:
        la = look_at_point(center, fwd)
        cfg["position"] = [round(float(v), 4) for v in center]
        cfg["look_at"] = [round(float(v), 4) for v in la]

    out = Path(cfg_path)
    if WRITE_BACKUP:
        shutil.copy2(out, out.with_suffix(out.suffix + ".bak"))
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"  -> wrote {out}")
    return center, fwd


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--exposure", type=int, default=DETECT_EXPOSURE,
                    metavar="0-255",
                    help=f"temporary detection exposure (default {DETECT_EXPOSURE})")
    ap.add_argument("--gain", type=int, default=DETECT_GAIN, metavar="0-63",
                    help=f"temporary detection gain (default {DETECT_GAIN})")
    ap.add_argument("--fps", type=int, default=DETECT_FPS, metavar="FPS",
                    help=f"lower fps = longer exposure = brighter "
                         f"(default {DETECT_FPS})")
    args = ap.parse_args()
    if not 0 <= args.exposure <= 255:
        ap.error("--exposure must be 0-255")
    if not 0 <= args.gain <= 63:
        ap.error("--gain must be 0-63")
    if args.fps < 1:
        ap.error("--fps must be >= 1")
    return args


def main():
    global DETECT_EXPOSURE, DETECT_GAIN, DETECT_FPS
    args = parse_args()
    DETECT_EXPOSURE, DETECT_GAIN, DETECT_FPS = args.exposure, args.gain, args.fps

    load_marker_spec()
    print(f"marker: {ARUCO_DICT_NAME} id {MARKER_ID}, "
          f"{MARKER_EDGE_LENGTH * 1000:.0f} mm outer edge")
    print("Do NOT move the marker between cameras -- it defines the shared frame.")

    config_dir = resolve_config_dir()
    print(f"config dir: {config_dir}")

    cfgs = [json.loads((config_dir / n).read_text()) for n in CAMERA_FILES]
    if BACKEND == "opencv":
        indices = [int(c["index"]) for c in cfgs]
    else:
        try:
            indices = camera_ids.resolve_indices(cfgs)
        except ImportError as e:
            print(f"  ** cannot resolve USB ports ({e}); using index fields, "
                  "which may be swapped")
            indices = [int(c["index"]) for c in cfgs]

    detect = make_detect_fn()
    results = []
    for name, index in zip(CAMERA_FILES, indices):
        path = config_dir / name
        results.append((path, *locate_camera(path, detect, index)))

    if len(results) >= 2:
        print("\n=== sanity check ===")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                (pi, ci, fi), (pj, cj, fj) = results[i], results[j]
                baseline = float(np.linalg.norm(ci - cj))
                ang = np.degrees(np.arccos(np.clip(float(np.dot(fi, fj)), -1, 1)))
                print(f"  {Path(pi).stem} <-> {Path(pj).stem}: "
                      f"baseline {baseline:.3f} m, optical axes {ang:.1f} deg apart")
        print("  Check the baseline against a tape measure. If it is off by a "
              "constant factor, MARKER_EDGE_LENGTH or fx/fy is wrong.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
