#!/usr/bin/env python3
"""
Locate two PS3Eye cameras in a shared world frame from a single ArUco marker
(DICT_4X4_250, id 241) and write the result back into camera0.json / camera1.json.

WORLD FRAME == MARKER FRAME
    origin : marker centre
    +X     : marker "right"  (corner 0 -> corner 1)
    +Y     : marker "up", still in the marker plane
    +Z     : out of the printed face

Lay the marker flat on the floor, printed side up, and the world is Z-up --
which is what the configs already assume ("up": [0, 0, 1]).

WHAT GETS WRITTEN (same convention already in the files)
    extrinsics.rvec           Rodrigues vector, world -> camera
    extrinsics.tvec           translation, world -> camera, metres
    extrinsics._position_m    -R^T @ t   (camera centre in world)
    extrinsics._reproj_rms_px RMS reprojection error over the 4 corners

  i.e.  X_cam = R(rvec) @ X_world + tvec        (standard OpenCV)

EXPOSURE / GAIN
    The exposure and gain stored in each json are for blob tracking and are
    usually far too dark for a printed marker.  This script opens the cameras
    with DETECT_EXPOSURE / DETECT_GAIN instead, restores the json values on the
    way out, and never writes the boosted numbers to disk.

Just run it:   python aruco_locate_cameras.py
"""

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ==========================================================================
# everything configurable lives here
# ==========================================================================
# Where the camera jsons live. Edit PROJECT_DIR to your actual project folder;
# "~" is expanded for you. If that path does not pan out the script also looks
# in <script dir>/config and ./config before giving up.
PROJECT_DIR = Path("~/drone-code").expanduser()
CONFIG_DIR = PROJECT_DIR / "config"
CAMERA_FILES = ["camera0.json", "camera1.json"]

MARKER_ID = 241
MARKER_EDGE_LENGTH = 0.15          # metres, outer edge of the black square
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250

DETECT_EXPOSURE = 180              # PS3Eye range 0-255, temporary, not saved
DETECT_GAIN = 30                   # PS3Eye range 0-63,  temporary, not saved

SAMPLES = 60                       # good detections to average per camera
MIN_SAMPLES = 10                   # below this, give up rather than guess
WARMUP_FRAMES = 15                 # discarded while the sensor settles
TIMEOUT_S = 30.0                   # per camera

BACKEND = "auto"                   # "auto" | "pseyepy" | "opencv"
SHOW_PREVIEW = True
UPDATE_LOOKAT = True               # also refresh the legacy position/look_at
WRITE_BACKUP = True                # leaves a camera0.json.bak next to each file
# ==========================================================================


def resolve_config_dir():
    candidates = [CONFIG_DIR,
                  Path(__file__).resolve().parent / "config",
                  Path.cwd() / "config"]
    seen = []
    for c in candidates:
        if c in seen:
            continue
        seen.append(c)
        if c.is_dir() and all((c / n).is_file() for n in CAMERA_FILES):
            if c != CONFIG_DIR:
                print(f"note: {CONFIG_DIR} not usable, using {c} instead")
            return c
    raise SystemExit(
        "could not find " + " and ".join(CAMERA_FILES) + " in any of:\n  "
        + "\n  ".join(str(c) for c in seen)
        + "\nEdit PROJECT_DIR at the top of this script."
    )


# --------------------------------------------------------------------------
# aruco compatibility (OpenCV 4.7 renamed most of this API)
# --------------------------------------------------------------------------
def _tune_params(p):
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 50
    p.cornerRefinementMinAccuracy = 0.01
    # PS3Eye at 640x480 vignettes hard -> be generous with the thresholding
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 43
    p.adaptiveThreshWinSizeStep = 8
    return p


def make_detect_fn():
    """Returns detect(gray) -> (corners, ids)."""
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    else:
        d = cv2.aruco.Dictionary_get(ARUCO_DICT_ID)

    if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7
        det = cv2.aruco.ArucoDetector(d, _tune_params(cv2.aruco.DetectorParameters()))
        return lambda gray: det.detectMarkers(gray)[:2]

    params = _tune_params(cv2.aruco.DetectorParameters_create())
    return lambda gray: cv2.aruco.detectMarkers(gray, d, parameters=params)[:2]


# --------------------------------------------------------------------------
# capture backends -- both boost exposure/gain on open, restore them on close
# --------------------------------------------------------------------------
class PseyepyGrabber:
    def __init__(self, cfg):
        from pseyepy import Camera
        self.native_exposure = int(cfg.get("exposure", 60))
        self.native_gain = int(cfg.get("gain", 40))
        large = str(cfg.get("resolution", "large")).lower().startswith("l")
        self.cam = Camera(
            cfg["index"],
            fps=int(cfg.get("fps", 60)),
            resolution=Camera.RES_LARGE if large else Camera.RES_SMALL,
            colour=False,
            gain=DETECT_GAIN,
            exposure=DETECT_EXPOSURE,
        )

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
    def __init__(self, cfg):
        self.native_exposure = float(cfg.get("exposure", 60))
        self.native_gain = float(cfg.get("gain", 40))
        large = str(cfg.get("resolution", "large")).lower().startswith("l")
        w, h = (640, 480) if large else (320, 240)
        self.cap = cv2.VideoCapture(int(cfg["index"]))
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera index {cfg['index']}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, int(cfg.get("fps", 60)))
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
        # V4L2 settings stick to the device after release, so put them back
        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.native_exposure)
            self.cap.set(cv2.CAP_PROP_GAIN, self.native_gain)
        except Exception:
            pass
        self.cap.release()


def open_camera(cfg):
    if BACKEND in ("auto", "pseyepy"):
        try:
            return PseyepyGrabber(cfg)
        except Exception as e:
            if BACKEND == "pseyepy":
                raise
            print(f"  pseyepy unavailable ({e}); falling back to cv2.VideoCapture")
    return CvGrabber(cfg)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def marker_object_points():
    """Corner order matches cv2.aruco: TL, TR, BR, BL in the marker's own frame."""
    h = MARKER_EDGE_LENGTH / 2.0
    return np.array([[-h,  h, 0.0],
                     [ h,  h, 0.0],
                     [ h, -h, 0.0],
                     [-h, -h, 0.0]], dtype=np.float64)


def solve_pose(obj_pts, img_pts, K, dist):
    """
    IPPE_SQUARE is the right solver for a single square planar marker: it is
    exact, and it hands back BOTH mirror solutions so the ambiguity of the view
    can be measured instead of silently picking a flipped pose.
    """
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


def camera_centre(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ tvec.reshape(3, 1)).ravel()


def optical_axis(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return (R.T @ np.array([0.0, 0.0, 1.0])).ravel()


def look_at_point(centre, forward):
    """Where the optical axis hits the marker plane z = 0; else a point ahead."""
    if abs(forward[2]) > 1e-6:
        t = -centre[2] / forward[2]
        if t > 0:
            return centre + t * forward
    return centre + float(np.linalg.norm(centre)) * forward


# --------------------------------------------------------------------------
# per-camera capture + solve
# --------------------------------------------------------------------------
def locate_camera(cfg_path, detect):
    cfg = json.loads(Path(cfg_path).read_text())
    K = np.array([[cfg["fx"], 0.0, cfg["cx"]],
                  [0.0, cfg["fy"], cfg["cy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(cfg.get("dist", cfg.get("distortion", [0, 0, 0, 0, 0])),
                    dtype=np.float64).reshape(-1, 1)

    print(f"\n=== {cfg_path}  (device index {cfg['index']}) ===")
    print(f"  exposure {cfg.get('exposure')} -> {DETECT_EXPOSURE}, "
          f"gain {cfg.get('gain')} -> {DETECT_GAIN}  (temporary)")

    grab = open_camera(cfg)
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
        raise RuntimeError(
            f"only {len(samples)} detections of marker {MARKER_ID} in "
            f"{seen_frames} frames. Raise DETECT_EXPOSURE / DETECT_GAIN, check "
            "focus, and make sure the whole marker plus its white border is in view."
        )

    stack = np.stack(samples)                       # (N, 4, 2)
    img_pts = np.median(stack, axis=0)              # robust to the odd bad frame
    jitter = float(np.mean(np.std(stack, axis=0)))

    rvec, tvec, ambiguity = solve_pose(obj_pts, img_pts, K, dist)
    rms = reproj_rms(obj_pts, img_pts, rvec, tvec, K, dist)
    centre = camera_centre(rvec, tvec)
    fwd = optical_axis(rvec)

    print(f"  detections     : {len(samples)} / {seen_frames} frames "
          f"(corner jitter {jitter:.3f} px)")
    print(f"  reproj RMS     : {rms:.3f} px")
    print(f"  position (m)   : [{centre[0]:+.4f}, {centre[1]:+.4f}, {centre[2]:+.4f}]")
    print(f"  distance       : {np.linalg.norm(tvec):.3f} m to marker centre")
    if ambiguity < 3.0:
        print(f"  ** WARNING: planar pose ambiguity (error ratio {ambiguity:.2f}). "
              "View the marker more obliquely or from closer up.")
    if rms > 1.0:
        print("  ** WARNING: high reprojection error. fx/fy/cx/cy look like "
              "placeholders and there are no distortion coefficients -- run an "
              "intrinsic calibration for real accuracy.")

    cfg["extrinsics"] = {
        "rvec": [round(float(v), 6) for v in rvec],
        "tvec": [round(float(v), 6) for v in tvec],
        "_position_m": [round(float(v), 4) for v in centre],
        "_reproj_rms_px": round(rms, 3),
    }
    if UPDATE_LOOKAT:
        la = look_at_point(centre, fwd)
        cfg["position"] = [round(float(v), 4) for v in centre]
        cfg["look_at"] = [round(float(v), 4) for v in la]

    out = Path(cfg_path)
    if WRITE_BACKUP:
        shutil.copy2(out, out.with_suffix(out.suffix + ".bak"))
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"  -> wrote {out}")
    return centre, fwd


# --------------------------------------------------------------------------
def main():
    print(f"marker: DICT_4X4_250 id {MARKER_ID}, "
          f"{MARKER_EDGE_LENGTH * 1000:.0f} mm outer edge")
    print("Do NOT move the marker between cameras -- it defines the shared frame.")

    config_dir = resolve_config_dir()
    print(f"config dir: {config_dir}")

    detect = make_detect_fn()
    results = []
    for name in CAMERA_FILES:
        path = config_dir / name
        results.append((path, *locate_camera(path, detect)))

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
