import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pseyepy import Camera

from calibration import camera_ids

# project files
PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_DIR / "config"

# marker info
MARKER_ID = 227
MARKER_EDGE_LENGTH = 0.16 # [m]
ARUCO_DICT_NAME = "DICT_4X4_250"

# camera detection settings
DETECT_EXPOSURE = 200
DETECT_GAIN = 63
DETECT_FPS = 15

SAMPLES = 60
MIN_SAMPLES = 10
WARMUP_FRAMES = 15
TIMEOUT_S = 45

SHOW_PREVIEW = True
UPDATE_LOOKAT = True
WRITE_BACKUP = True


def resolve_config_dir():
    """
    Finds the config directory where all of the important info on the cameras is stored
    """
    for c in [CONFIG_DIR, Path.cwd() / "config"]:
        if c.is_dir() and len(camera_ids.camera_files(c)) >= 2:
            return c
    raise SystemExit("could not find at least two cameraN.json files in "
                     f"{CONFIG_DIR} or {Path.cwd() / 'config'}")



def _tune_params(p):
    """
    aruco corner detection settings
    """
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
    Makes the gray scale aruco detector function
    """
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
    det = cv2.aruco.ArucoDetector(d, _tune_params(cv2.aruco.DetectorParameters()))
    return lambda gray: det.detectMarkers(gray)[:2]


class PseyepyGrabber:
    def __init__(self, cfg, index):
        # get camera config values or use defaults
        self.native_exposure = int(cfg.get("exposure", 60))
        self.native_gain = int(cfg.get("gain", 40))

        # set large resolution
        large = str(cfg.get("resolution", "large")).lower().startswith("l")

        # initialize the pseye camera
        self.cam = Camera(index,
                          fps=DETECT_FPS,
                          resolution=Camera.RES_LARGE if large else Camera.RES_SMALL,
                          colour=False,
                          gain=DETECT_GAIN,
                          exposure=DETECT_EXPOSURE)

    def read_gray(self):
        """
        get the camera frame
        """
        frame, _ts = self.cam.read()
        return np.asarray(frame, dtype=np.uint8)

    def close(self):
        """
        Close the camera
        """
        try:
            self.cam.exposure = self.native_exposure
            self.cam.gain = self.native_gain
        except Exception:
            pass
        try:
            self.cam.end()
        except Exception:
            pass



def open_camera(cfg, index):
    """
    opens the ps3eye camera
    """
    return PseyepyGrabber(cfg, index)

def marker_object_points():
    """
    Gets the camera corner points in the marker frame
    """
    h = MARKER_EDGE_LENGTH / 2

    return np.array([[-h, h, 0],
                     [h, h, 0],
                     [h, -h, 0],
                     [-h, -h, 0]], dtype=np.float64)


def solve_pose(obj_pts, img_pts, K, dist):
    """
    Estimate pose of aruco marker relative to the camera
    """
    # run solvePNP to get the pose and errors
    _, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)

    # flatten the erros
    errs = np.asarray(errs).ravel()

    # order the errors from smallest to largest
    order = np.argsort(errs)

    # extract the best error
    best = order[0]

    # the ratio of the second best to best error for filtering out unstable poses
    ambiguity = (float(errs[order[1]] / max(errs[best], 1e-9))
                 if len(order) > 1 else float("inf"))

    # extract the best pose estimate
    best_rvec = rvecs[best].reshape(3)
    best_tvec = tvecs[best].reshape(3)

    return best_rvec, best_tvec, ambiguity


def reproj_rms(obj_pts, img_pts, rvec, tvec, K, dist):
    """
    Computes the reprojection rms error
    """
    # compute 2D image coordinates
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)

    # difference between where 3D model thinks the point is vs where it actually is
    d = proj.reshape(-1, 2) - img_pts.reshape(-1, 2)

    # rms error of that difference
    rms_err = float(np.sqrt(np.mean(np.sum(d*d, axis=1))))

    return rms_err


def camera_center(rvec, tvec):
    """
    Finds the position of the camera center in the world frame
    """
    # world frame to camera frame rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # extracts the camera center position in the world frame
    camera_center_world = (-R.T @ tvec.reshape(3, 1)).ravel()

    return camera_center_world


def optical_axis(rvec):
    """
    Finds the camera optical axis
    """
    # world frame to camera frame rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # extracts the normal vector off the camera lens in the world frame
    optical_axis_vector = (R.T @ np.array([0, 0, 1])).ravel()

    return optical_axis_vector


def look_at_point(center, forward):
    """
    Finds the intersection a ray normal to the camera on the aruco ground plane
    """
    t = -center[2] / forward[2]

    return center + t*forward


def locate_camera(cfg_path, detect, index):
    """
    Finds the camera center and optical axis in the world frame
    """
    # open the camera config
    cfg = json.loads(Path(cfg_path).read_text())

    # create intrinsics matrix
    K = np.array([[cfg["fx"], 0, cfg["cx"]],
                  [0, cfg["fy"], cfg["cy"]],
                  [0, 0, 1]], dtype=np.float64)

    # create distortion matrix
    dist = np.array(cfg.get("dist", cfg.get("distortion", [0, 0, 0, 0, 0])),
                    dtype=np.float64).reshape(-1, 1)

    # get the usb port
    port = cfg.get("usb_port")

    grab = open_camera(cfg, index)

    # compute the object points in the marker frame
    obj_pts = marker_object_points()

    # initialize sample collection
    samples = []
    seen_frames = 0
    t0 = time.time()
    win = f"{Path(cfg_path).stem} - marker {MARKER_ID}"

    # begin collectign samples
    try:
        while len(samples) < SAMPLES:
            if time.time() - t0 > TIMEOUT_S:
                break
            try:
                gray = grab.read_gray()
            except RuntimeError:
                continue
            seen_frames += 1

            # find the corners and marker ids
            corners, ids = detect(gray)
            hit = None
            if ids is not None:
                ids = np.asarray(ids).ravel()
                idx = np.where(ids == MARKER_ID)[0]
                if len(idx):
                    hit = corners[idx[0]].reshape(4, 2).astype(np.float64)
                    if seen_frames > WARMUP_FRAMES:
                        samples.append(hit)

            # open the preview window
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
        raise SystemExit(f"{Path(cfg_path).name}: only {len(samples)} sightings "
                         f"of marker {MARKER_ID} (need {MIN_SAMPLES}) -- move the "
                         f"marker so this camera can see it")

    stack = np.stack(samples)
    img_pts = np.median(stack, axis=0)

    rvec, tvec, _ = solve_pose(obj_pts, img_pts, K, dist)
    rms = reproj_rms(obj_pts, img_pts, rvec, tvec, K, dist)
    center = camera_center(rvec, tvec)
    fwd = optical_axis(rvec)

    cfg["extrinsics"] = {"rvec": [round(float(v), 6) for v in rvec],
                         "tvec": [round(float(v), 6) for v in tvec],
                         "_position_m": [round(float(v), 4) for v in center],
                         "_reproj_rms_px": round(rms, 3)}
    if UPDATE_LOOKAT:
        la = look_at_point(center, fwd)
        cfg["position"] = [round(float(v), 4) for v in center]
        cfg["look_at"] = [round(float(v), 4) for v in la]

    # path to save config to
    out = Path(cfg_path)
    out.write_text(json.dumps(cfg, indent=2) + "\n")

    return center, fwd


def main():
    # get the config directory
    config_dir = resolve_config_dir()

    # open the configs for each camera
    names = camera_ids.camera_files(config_dir)
    cfgs = [json.loads((config_dir / n).read_text()) for n in names]
    indices = camera_ids.resolve_indices(cfgs)

    # make the detector
    detect = make_detect_fn()

    # start collection pose reults
    print(f"locating {len(names)} cameras against marker {MARKER_ID}")
    results = []
    for name, index in zip(names, indices):
        path = config_dir / name
        results.append((path, *locate_camera(path, detect, index)))


if __name__ == "__main__":
    main()
