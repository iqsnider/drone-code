"""
calibrate_extrinsics.py -- solve each camera's full pose (position + orientation,
including lens roll) from a single IR marker placed at tape-measured world points.

Use this instead of hand-measuring orientation -- essential for rolled / bird's-eye
cameras, where the roll about the optical axis can't be captured by look-at + up.

WHAT YOU NEED
  * one movable IR marker (a single IR LED on a stick works; the cameras already
    have IR filters and tuned detection)
  * a tape measure and the floor origin you chose (the world (0,0,0) point)
  * a list of world points you'll place the marker at, in metres, in the world
    frame (X = cam0->cam1, Y = into room, Z = up). Put most on the floor (Z=0) at
    grid crossings you can measure, and a FEW on a box of known height (Z>0) --
    the height points remove a planar ambiguity and improve orientation.
    Aim for >= 6 points per camera, spread across its view.

USAGE
  python calibrate_extrinsics.py --make-template points.csv   # write a CSV template
  # edit points.csv with YOUR measured coordinates, then:
  python calibrate_extrinsics.py --points points.csv          # run the capture

FLOW: it steps through each point. Place the marker there, press SPACE to grab the
brightest blob in each camera that can see it (cameras are captured independently,
so partial FOV overlap is fine). n=skip, u=undo, f=finish early, q=abort. At the
end it runs solvePnP per camera, prints the reprojection error, and writes rvec/
tvec into the camera JSONs (intrinsics/detection preserved). Re-run camera_setup.py
to confirm the frustums and a point-on-origin check.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import mocap_core as mc
from hover import Rig, detect_centroids, read_grays, open_cameras

MIN_POINTS = 6

TEMPLATE = """# world points for extrinsic calibration, metres, world frame
# X = cam0->cam1, Y = into room, Z = up. Origin = your taped floor point.
# Most on the floor (Z=0); a few on a box at known height (Z>0). >=6 per camera.
X,Y,Z
-0.6,-0.6,0.0
-0.6, 0.6,0.0
 0.6,-0.6,0.0
 0.6, 0.6,0.0
 0.0, 0.0,0.0
-0.3, 0.3,0.0
 0.3,-0.3,0.0
-0.3,-0.3,0.25
 0.3, 0.3,0.25
 0.0, 0.0,0.25
"""


def load_points(path):
    pts = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if row[0].strip().upper() == "X":
                continue
            pts.append(np.array([float(v) for v in row[:3]]))
    return pts


def solve_camera(K, world, uv):
    """Return (rvec, tvec, rms_px). Chooses a planar-safe solver if the points
    are coplanar (all similar Z)."""
    import cv2
    world = np.asarray(world, np.float64)
    uv = np.asarray(uv, np.float64)
    coplanar = np.ptp(world[:, 2]) < 0.02
    flag = cv2.SOLVEPNP_IPPE if coplanar and len(world) >= 4 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(world, uv, K, None, flags=flag)
    if not ok:
        return None
    cam = mc.CameraModel.from_rvec_tvec(K, rvec, tvec)
    rep = np.array([cam.project(w) for w in world]) - uv
    rms = float(np.sqrt(np.mean(np.sum(rep ** 2, 1))))
    return rvec.reshape(3), tvec.reshape(3), rms, coplanar


def write_extrinsics(cfgdir, idx, cam_cfg, rvec, tvec, rms):
    C = (-mc.rodrigues(rvec).T @ tvec).tolist()      # camera centre in world
    cam_cfg["extrinsics"] = {
        "rvec": [round(v, 6) for v in rvec.tolist()],
        "tvec": [round(v, 6) for v in tvec.tolist()],
        "_position_m": [round(v, 4) for v in C],     # human-readable, not read back
        "_reproj_rms_px": round(rms, 3),
    }
    (cfgdir / f"camera{idx}.json").write_text(json.dumps(cam_cfg, indent=2))
    print(f"  cam{idx}: pos={[round(v,3) for v in C]} m  reproj RMS={rms:.3f} px  -> saved")


def run_capture(cfgdir, points):
    import cv2
    rig = Rig(cfgdir)
    Ks = [mc.K_from_intrinsics(c["fx"], c["fy"], c["cx"], c["cy"]) for c in rig.cam_cfg]
    cam = open_cameras(rig)
    ncam = len(rig.cam_cfg)
    caps = [[] for _ in range(ncam)]          # per camera: list of (world, uv)
    history = []                               # for undo: list of [cam indices appended]
    idx = 0
    print(f"{len(points)} points, {ncam} cameras. SPACE=capture n=skip u=undo f=finish q=abort")
    try:
        while True:
            grays = read_grays(cam)
            here = points[idx] if idx < len(points) else None
            blobs = []
            disp = []
            for i, g in enumerate(grays):
                pts, _ = detect_centroids(g, rig.cam_cfg[i], max_n=1)
                blobs.append(pts[0] if pts else None)
                d = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                if pts:
                    x, y = pts[0]
                    cv2.drawMarker(d, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(d, f"cam{i}  captured={len(caps[i])}  "
                            f"{'BLOB' if pts else 'no blob'}", (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if pts else (0, 0, 255), 1)
                disp.append(d)
            montage = np.hstack(disp) if ncam == 2 else disp[0]
            msg = (f"point {idx+1}/{len(points)} -> place marker at "
                   f"({here[0]:+.2f},{here[1]:+.2f},{here[2]:+.2f}) m"
                   if here is not None else "all points visited -- f to finish")
            cv2.putText(montage, msg, (6, montage.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("extrinsic calibration", montage)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("aborted; nothing written."); return
            elif key == ord("f"):
                break
            elif key == ord("u") and history:
                for i in history.pop():
                    caps[i].pop()
                idx = max(0, idx - 1)
                print(f"undo -> back to point {idx+1}")
            elif key == ord("n"):
                idx = min(idx + 1, len(points))
            elif key == ord(" ") and here is not None:
                appended = []
                for i in range(ncam):
                    if blobs[i] is not None:
                        caps[i].append((here.copy(), np.array(blobs[i], float)))
                        appended.append(i)
                history.append(appended)
                print(f"captured point {idx+1} on cams {appended}")
                idx += 1
    finally:
        cam.end()
        cv2.destroyAllWindows()

    print("\nsolving...")
    for i in range(ncam):
        n = len(caps[i])
        if n < MIN_POINTS:
            print(f"  cam{i}: only {n} captures (need >={MIN_POINTS}) -- not solved.")
            continue
        world = [w for w, _ in caps[i]]
        uv = [p for _, p in caps[i]]
        res = solve_camera(Ks[i], world, uv)
        if res is None:
            print(f"  cam{i}: solvePnP failed."); continue
        rvec, tvec, rms, coplanar = res
        if rms > 2.0:
            print(f"  cam{i}: WARNING reproj RMS {rms:.2f}px is high -- check "
                  "point coordinates / correspondences before trusting this.")
        if coplanar:
            print(f"  cam{i}: points were coplanar; add a few height points if the "
                  "pose looks off.")
        write_extrinsics(cfgdir, i, rig.cam_cfg[i], rvec, tvec, rms)
    print("\ndone. Verify with:  python camera_setup.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config", type=Path)
    ap.add_argument("--points", type=Path, help="CSV of world points (X,Y,Z metres)")
    ap.add_argument("--make-template", type=Path, metavar="CSV",
                    help="write a points CSV template and exit")
    args = ap.parse_args()

    if args.make_template:
        args.make_template.write_text(TEMPLATE)
        print(f"wrote template -> {args.make_template}  (edit with your measurements)")
        return 0
    if not args.points:
        print("give --points CSV (or --make-template CSV first). See --help.")
        return 2
    if not (args.config / "camera0.json").exists():
        print(f"no config in {args.config}/ -- run:  python hover.py --write-config")
        return 2
    run_capture(args.config, load_points(args.points))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
