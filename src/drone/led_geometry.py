"""Re-measure the drone's LED triangle from the camera rig.

`led_body` is where the three LEDs sit in the drone's own frame. Everything
downstream leans on it: the pose fit solves for the rigid transform that maps it
onto the triangulated LEDs, and the EKF gates each update on how well it agrees.
If the stored triangle is the wrong shape -- LEDs moved, or it was measured with
a calibration that has since been redone -- the fit residual never drops, every
update fails the chi-square gate, and the filter coasts forever while the raw
pose still looks fine.

The rig measures *shape* well and knows nothing about the airframe, so the body
frame has to come from somewhere. This aligns the freshly measured triangle onto
the stored `led_body` with a best-fit rigid transform and keeps that frame: the
origin stays put relative to the LED cluster and the axes keep pointing the same
way, so `setpoint`, `yaw_offset_deg` and the control gains all still mean what
they did. Only the shape changes.

If the LEDs were physically moved, mind that the origin follows the cluster --
it cannot know where the FC mounting square went. Re-measure
`_physical.led_from_datum_mm` with calipers if the offset matters, and re-run
the level calibration afterwards either way.

Sit the drone flat and still where at least two cameras see all three LEDs.
"""

import json
import shutil
from datetime import date
from itertools import permutations
from pathlib import Path
import time

import numpy as np

from drone.pose import PoseTracker, kabsch

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
SECONDS = 10.0
MIN_FRAMES = 50
OUTLIER_M = 0.05        # a labelled LED this far off the running reference is
                        # a spurious blob, not the drone moving
SIDE_SPAN_MM = 5.0      # p10-to-p90 spread a stationary triangle may show
CALIPER_TOL_MM = 25.0   # how far the rig may disagree with the caliper record


def _label(frame, ref):
    """Reorder one frame's LEDs onto the reference by nearest neighbour."""
    perm = min(permutations(range(len(frame))),
               key=lambda p: float(((frame[list(p)] - ref) ** 2).sum()))
    return frame[list(perm)]


def collect(tracker, seconds):
    """Median world position of each LED, labelled consistently across frames.

    The drone is stationary, so any one frame could in principle fix the
    labelling -- but a single spurious triangulation makes a terrible anchor,
    and picking the first frame means one bad frame at the start discards the
    whole run. So the reference starts as the median over every frame and is
    refined twice: the median is dragged around by outliers far less than any
    individual frame is.
    """
    rows, t0 = [], time.time()
    while time.time() - t0 < seconds:
        out = tracker.read()
        if out["raw"]:
            rows.append(np.asarray(out["raw"]["leds"], float))

    if len(rows) < MIN_FRAMES:
        raise SystemExit(
            f"only {len(rows)} frames with a pose (need {MIN_FRAMES}) -- check "
            "that at least two cameras see all three LEDs")

    ref = np.median(np.stack(rows), axis=0)
    for _ in range(2):
        ref = np.median(np.stack([_label(f, ref) for f in rows]), axis=0)

    kept = [o for o in (_label(f, ref) for f in rows)
            if np.abs(o - ref).max() <= OUTLIER_M]
    if len(kept) < MIN_FRAMES:
        raise SystemExit(
            f"only {len(kept)} of {len(rows)} frames agree to within "
            f"{OUTLIER_M * 1000:.0f} mm -- is the drone actually still?")

    # A stable median is not enough: if stereo correspondence flips between two
    # readings, most frames can agree with each other and still describe a
    # triangle the drone does not have. The side lengths are what a bad pairing
    # distorts, so they are what gets checked.
    spans = np.array([sorted(sides(f)) for f in kept])
    span_mm = (np.percentile(spans, 90, axis=0)
               - np.percentile(spans, 10, axis=0)).max() * 1000
    if span_mm > SIDE_SPAN_MM:
        raise SystemExit(
            f"the triangle is not stable -- side lengths vary by {span_mm:.0f} mm "
            f"across frames (limit {SIDE_SPAN_MM:.0f} mm).\n"
            "Something other than the three LEDs is being detected, or the "
            "drone moved. Open the pose viewer and check the camera panels "
            "before trusting this.")

    a = np.stack(kept)
    return np.median(a, axis=0), a.std(axis=0).max(axis=1), len(kept), len(rows) - len(kept)


def align(led_body, world):
    """Reorder `world` onto `led_body` and return the best-fit pose.

    Tries every labelling because the triangulator's output order is arbitrary
    and, once the LEDs have been moved, nearest-neighbour to the old geometry is
    not necessarily right.
    """
    best = None
    for perm in permutations(range(len(world))):
        ordered = world[list(perm)]
        R, p, rmsd = kabsch(np.asarray(led_body, float), ordered)
        if best is None or rmsd < best[3]:
            best = (ordered, R, p, rmsd)
    return best


def sides(pts):
    return [float(np.linalg.norm(pts[i] - pts[j])) for i, j in ((0, 1), (1, 2), (2, 0))]


def write(cfgdir, new_body, rmsd_before, rmsd_after, n_frames):
    path = Path(cfgdir) / "drone.json"
    shutil.copy2(path, path.with_suffix(".json.bak"))
    d = json.loads(path.read_text())
    d["led_body"] = [[round(float(v), 5) for v in row] for row in new_body]
    d.setdefault("_led_source", {}).update({
        "geometry": "triangulated from the camera rig",
        "frame": "best-fit onto the previous led_body, so the body frame is unchanged",
        "measured_on": str(date.today()),
        "n_frames": int(n_frames),
        "fit_rmsd_mm": {"before": round(rmsd_before * 1000, 2),
                        "after": round(rmsd_after * 1000, 2)},
    })
    path.write_text(json.dumps(d, indent=2) + "\n")
    return path


def main():
    print(__doc__.strip().splitlines()[-1] + "\n")

    tracker = PoseTracker(CONFIG_DIR)
    old = np.asarray(tracker.led_body, float)

    # Opening the array takes ~10 s and sits inside a blocking driver call, so
    # ctrl-c will not land until it returns. Say so, or it reads as a hang and
    # the only way out looks like killing the terminal.
    print("opening cameras (~10 s, ctrl-c will not interrupt this) ...", flush=True)
    tracker.open()
    try:
        print(f"measuring for {SECONDS:g} s, hold the drone still ...", flush=True)
        world, spread, n, dropped = collect(tracker, SECONDS)
    finally:
        tracker.close()

    print(f"  {n} frames used, {dropped} outliers dropped")
    print("  per-LED spread: " +
          "   ".join(f"led{i} {s * 1000:.1f} mm" for i, s in enumerate(spread)))

    ordered, R, p, rmsd_before = align(old, world)
    new_body = (R.T @ (ordered - p).T).T
    rmsd_after = kabsch(new_body, ordered)[2]

    print("\ntriangle side lengths:")
    for (i, j), a, b in zip(((0, 1), (1, 2), (2, 0)), sides(old), sides(new_body)):
        print(f"  {i}-{j}   old {a * 1000:7.1f} mm   new {b * 1000:7.1f} mm   "
              f"{(b - a) * 1000:+6.1f} mm")

    print("\nled_body (mm):")
    for i, (o, v) in enumerate(zip(old, new_body)):
        print(f"  led{i}  {o[0] * 1000:+7.1f} {o[1] * 1000:+7.1f} {o[2] * 1000:+7.1f}"
              f"   ->  {v[0] * 1000:+7.1f} {v[1] * 1000:+7.1f} {v[2] * 1000:+7.1f}"
              f"   (moved {np.linalg.norm(v - o) * 1000:.1f} mm)")

    print(f"\nfit residual   old geometry {rmsd_before * 1000:6.1f} mm"
          f"   new geometry {rmsd_after * 1000:6.1f} mm")

    # Cross-check against the calipers. A consistent measurement is not
    # necessarily a correct one -- if stereo correspondence settles on the wrong
    # pairing it triangulates a stable triangle the drone does not have -- and
    # the caliper record is the only reference here that does not come from the
    # cameras.
    ref = tracker.drone.get("_physical", {}).get("caliper_edges_mm")
    if ref:
        got = sorted(s * 1000 for s in sides(new_body))
        off = max(abs(a - b) for a, b in zip(got, sorted(ref)))
        print(f"\ncaliper cross-check   measured {[round(v, 1) for v in got]} mm"
              f"   recorded {sorted(ref)} mm   worst {off:.1f} mm")
        if off > CALIPER_TOL_MM:
            raise SystemExit(
                f"\nthose disagree by {off:.0f} mm (limit {CALIPER_TOL_MM:.0f}). "
                "Either the cameras are not\nseeing the three LEDs, or the LEDs "
                "moved far enough that\n_physical.caliper_edges_mm needs "
                "re-measuring. Nothing was written.")

    path = write(CONFIG_DIR, new_body, rmsd_before, rmsd_after, n)
    print(f"\nwrote {path}  (previous saved as {path.name}.bak)")
    print("\nThe body frame was carried over from the old geometry, so if the "
          "LEDs were\nphysically moved, re-measure _physical.led_from_datum_mm "
          "with calipers.\nNext:  uv run python scripts/run_drone_level_calibration.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
