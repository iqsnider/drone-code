import json
import shutil
from datetime import date
from itertools import permutations
from pathlib import Path
import time

import numpy as np

from drone.pose import PoseTracker, kabsch

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
SECONDS = 10
OUTLIER_M = 0.05        # a labeled LED this far off the running reference is
                        # a spurious blob, not the drone moving


def _label(frame, ref):
    """
    Reorder one frame's LEDs onto the reference by nearest neighbor.
    """
    perm = min(permutations(range(len(frame))),
               key=lambda p: float(((frame[list(p)] - ref) ** 2).sum()))
    labeled = frame[list(perm)]

    return labeled


def collect(tracker, seconds):
    """
    Median world position of each LED, labeled consistently across frames.
    The reference starts as the median over every frame, not the first frame,
    since one bad frame at the start would otherwise discard the whole run,
    then is refined twice: the median is dragged around by outliers far less
    than any individual frame is.
    """
    rows, t0 = [], time.time()
    while time.time() - t0 < seconds:
        out = tracker.read()
        if out["raw"]:
            rows.append(np.asarray(out["raw"]["leds"], float))

    ref = np.median(np.stack(rows), axis=0)
    for _ in range(2):
        ref = np.median(np.stack([_label(f, ref) for f in rows]), axis=0)

    kept = [o for o in (_label(f, ref) for f in rows)
            if np.abs(o - ref).max() <= OUTLIER_M]

    a = np.stack(kept)
    med = np.median(a, axis=0)
    spread = a.std(axis=0).max(axis=1)
    n_dropped = len(rows) - len(kept)

    return med, spread, len(kept), n_dropped


def align(led_body, world):
    """
    Reorder world onto led_body and return the best-fit pose. Tries every
    labeling since the triangulator's output order is arbitrary and, once the
    LEDs have been moved, nearest-neighbor to the old geometry is not
    necessarily right.
    """
    best = None
    for perm in permutations(range(len(world))):
        ordered = world[list(perm)]
        R, p, rmsd = kabsch(np.asarray(led_body, float), ordered)
        if best is None or rmsd < best[3]:
            best = (ordered, R, p, rmsd)

    return best


def sides(pts):
    lengths = [float(np.linalg.norm(pts[i] - pts[j])) for i, j in ((0, 1), (1, 2), (2, 0))]

    return lengths


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
    print("Sit the drone flat and still where at least two cameras see all "
          "three LEDs.\n")

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
    # necessarily a correct one: if stereo correspondence settles on the wrong
    # pairing it triangulates a stable triangle the drone does not have, and
    # the caliper record is the only reference here that does not come from the
    # cameras.
    ref = tracker.drone.get("_physical", {}).get("caliper_edges_mm")
    if ref:
        got = sorted(s * 1000 for s in sides(new_body))
        off = max(abs(a - b) for a, b in zip(got, sorted(ref)))
        print(f"\ncaliper cross-check   measured {[round(v, 1) for v in got]} mm"
              f"   recorded {sorted(ref)} mm   worst {off:.1f} mm")

    path = write(CONFIG_DIR, new_body, rmsd_before, rmsd_after, n)
    print(f"\nwrote {path}  (previous saved as {path.name}.bak)")
    print("\nThe body frame was carried over from the old geometry, so if the "
          "LEDs were\nphysically moved, re-measure _physical.led_from_datum_mm "
          "with calipers.\nNext:  uv run python scripts/run_drone_level_calibration.py")


if __name__ == "__main__":
    main()
