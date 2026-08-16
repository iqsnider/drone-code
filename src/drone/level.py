"""Level the stored LED body geometry against the drone sitting flat."""

import argparse
import json
import shutil
import time
from datetime import date
from pathlib import Path

import numpy as np

from drone.pose import PoseTracker

MIN_SAMPLES = 100
MAX_SPREAD_DEG = 1.5


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def correction_matrix(roll_deg, pitch_deg):
    return _Ry(np.radians(pitch_deg)) @ _Rx(np.radians(roll_deg))


def measure(tracker, seconds):
    rows = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        out = tracker.read()
        if out["raw"] and out["raw"]["ok"]:
            rows.append(out["raw"]["rpy"][:2])
    if not rows:
        return None
    a = np.array(rows)
    spread = float(np.max(np.std(a, axis=0)))
    return float(np.median(a[:, 0])), float(np.median(a[:, 1])), len(a), spread


def apply_correction(cfgdir, led_body, roll, pitch):
    C = correction_matrix(roll, pitch)
    new = (C @ np.asarray(led_body, float).T).T

    path = Path(cfgdir) / "drone.json"
    shutil.copy2(path, path.with_suffix(".json.bak"))
    d = json.loads(path.read_text())
    d["led_body"] = [[round(float(v), 5) for v in row] for row in new]
    d.setdefault("_led_source", {})["level_correction_deg"] = {
        "roll": round(roll, 3), "pitch": round(pitch, 3), "on": str(date.today())}
    path.write_text(json.dumps(d, indent=2) + "\n")
    return new, path


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[2] / "config")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="how long to average (default 5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the correction without writing it")
    args = ap.parse_args()

    print("Place the drone FLAT on the same level surface the ArUco marker sat "
          "on,\nwith all three LEDs visible to at least two cameras. Do not "
          "hold it.\n")

    tracker = PoseTracker(args.config)
    tracker.open()
    try:
        print(f"measuring for {args.seconds:g} s ...")
        got = measure(tracker, args.seconds)
        if got is None:
            print("\nno usable pose -- check that at least two cameras see all "
                  "three LEDs (run the settings calibration if not)")
            return 1
        roll, pitch, n, spread = got
        print(f"  samples {n}   roll {roll:+.2f} deg   pitch {pitch:+.2f} deg"
              f"   spread {spread:.2f} deg")
        if n < MIN_SAMPLES:
            print(f"\nonly {n} usable samples (want {MIN_SAMPLES}+) -- "
                  "the LEDs are not being seen reliably enough")
            return 1
        if spread > MAX_SPREAD_DEG:
            print(f"\nattitude moved by {spread:.2f} deg during the measurement "
                  f"(limit {MAX_SPREAD_DEG}) -- put the drone down and retry")
            return 1

        if args.dry_run:
            new = (correction_matrix(roll, pitch) @ tracker.led_body.T).T
            print("\n--dry-run, nothing written. led_body would become:")
            for i, row in enumerate(new):
                print(f"  LED{i}: [{row[0]:+.5f}, {row[1]:+.5f}, {row[2]:+.5f}]")
            return 0

        new, path = apply_correction(args.config, tracker.led_body, roll, pitch)
        print(f"\napplied roll {roll:+.2f} / pitch {pitch:+.2f} deg -> {path}"
              f"  (backup at {path.name}.bak)")
        for i, (o, v) in enumerate(zip(tracker.led_body, new)):
            print(f"  LED{i}: [{o[0]:+.4f}, {o[1]:+.4f}, {o[2]:+.4f}]"
                  f" -> [{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]")

        tracker.led_body = new
        tracker.ekf.b = new
        tracker.ekf.reset()
        print("\nverifying ...")
        got = measure(tracker, min(args.seconds, 3.0))
        if got is not None:
            r2, p2, n2, _ = got
            print(f"  now reads roll {r2:+.2f} deg   pitch {p2:+.2f} deg "
                  f"({n2} samples)")
            if max(abs(r2), abs(p2)) > 0.5:
                print("  ** still off by more than 0.5 deg -- was the drone "
                      "actually level, and is the marker surface level too?")
    finally:
        tracker.close()


if __name__ == "__main__":
    main
