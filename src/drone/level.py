import json
import shutil
import time
from datetime import date
from pathlib import Path

import numpy as np
import typer

from drone.pose import PoseTracker

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config"


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    return R


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    return R


def correction_matrix(roll_deg, pitch_deg):
    C = _Ry(np.radians(pitch_deg)) @ _Rx(np.radians(roll_deg))

    return C


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
    roll, pitch = float(np.median(a[:, 0])), float(np.median(a[:, 1]))

    return roll, pitch, len(a), spread


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


def main(
    config: Path = DEFAULT_CONFIG,
    seconds: float = typer.Option(5, help="how long to average"),
):
    """
    Level the stored LED body geometry against the drone sitting flat.
    """
    print("Place the drone FLAT on the same level surface the ArUco marker sat "
          "on,\nwith all three LEDs visible to at least two cameras. Do not "
          "hold it.\n")

    tracker = PoseTracker(config)
    tracker.open()
    try:
        print(f"measuring for {seconds:g} s ...")
        roll, pitch, n, spread = measure(tracker, seconds)
        print(f"  samples {n}   roll {roll:+.2f} deg   pitch {pitch:+.2f} deg"
              f"   spread {spread:.2f} deg")

        new, path = apply_correction(config, tracker.led_body, roll, pitch)
        print(f"\napplied roll {roll:+.2f} / pitch {pitch:+.2f} deg -> {path}"
              f"  (backup at {path.name}.bak)")
        for i, (o, v) in enumerate(zip(tracker.led_body, new)):
            print(f"  LED{i}: [{o[0]:+.4f}, {o[1]:+.4f}, {o[2]:+.4f}]"
                  f" -> [{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]")

        tracker.led_body = new
        tracker.ekf.b = new
        tracker.ekf.reset()
        print("\nverifying ...")
        got = measure(tracker, min(seconds, 3))
        if got is not None:
            r2, p2, n2, _ = got
            print(f"  now reads roll {r2:+.2f} deg   pitch {p2:+.2f} deg "
                  f"({n2} samples)")
            if max(abs(r2), abs(p2)) > 0.5:
                print("  ** still off by more than 0.5 deg, was the drone "
                      "actually level, and is the marker surface level too?")
    finally:
        tracker.close()


if __name__ == "__main__":
    typer.run(main)
