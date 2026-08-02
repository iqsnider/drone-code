#!/usr/bin/env python3
"""Measure hover_ff: ramp throttle until the drone just floats off the floor.

    uv run scripts/run_hover_trim_calibration.py
    uv run scripts/run_hover_trim_calibration.py --dry-run

Props ON, drone on the floor, in view of both cameras. The ramp stops the
instant it lifts, so it never commands more than the throttle that holds it up.
Keep the transmitter bound as an independent kill path.
"""
from drone.trim import main

if __name__ == "__main__":
    raise SystemExit(main())
