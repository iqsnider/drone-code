#!/usr/bin/env python3
"""Zero the drone's roll/pitch by levelling the body frame that led_body defines.

    uv run scripts/run_drone_level_calibration.py
    uv run scripts/run_drone_level_calibration.py --dry-run
"""
from drone.level import main

if __name__ == "__main__":
    raise SystemExit(main())
