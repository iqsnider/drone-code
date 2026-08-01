#!/usr/bin/env python3
"""Locate the cameras in space from a single ArUco marker.

    uv run scripts/run_camera_pose_calibration.py
"""
from calibration.calibrate_aruco import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)
