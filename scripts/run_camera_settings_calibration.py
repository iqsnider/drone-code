#!/usr/bin/env python3
"""Tune the IR blob detection settings (exposure, gain, threshold, dot size).

    uv run scripts/run_camera_settings_calibration.py
"""
from calibration.camera_setup import main

if __name__ == "__main__":
    raise SystemExit(main())
