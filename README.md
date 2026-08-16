# drone-code

Vision based drone hover. Four PS3Eye cameras track three IR LEDs, triangulate
the drone's pose at 30 Hz, and an LQR controller flies it over wifi to an ESP32.

## Setup

```bash
git submodule update --init
uv sync
pio run -t upload          # ESP32 firmware
```

## Calibrate

Run in order. Each step depends on the one before.

```bash
cd calibrate-ps3eyes && uv run main.py capture && uv run main.py calibrate && cd ..
uv run python scripts/run_calibration_matching.py        # cover each camera to bind it
uv run python scripts/run_camera_settings_calibration.py # exposure and blob thresholds
uv run python scripts/run_camera_pose_calibration.py     # camera extrinsics, ArUco marker 227
uv run python scripts/run_led_geometry_calibration.py    # LED triangle in body frame
uv run python scripts/run_drone_level_calibration.py     # level the LED geometry
uv run python scripts/run_hover_trim_calibration.py      # hover throttle
```

## Fly

```bash
uv run python scripts/drone_pose_viewer.py   # live pose and camera views on :8000
uv run python scripts/hover.py               # z arm, e engage, space cut
uv run python scripts/teleop.py              # manual control
```

## Notes

All four cameras share one USB 2.0 bus, so 640x480 is capped at 30 fps. Asking
for 60 collapses to 9 fps.

`uv run python -m calibration.camera_ids` reports which camera is on which USB
port, `--record` pins the mapping.
