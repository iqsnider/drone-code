# drone-code

A low cost mocap system for home drone autonomy

## Setup

```bash
git submodule update --init
uv sync
pio run -t upload # ESP32 firmware
```

## Calibrate


```bash
cd calibrate-ps3eyes && uv run main.py capture && uv run main.py calibrate
uv run python scripts/run_calibration_matching.py
uv run python scripts/run_camera_settings_calibration.py
uv run python scripts/run_camera_pose_calibration.py
uv run python scripts/run_led_geometry_calibration.py
uv run python scripts/run_drone_level_calibration.py
uv run python scripts/run_hover_trim_calibration.py
```

## Fly

```bash
uv run python scripts/drone_pose_viewer.py
uv run python scripts/hover.py
uv run python scripts/teleop.py
```
