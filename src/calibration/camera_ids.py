"""Pin each camera config file to a physical USB port.

PS3Eye cameras enumerate in whatever order the kernel finds them, so camera0
and camera1 can swap between runs. Recording the USB port path in each config
makes the assignment stick.
"""
import ctypes
import json
from pathlib import Path

CAMERA_FILES = ["camera0.json", "camera1.json"]
IDENT_MAX = 64          # matches the buffer the driver writes into

_lib_cache = None


# --------------------------------------------------------------------------
# ctypes bridge to the driver's ps3eye_get_unique_identifier()
# --------------------------------------------------------------------------
def _lib():
    """The compiled pseyepy extension, loaded so its C API can be called.

    pseyepy declares ps3eye_get_unique_identifier() in its .pyx but never wraps
    it in Python, so we reach it through the same shared object. dlopen on an
    already-loaded library hands back the same mapping, so this shares the
    driver's global context with pseyepy rather than making a second one.
    """
    global _lib_cache
    if _lib_cache is None:
        import pseyepy
        pkg = Path(pseyepy.__file__).parent
        so = next((p for p in sorted(pkg.glob("cameras*.so"))), None)
        if so is None:                       # windows / unexpected layout
            raise OSError(f"no compiled pseyepy extension found in {pkg}")
        lib = ctypes.CDLL(str(so))
        lib.ps3eye_get_unique_identifier.argtypes = [ctypes.c_int,
                                                     ctypes.c_char_p,
                                                     ctypes.c_int]
        lib.ps3eye_get_unique_identifier.restype = ctypes.c_int
        _lib_cache = lib
    return _lib_cache


def port_path(index):
    """USB port path of the camera at enumeration `index`, or None.

    The camera must already be open -- the driver looks the identifier up in its
    table of opened handles and returns failure for anything else.
    """
    try:
        lib = _lib()
    except Exception:
        return None
    buf = ctypes.create_string_buffer(IDENT_MAX)
    if lib.ps3eye_get_unique_identifier(int(index), buf, IDENT_MAX) != 0:
        return None
    return buf.value.decode(errors="replace") or None


def probe_ports():
    """Open every connected camera briefly to learn its port path.

    Returns {port_path: enumeration_index}. Opened at the small resolution and a
    low frame rate because no frames are read -- this is only about identity.
    """
    from pseyepy import Camera, cam_count
    n = cam_count()
    if n == 0:
        return {}
    cam = Camera(list(range(n)), fps=30, resolution=Camera.RES_SMALL,
                 colour=False)
    try:
        found = {}
        for i in range(n):
            p = port_path(i)
            if p:
                found[p] = i
        return found
    finally:
        cam.end()


# --------------------------------------------------------------------------
# config slot -> enumeration index
# --------------------------------------------------------------------------
def resolve_indices(cam_cfg, verbose=True):
    """Enumeration index to open for each config slot, in slot order.

    `cam_cfg` is the list of camera config dicts in slot order (camera0 first).
    Slots carrying a "usb_port" are placed by port path; any slot without one
    falls back to its "index" field. If no slot has been recorded at all this
    returns the "index" fields untouched and never touches the hardware.
    """
    wanted = [c.get("usb_port") for c in cam_cfg]
    fallback = [int(c["index"]) for c in cam_cfg]
    if not any(wanted):
        if verbose:
            print("camera identity: no usb_port recorded, trusting index "
                  "fields (run `uv run python -m calibration.camera_ids "
                  "--record` to pin them)")
        return fallback

    found = probe_ports()
    ids = []
    for slot, (want, fb) in enumerate(zip(wanted, fallback)):
        if want is None:
            print(f"  ** camera{slot}: no usb_port recorded, falling back to "
                  f"index {fb}")
            ids.append(fb)
        elif want in found:
            ids.append(found[want])
        else:
            raise RuntimeError(
                f"camera{slot} is recorded at USB port {want}, which has no "
                f"camera on it now. Connected ports: "
                f"{', '.join(sorted(found)) or 'none'}.\n"
                "Either plug that camera back into its old socket, or re-run "
                "`uv run python -m calibration.camera_ids --record` to pin "
                "the new arrangement."
            )

    if len(set(ids)) != len(ids):
        raise RuntimeError(
            f"two config slots resolved to the same camera ({ids}) -- the "
            "usb_port fields are duplicated. Re-run `uv run python -m "
            "calibration.camera_ids --record`."
        )
    if verbose:
        for slot, (want, i) in enumerate(zip(wanted, ids)):
            print(f"camera{slot}: USB port {want} -> index {i}")
    return ids


# --------------------------------------------------------------------------
# CLI: report / record the mapping
# --------------------------------------------------------------------------
def _find_config_dir(explicit=None):
    candidates = [Path(explicit)] if explicit else [
        Path(__file__).resolve().parents[2] / "config",
        Path.cwd() / "config",
    ]
    for c in candidates:
        if c.is_dir() and all((c / n).is_file() for n in CAMERA_FILES):
            return c
    raise SystemExit("could not find camera0.json and camera1.json in any of:\n  "
                     + "\n  ".join(str(c) for c in candidates))


def _to_gray(frame):
    import cv2
    import numpy as np
    f = np.asarray(frame)
    if f.ndim == 3:
        f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    if f.dtype != np.uint8:
        f = cv2.convertScaleAbs(f)
    return f


def report(cfgdir):
    """Print what is recorded, what is connected, and whether they agree."""
    cfgs = [json.loads((cfgdir / n).read_text()) for n in CAMERA_FILES]
    found = probe_ports()
    print(f"\nconnected cameras ({len(found)}):")
    for p, i in sorted(found.items(), key=lambda kv: kv[1]):
        print(f"  index {i}  USB port {p}")
    print("\nconfig:")
    for name, cfg in zip(CAMERA_FILES, cfgs):
        rec = cfg.get("usb_port")
        if rec is None:
            print(f"  {name}: no usb_port recorded, index field says "
                  f"{cfg['index']}  -- WILL SWAP")
        elif rec in found:
            i = found[rec]
            stale = "" if i == cfg.get("index") else \
                    f"  (index field says {cfg['index']}, now stale but unused)"
            print(f"  {name}: USB port {rec} -> index {i}{stale}")
        else:
            print(f"  {name}: USB port {rec} -> NOT CONNECTED")
    if not all(c.get("usb_port") for c in cfgs):
        print("\nRun `uv run python -m calibration.camera_ids --record` to pin the mapping.")


def record(cfgdir, no_preview=False):
    """Bind each config file to the USB port of the camera it should own.

    Opens the cameras in the order the configs currently ask for and shows each
    one labelled with the file it is about to be bound to, so the assignment is
    confirmed by eye before anything is written. 's' swaps, Enter accepts.
    """
    from pseyepy import Camera
    cfgs = [json.loads((cfgdir / n).read_text()) for n in CAMERA_FILES]

    found = probe_ports()
    if len(found) < len(CAMERA_FILES):
        raise SystemExit(f"need {len(CAMERA_FILES)} cameras, found {len(found)}: "
                         f"{sorted(found) or 'none'}")
    by_index = {i: p for p, i in found.items()}

    # start from what the configs already claim, so a rig that is currently
    # correct only needs an Enter
    order = [int(c["index"]) for c in cfgs]
    if sorted(order) != sorted(by_index):
        print(f"note: index fields {order} are not the connected indices "
              f"{sorted(by_index)}; starting from enumeration order instead")
        order = sorted(by_index)

    if no_preview:
        print("\n--no-preview: binding by the current index fields, unverified")
    else:
        import cv2
        cam = Camera(order, fps=60, resolution=Camera.RES_LARGE, colour=False)
        wins = [f"{n}  (slot {i})" for i, n in enumerate(CAMERA_FILES)]
        print("\nBlock one camera with your hand to tell them apart.")
        print("  s     swap the two assignments")
        print("  enter accept and write usb_port into the configs")
        print("  esc   cancel without writing")
        try:
            for w in wins:
                cv2.namedWindow(w, cv2.WINDOW_NORMAL)
            while True:
                frames, _ = cam.read()
                if not isinstance(frames, (list, tuple)):
                    frames = [frames]
                for slot, (w, f) in enumerate(zip(wins, frames)):
                    vis = cv2.cvtColor(_to_gray(f), cv2.COLOR_GRAY2BGR)
                    idx = order[slot]
                    cv2.putText(vis, f"-> {CAMERA_FILES[slot]}", (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis, f"index {idx}  port {by_index[idx]}",
                                (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1)
                    cv2.imshow(w, vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (13, 10):                     # enter
                    break
                if k == 27:                           # esc
                    raise SystemExit("cancelled, nothing written")
                if k in (ord("s"), ord("S")):
                    order.reverse()
                    print(f"swapped -> {CAMERA_FILES[0]} is index {order[0]}, "
                          f"{CAMERA_FILES[1]} is index {order[1]}")
        finally:
            cam.end()
            cv2.destroyAllWindows()

    for name, cfg, idx in zip(CAMERA_FILES, cfgs, order):
        cfg["usb_port"] = by_index[idx]
        cfg["index"] = idx                # kept as the no-usb_port fallback
        path = cfgdir / name
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"  {name}: usb_port {by_index[idx]} (index {idx}) -> wrote {path}")
    print("\nDone. The cameras will now come up the same way round regardless "
          "of enumeration order, as long as they stay in those sockets.")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--record", action="store_true",
                    help="bind each config file to a USB port (visual confirm)")
    ap.add_argument("--no-preview", action="store_true",
                    help="with --record, skip the visual confirmation")
    ap.add_argument("--config", default=None, help="config directory")
    args = ap.parse_args()

    cfgdir = _find_config_dir(args.config)
    print(f"config dir: {cfgdir}")
    if args.record:
        record(cfgdir, no_preview=args.no_preview)
    else:
        report(cfgdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
