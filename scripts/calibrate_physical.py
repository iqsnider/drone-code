"""
calibrate_physical.py -- interactive walkthrough for the drone's PHYSICAL
parameters in config/drone.json: mass, centre of mass, IR LED constellation
geometry, and the hover feed-forward throttle.

    python scripts/calibrate_physical.py        (run from the repo root)

Nothing here touches the cameras or the world frame -- that is calibrate_aruco.py
(extrinsics) and camera_setup.py (blob detection). This script only cares about
the aircraft itself.

WHAT IT SETS
    mass_kg                   as-flown mass, battery and props on
    led_body                  the 3 LED positions in BODY frame, origin at CoM
    control.reject_rmsd_m     Kabsch fit gate, derived from your measurement error
    gains.hover_ff            throttle that holds weight
    gains.throttle_cap        ceiling, kept a sane margin above hover_ff

BODY FRAME (FLU, matches mocap_core.py)
    +X forward (nose)   +Y left   +Z up   origin at the centre of mass

WHY led_body ACCURACY MATTERS
    identify_and_pose() fits your measured constellation onto the triangulated
    points. The *relative* LED geometry sets the yaw accuracy and the Kabsch
    RMSD; a 2 mm error across a 300 mm span is roughly 0.4 deg of yaw. The CoM
    position is far more forgiving -- getting it wrong just moves the tracked
    point off the true CoM, which shows up as a small tilt-coupled wobble
    (20 mm of CoM error at 12 deg of lean is about 4 mm of apparent motion).
    So: measure the LED-to-LED geometry carefully, and don't agonise over CoM.

SAFETY
    Only the hover_ff section spins motors, and it is opt-in with its own
    confirmation. Props are secured against a scale for it -- the aircraft never
    leaves the bench. Keep the transmitter bound and in reach anyway.
"""
import json
import math
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mocap_core as mc

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DRONE_JSON = CONFIG_DIR / "drone.json"

G = 9.80665
RULE = "-" * 72


# ===========================================================================
# input helpers -- Enter always keeps the current value
# ===========================================================================
def ask_float(prompt, current, unit="", lo=None, hi=None, allow_none=False):
    cur_s = "none" if current is None else f"{current:g}"
    while True:
        raw = input(f"  {prompt} [{cur_s}{unit}]: ").strip()
        if raw == "":
            return current
        if allow_none and raw.lower() in ("none", "-"):
            return None
        try:
            v = float(raw)
        except ValueError:
            print("    not a number, try again")
            continue
        if lo is not None and v < lo:
            print(f"    must be >= {lo}")
            continue
        if hi is not None and v > hi:
            print(f"    must be <= {hi}")
            continue
        return v


def ask_yn(prompt, default=True):
    d = "Y/n" if default else "y/N"
    while True:
        raw = input(f"  {prompt} [{d}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def ask_text(prompt, current=""):
    raw = input(f"  {prompt} [{current}]: ").strip()
    return raw if raw else current


def pause(msg="press Enter when done"):
    input(f"  ({msg}) ")


def header(n, title):
    print(f"\n{RULE}\n  STEP {n}  {title}\n{RULE}")


# ===========================================================================
# geometry helpers
# ===========================================================================
EDGES = ((0, 1), (0, 2), (1, 2))


def edge_lengths(led):
    led = np.asarray(led, float)
    return np.array([np.linalg.norm(led[i] - led[j]) for i, j in EDGES])


def triangle_quality(led):
    """Returns (edges_m, area_m2, min_altitude_m, scalene_spread_m)."""
    led = np.asarray(led, float)
    d = edge_lengths(led)
    area = 0.5 * np.linalg.norm(np.cross(led[1] - led[0], led[2] - led[0]))
    min_alt = 2.0 * area / max(d.max(), 1e-9)
    ds = np.sort(d)
    spread = min(ds[1] - ds[0], ds[2] - ds[1])
    return d, area, min_alt, spread


def report_constellation(led):
    d, area, min_alt, spread = triangle_quality(led)
    print(f"    edge lengths : {d[0]*1000:.1f}, {d[1]*1000:.1f}, {d[2]*1000:.1f} mm")
    print(f"    min altitude : {min_alt*1000:.1f} mm  (how far from collinear)")
    print(f"    scalene gap  : {spread*1000:.1f} mm  (closest pair of edge lengths)")
    ok = True
    if min_alt < 0.030:
        print("    ** too close to collinear. A near-straight line has an almost")
        print("       free rotation about it, so roll/pitch will be mush.")
        ok = False
    if spread < 0.020:
        print("    ** edges too similar -- this is the same warning Rig._check_")
        print("       constellation() prints. Similar edges let Kabsch fit the")
        print("       wrong labelling, which reads as a sudden yaw flip.")
        ok = False
    if ok:
        print("    constellation geometry looks good")
    return ok


# ===========================================================================
# camera models, for the simulation check
# ===========================================================================
def load_camera_models():
    models, notes = [], []
    for i in (0, 1):
        p = CONFIG_DIR / f"camera{i}.json"
        if not p.is_file():
            return None, f"{p} not found"
        c = json.loads(p.read_text())
        K = mc.K_from_intrinsics(c["fx"], c["fy"], c["cx"], c["cy"])
        ex = c.get("extrinsics", c)
        if "rvec" in ex and "tvec" in ex:
            models.append(mc.CameraModel.from_rvec_tvec(K, ex["rvec"], ex["tvec"]))
            notes.append("solved extrinsics" + (" (STALE)" if ex.get("_stale") else ""))
        else:
            models.append(mc.CameraModel.from_look_at(
                K, c["position"], c["look_at"], c.get("up", [0, 0, 1])))
            notes.append("look_at extrinsics")
    return models, ", ".join(notes)


def simulate(led, models, setpoint, gate, px_noise=0.3, n=1500, seed=0):
    """Synthetic round trip through the real estimation chain -- same shape as
    hover.py --mode selftest, but with YOUR constellation and YOUR cameras.

    The number that matters is not how often the chain fails; it is how often it
    fails WITHOUT the rmsd gate noticing. A rejected frame costs you one sample.
    An accepted wrong pose gets flown."""
    rng = np.random.default_rng(seed)
    led = np.asarray(led, float)
    c0, c1 = models
    acc_pos, acc_yaw, good_rmsd, bad_rmsd = [], [], [], []
    caught, undetected = [], []
    rmsd_all, wrong_all = [], []
    rejected = skipped = 0
    total = 0
    for _ in range(n):
        pt = np.asarray(setpoint, float) + rng.uniform(-0.3, 0.3, 3)
        yaw = rng.uniform(-np.pi, np.pi)
        roll, pitch = rng.uniform(-0.2, 0.2, 2)
        cr, sr = math.cos(roll), math.sin(roll)
        cp_, sp_ = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        R = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @
             np.array([[cp_, 0, sp_], [0, 1, 0], [-sp_, 0, cp_]]) @
             np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
        world = (R @ led.T).T + pt
        if not all(c.in_front(w) for c in models for w in world):
            skipped += 1
            continue
        total += 1
        uv0 = np.array([c0.project(w) for w in world]) + rng.normal(0, px_noise, (3, 2))
        uv1 = np.array([c1.project(w) for w in world]) + rng.normal(0, px_noise, (3, 2))
        shuf = rng.permutation(3)                     # correspondence is unknown
        Xs, _, _ = mc.match_stereo(c0, c1, uv0, uv1[shuf])
        Rf, p, rmsd, _ = mc.identify_and_pose(Xs, led)

        pe = float(np.linalg.norm(p - pt))
        ye = abs(mc.wrap_pi(mc.yaw_from_R(Rf) - yaw))
        wrong = pe > 0.05 or ye > math.radians(10)
        rmsd_all.append(rmsd)
        wrong_all.append(wrong)
        (bad_rmsd if wrong else good_rmsd).append(rmsd)
        if wrong:
            (undetected if rmsd <= gate else caught).append(rmsd)
            if rmsd <= gate:
                acc_pos.append(pe)
                acc_yaw.append(ye)
            continue
        if rmsd > gate:
            rejected += 1                              # good frame thrown away
            continue
        acc_pos.append(pe)
        acc_yaw.append(ye)
    if not acc_pos:
        return None
    return {
        "pos_mm": np.array(acc_pos) * 1000.0,
        "yaw_deg": np.degrees(acc_yaw),
        "good_rmsd_mm": np.array(good_rmsd) * 1000.0,
        "bad_rmsd_mm": np.array(bad_rmsd) * 1000.0,
        "rejected": rejected,
        "caught": len(caught),
        "undetected": len(undetected),
        "rmsd_all": np.array(rmsd_all),
        "wrong_all": np.array(wrong_all, dtype=bool),
        "total": total,
        "skipped": skipped,
    }


# ===========================================================================
# steps
# ===========================================================================
def step_mass(cfg):
    header(1, "MASS")
    print("""
  Weigh the aircraft exactly as it flies: battery in and strapped, props ON,
  LEDs and their wiring attached, canopy on. Props matter -- they are often
  15-20 g of the total on a small quad.

  A kitchen scale reading to 1 g is plenty.
""")
    grams = ask_float("as-flown mass", cfg["mass_kg"] * 1000.0, " g", lo=20, hi=25000)
    cfg["mass_kg"] = round(grams / 1000.0, 4)
    print(f"    mass_kg = {cfg['mass_kg']}   (weight {cfg['mass_kg']*G:.2f} N)")
    return cfg


def step_com(cfg, phys):
    header(2, "CENTRE OF MASS")
    print("""
  led_body is measured from the CoM, so we need the CoM first -- but only
  roughly. Pick a DATUM you can point at unambiguously and measure everything
  from it. Good choices: the centre of the flight-controller mounting square,
  or the geometric centre of the four motor shafts. Mark it with a dot of paint
  or tape so you use the same point for the LEDs in step 3.

  Then balance the drone (battery in, props on) across a straight edge -- a
  ruler on its side, a pencil, the edge of a table:

    1. Edge running LEFT-RIGHT. Slide until it balances. The balance line is
       the CoM's fore/aft station. Measure from the datum to that line.
    2. Edge running FORE-AFT. Same again, gives the lateral station.

  Sign convention: forward of datum is +, left of datum is +.
""")
    datum = ask_text("describe your datum (free text, stored as a note)",
                     phys.get("datum", "centre of FC mounting square"))
    cx = ask_float("CoM forward of datum", phys.get("com_from_datum_mm", [0, 0, 0])[0],
                   " mm", lo=-500, hi=500)
    cy = ask_float("CoM left of datum", phys.get("com_from_datum_mm", [0, 0, 0])[1],
                   " mm", lo=-500, hi=500)
    print("""
  Vertical CoM is awkward to measure and barely matters here (see the module
  docstring). If you have no better number, estimate it: on most builds it sits
  at about the battery's mid-height. Measured up from the datum, + is up.
""")
    cz = ask_float("CoM above datum", phys.get("com_from_datum_mm", [0, 0, 0])[2],
                   " mm", lo=-300, hi=300)
    phys["datum"] = datum
    phys["com_from_datum_mm"] = [cx, cy, cz]
    print(f"    CoM is at ({cx:+.1f}, {cy:+.1f}, {cz:+.1f}) mm from the datum")
    if abs(cx) > 60 or abs(cy) > 60:
        print("    note: that is a fairly off-centre CoM. Worth checking the")
        print("    battery position before you trust it -- a badly offset CoM")
        print("    also makes the aircraft harder for Betaflight to trim.")
    return cfg, phys


def step_leds(cfg, phys):
    header(3, "LED POSITIONS")
    com = np.array(phys["com_from_datum_mm"], float)
    cur = np.asarray(cfg["led_body"], float) * 1000.0 + com   # back to datum frame
    print(f"""
  Now the three IR LEDs, measured from the SAME datum as step 2 (not from the
  CoM -- the script subtracts the CoM offset for you).

  For each LED give forward (+X), left (+Y), up (+Z) in mm. Measure to the
  emitting dome, not the base of the package. Calipers beat a ruler here;
  everything downstream inherits this error.

  Datum: {phys['datum']}
""")
    led_datum = []
    for i in range(3):
        print(f"  LED {i}:")
        x = ask_float(f"  LED{i} forward", cur[i][0], " mm", lo=-1000, hi=1000)
        y = ask_float(f"  LED{i} left", cur[i][1], " mm", lo=-1000, hi=1000)
        z = ask_float(f"  LED{i} up", cur[i][2], " mm", lo=-1000, hi=1000)
        led_datum.append([x, y, z])
    led_m = (np.array(led_datum, float) - com) / 1000.0

    print("\n  In BODY frame (from CoM), metres:")
    for i, p in enumerate(led_m):
        print(f"    LED{i}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")
    print()
    report_constellation(led_m)

    # ---- independent cross-check -------------------------------------
    print("""
  CROSS-CHECK. Now measure the three LED-to-LED distances directly with
  calipers, dome to dome. This catches a transposed digit or a sign error in
  the coordinates above, which is otherwise invisible until the drone yaws
  90 deg for no reason.
""")
    if ask_yn("do the caliper cross-check?", True):
        calc = edge_lengths(led_m) * 1000.0
        meas = []
        for (i, j), c in zip(EDGES, calc):
            meas.append(ask_float(f"measured LED{i}-LED{j}", round(float(c), 1), " mm",
                                  lo=1, hi=2000))
        meas = np.array(meas, float)
        resid = meas - calc
        print("\n    edge     from coords   measured   difference")
        for (i, j), c, m, r in zip(EDGES, calc, meas, resid):
            flag = "  <-- check this" if abs(r) > 3.0 else ""
            print(f"    {i}-{j}      {c:8.1f}   {m:8.1f}   {r:+7.1f} mm{flag}")
        worst = float(np.max(np.abs(resid)))
        scale = float(np.mean(meas / np.maximum(calc, 1e-9)))
        if worst > 3.0:
            print(f"\n    Worst disagreement {worst:.1f} mm. Something is off.")
            if abs(scale - 1.0) > 0.02 and np.std(meas / calc) < 0.01:
                print(f"    All three are out by a consistent factor of {scale:.3f} --")
                print("    that smells like a unit slip or a mis-set caliper zero,")
                print("    not a coordinate typo.")
            else:
                print("    The errors are inconsistent, so it is more likely one")
                print("    coordinate is wrong. Re-measure the worst edge's two LEDs.")
            if not ask_yn("keep the coordinates anyway?", False):
                return step_leds(cfg, phys)
        else:
            print(f"\n    Agreement within {worst:.1f} mm. Coordinates look sound.")
        phys["caliper_edges_mm"] = [round(float(x), 1) for x in meas]
        phys["led_error_mm"] = round(max(worst, 1.0), 1)
    else:
        phys["led_error_mm"] = phys.get("led_error_mm", 3.0)

    phys["led_from_datum_mm"] = [[round(float(v), 1) for v in p] for p in led_datum]
    cfg["led_body"] = [[round(float(v), 4) for v in p] for p in led_m]
    return cfg, phys


def step_simulate(cfg, phys):
    header(4, "SIMULATED CHECK")
    models, note = load_camera_models()
    if models is None:
        print(f"  skipping -- {note}")
        return cfg
    print(f"  cameras loaded ({note})")
    print("""
  Running your constellation through the real chain: project -> shuffle the
  correspondence -> match_stereo -> identify_and_pose. Every failure mode this
  rig has starts as a wrong stereo pairing, so what we care about is whether
  the rmsd gate catches those before the controller acts on them.
""")
    led = cfg["led_body"]
    sp = cfg.get("setpoint", [0, 0, 1])
    gate = cfg["control"]["reject_rmsd_m"]
    results = {}
    for noise in (0.3, 0.6, 1.0):
        r = simulate(led, models, sp, gate, px_noise=noise, n=1200, seed=1)
        if r is None:
            print(f"    {noise} px: no usable frames -- the setpoint may sit outside")
            print("               a camera's view. Check the extrinsics first.")
            continue
        results[noise] = r
        rej_pct = 100.0 * r["rejected"] / r["total"]
        print(f"    {noise:.1f} px blob noise, {r['total']} frames")
        print(f"      accepted position err mm   p50 {np.percentile(r['pos_mm'],50):6.2f}"
              f"   p99 {np.percentile(r['pos_mm'],99):6.2f}")
        print(f"      accepted yaw err deg       p50 {np.percentile(r['yaw_deg'],50):6.3f}"
              f"   p99 {np.percentile(r['yaw_deg'],99):6.3f}")
        print(f"      good frames lost to gate   {r['rejected']} ({rej_pct:.1f}%)")
        print(f"      bad poses caught by gate   {r['caught']}")
        print(f"      BAD POSES THAT GOT THROUGH {r['undetected']}")

    base = results.get(0.6)
    if base is None:
        return cfg

    if base["undetected"] > 0:
        print("\n  ** Wrong poses are passing the gate. This is the one result here")
        print("     that should stop you flying. Make the triangle more scalene so")
        print("     a mis-pairing cannot masquerade as a good fit.")
    elif 100.0 * base["rejected"] / base["total"] > 8:
        print("\n  ** The gate is throwing away a lot of frames. Either it is set")
        print("     too tight, or blob noise is high -- retune detection in")
        print("     camera_setup.py before blaming the geometry.")
    else:
        print("\n  Every failure was caught by the gate. Constellation looks safe.")

    # ---- gate placement: sweep it rather than trust a tail statistic ----
    pool_r = np.concatenate([results[k]["rmsd_all"] for k in results if k >= 0.6])
    pool_w = np.concatenate([results[k]["wrong_all"] for k in results if k >= 0.6])
    led_err = phys.get("led_error_mm", 3.0) / 1000.0
    pool_r = pool_r + led_err          # your ruler error rides on every real frame

    print("\n  Gate placement. Sweeping reject_rmsd_m over the pooled 0.6 and 1.0 px")
    print("  runs -- a bigger gate keeps more good frames but lets more bad ones by.")
    print("\n    gate mm   good frames kept   bad poses passed")
    cands = np.arange(0.004, 0.061, 0.001)
    feasible = []
    for gcand in cands:
        keep = np.mean(pool_r[~pool_w] <= gcand)
        passed = int(np.sum(pool_r[pool_w] <= gcand))
        if passed == 0:
            feasible.append((gcand, keep))
        if abs(gcand * 1000 - round(gcand * 1000)) < 1e-9 and int(round(gcand * 1000)) % 5 == 0:
            mark = "" if passed else "   <- safe"
            print(f"    {gcand*1000:5.0f}     {keep*100:6.2f}%            {passed}{mark}")

    if feasible:
        gbest, keep = max(feasible, key=lambda t: t[0])
        suggest = round(gbest * 0.7, 3)          # back off from the exact edge
        keep_at = np.mean(pool_r[~pool_w] <= suggest)
        print(f"\n    Largest gate that leaks nothing: {gbest*1000:.0f} mm.")
        print(f"    Backing off 30% for sampling margin -> {suggest:.3f} m,")
        print(f"    which still keeps {keep_at*100:.1f}% of good frames.")
    else:
        suggest = 0.006
        print("\n    ** No gate value separates good poses from bad ones. The")
        print("       constellation is ambiguous -- a wrong labelling fits it")
        print("       about as well as the right one. Move an LED and re-run;")
        print("       do not fly this geometry.")

    cur = cfg["control"]["reject_rmsd_m"]
    keep_cur = np.mean(pool_r[~pool_w] <= cur)
    pass_cur = int(np.sum(pool_r[pool_w] <= cur))
    print(f"\n    current {cur} m: keeps {keep_cur*100:.1f}% of good frames,"
          f" passes {pass_cur} bad")
    print(f"    suggested reject_rmsd_m = {suggest}")
    cfg["control"]["reject_rmsd_m"] = ask_float(
        "reject_rmsd_m", suggest if ask_yn("adopt the suggestion?", True) else cur,
        " m", lo=0.002, hi=0.2)
    return cfg


def step_hover_ff(cfg):
    header(5, "HOVER FEED-FORWARD THROTTLE")
    m = cfg["mass_kg"]
    print(f"""
  hover_ff is the throttle (0..1) that produces thrust equal to weight. The
  altitude loop adds its PID on top, so a good hover_ff means the integrator
  starts near zero instead of winding up on takeoff.

  Your mass is {m:.3f} kg, so you need {m*1000:.0f} g of total static thrust.

  THE SAFE WAY TO MEASURE IT -- the aircraft stays on the bench:
    * Invert the drone and strap it to a kitchen scale so thrust pushes DOWN
      into the scale. Strap it, do not hold it.
    * Clear the area. Eye protection. Transmitter bound and in your hand as
      the kill path. Nothing loose within a couple of metres.
    * Arm, then step the throttle up in increments, letting each settle for a
      second, and read the scale each time. 4-6 points from about 20% to just
      past where the reading passes your mass.
    * Use a FRESH, fully charged pack. Voltage sag under load is real and it
      is why hover_ff creeps up as a flight goes on.

  You can skip this and type a number straight in if you already know it.
""")
    if not ask_yn("run the thrust-curve fit?", True):
        cfg["gains"]["hover_ff"] = ask_float("hover_ff directly",
                                             cfg["gains"]["hover_ff"], "",
                                             lo=0.05, hi=0.95)
        return cfg

    print("\n  Type 'PROPS ON BENCH SECURED' to confirm the drone is strapped down,")
    print("  the area is clear, and you are holding the transmitter.")
    if input("  > ").strip().upper() != "PROPS ON BENCH SECURED":
        print("  not confirmed -- skipping the powered measurement")
        cfg["gains"]["hover_ff"] = ask_float("hover_ff directly",
                                             cfg["gains"]["hover_ff"], "",
                                             lo=0.05, hi=0.95)
        return cfg

    pts = []
    print("\n  Enter throttle (0..1) and the scale reading in grams. Blank throttle ends.")
    while True:
        raw = input(f"    point {len(pts)+1} throttle: ").strip()
        if raw == "":
            break
        try:
            u = float(raw)
        except ValueError:
            print("      not a number")
            continue
        if not 0.0 < u <= 1.0:
            print("      throttle must be in (0, 1]")
            continue
        g = ask_float("  thrust reading", 0.0, " g", lo=0.0, hi=50000)
        pts.append((u, g / 1000.0))

    if len(pts) < 3:
        print("  need at least 3 points for a fit -- entering hover_ff manually")
        cfg["gains"]["hover_ff"] = ask_float("hover_ff directly",
                                             cfg["gains"]["hover_ff"], "",
                                             lo=0.05, hi=0.95)
        return cfg

    u = np.array([p[0] for p in pts])
    T = np.array([p[1] for p in pts])
    A = np.stack([u ** 2, u], axis=1)             # T = a*u^2 + b*u, through origin
    coef, *_ = np.linalg.lstsq(A, T, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(np.sum((T - pred) ** 2))
    ss_tot = float(np.sum((T - T.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if abs(a) < 1e-9:
        u_hover = m / b if b > 0 else None
    else:
        disc = b * b + 4 * a * m
        u_hover = (-b + math.sqrt(disc)) / (2 * a) if disc >= 0 else None

    print(f"\n    fit: thrust = {a:.3f}*u^2 + {b:.3f}*u   (R^2 = {r2:.4f})")
    if r2 < 0.97:
        print("    ** poor fit. Usually a point taken before the reading settled,")
        print("       or the pack sagging across the sweep. Re-run on a fresh pack.")
    if u_hover is None or not 0 < u_hover < 1:
        print("    ** the curve never reaches your weight inside 0..1 throttle.")
        print("       This airframe cannot hover as built -- check prop/motor/cell")
        print("       count before going further.")
        return cfg

    tw = (a + b) / m if m > 0 else 0.0
    print(f"    hover throttle = {u_hover:.3f}")
    print(f"    thrust-to-weight at full throttle ~ {tw:.2f}")
    if tw < 1.5:
        print("    ** T/W under 1.5 leaves almost nothing for control authority.")
    if u_hover > 0.55:
        print("    ** hovering above 55% throttle is a lot; little headroom left")
        print("       for the altitude loop to climb.")

    cfg["gains"]["hover_ff"] = round(float(u_hover), 3)

    cap = cfg["gains"]["throttle_cap"]
    floor = round(min(0.95, max(u_hover * 1.35, u_hover + 0.12)), 2)
    print(f"\n    throttle_cap is {cap}. It needs to clear hover ({u_hover:.3f}) with")
    print(f"    room for the altitude loop to climb -- at least {floor} here.")
    if cap <= u_hover:
        print("    ** the current cap is BELOW hover throttle. It cannot take off.")
        want_cap = floor
    elif cap < floor:
        print("    ** the current cap leaves almost no climb authority.")
        want_cap = floor
    else:
        print("    The current cap already clears that, so leave it alone -- a")
        print("    higher ceiling is only a bigger runaway, and you can raise it")
        print("    deliberately later.")
        want_cap = cap
    cfg["gains"]["throttle_cap"] = ask_float(
        "throttle_cap", want_cap if ask_yn("use the suggested cap?", True) else cap,
        "", lo=0.1, hi=1.0)
    return cfg


# ===========================================================================
def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def show_diff(old, new):
    fo, fn = flatten(old), flatten(new)
    changed = [k for k in fn if k in fo and fo[k] != fn[k]]
    added = [k for k in fn if k not in fo]
    if not changed and not added:
        print("  nothing changed")
        return False
    print("  changes:")
    for k in changed:
        print(f"    {k}\n        {fo[k]}\n     -> {fn[k]}")
    for k in added:
        print(f"    {k}  (new)\n     -> {fn[k]}")
    return True


def main():
    if not DRONE_JSON.is_file():
        print(f"cannot find {DRONE_JSON}")
        print("run this from the repo root, or run: python scripts/hover.py --write-config")
        return 1

    original = json.loads(DRONE_JSON.read_text())
    cfg = json.loads(json.dumps(original))
    phys = cfg.get("_physical", {})

    print(f"""
{RULE}
  PHYSICAL CALIBRATION -- {DRONE_JSON}
{RULE}

  You will need: a scale reading to 1 g, calipers or a good steel rule, a
  straight edge to balance on, and about 20 minutes. Enter accepts the value
  in brackets, so you can run this again and only redo one section.

  Body frame: +X forward, +Y left, +Z up, origin at the centre of mass.
""")
    steps = {
        "1": ("mass", lambda: step_mass(cfg)),
        "2": ("centre of mass", None),
        "3": ("LED positions", None),
        "4": ("simulated check", None),
        "5": ("hover feed-forward", None),
    }
    print("  Sections: 1 mass, 2 CoM, 3 LEDs, 4 simulated check, 5 hover_ff")
    sel = ask_text("which to run (e.g. '1235', or 'all')", "all").lower()
    run = set("12345") if sel in ("all", "") else set(c for c in sel if c in steps)

    if "1" in run:
        cfg = step_mass(cfg)
    if "2" in run:
        cfg, phys = step_com(cfg, phys)
    if "3" in run:
        if "com_from_datum_mm" not in phys:
            print("\n  step 3 needs the CoM from step 2 -- running that first")
            cfg, phys = step_com(cfg, phys)
        cfg, phys = step_leds(cfg, phys)
    if "4" in run:
        cfg = step_simulate(cfg, phys)
    if "5" in run:
        cfg = step_hover_ff(cfg)

    phys["measured_on"] = date.today().isoformat()
    cfg["_physical"] = phys

    print(f"\n{RULE}\n  SUMMARY\n{RULE}")
    if not show_diff(original, cfg):
        return 0
    print()
    if not ask_yn(f"write these to {DRONE_JSON}?", True):
        print("  nothing written")
        return 0
    shutil.copy2(DRONE_JSON, DRONE_JSON.with_suffix(".json.bak"))
    DRONE_JSON.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"  wrote {DRONE_JSON}  (previous version in {DRONE_JSON.name}.bak)")
    print("""
  Next: python scripts/hover.py --mode track
  Hold the drone in the volume by hand and confirm the reported position moves
  the way you expect and the RMSD stays under your new gate. Only then fly.
""")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted -- nothing written")
        sys.exit(130)
