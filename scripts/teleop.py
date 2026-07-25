"""
teleop.py  --  keyboard teleop for the ESP32 shakeout (CPython 3.9+)

Connect your PC to the ESP32's WiFi network ("drone-shakeout"), then run:

    uv run teleop.py          (or: python teleop.py)

A window opens. It must have focus for keys to register.

CONTROLS
  arrows            roll / pitch  (hold = lean, release = level)
  w / s             throttle up / down
  a / d             yaw ccw / cw
  z                 arm / disarm toggle
  space             instant throttle cut to zero (panic)
  esc               quit (sends disarm on the way out)

Keep your transmitter on and bound as the independent hardware kill path.
"""

import socket
import struct
import sys
import time

try:
    import pygame
except ImportError:
    sys.exit("pygame required:  uv add pygame   (or pip install pygame)")

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

ESP_IP = "192.168.4.1"
CMD_PORT = 9000
TELEM_PORT = 9001

CMD_FMT = "<IBffffB"          # seq, flags, roll, pitch, yaw, throttle, angle
TELEM_FMT = "<IBffffHI"       # + loopMax, armFlags
TELEM_SIZE = struct.calcsize(TELEM_FMT)

FLAG_ARM = 0x01

# Betaflight armingDisableFlags bit -> name (4.x ordering). Exact positions
# can shift slightly between BF versions, but the common ones are stable.
ARM_FLAG_NAMES = [
    "NO_GYRO", "FAILSAFE", "RX_FAILSAFE", "BAD_RX_RECOVERY",
    "BOXFAILSAFE", "RUNAWAY_TAKEOFF", "CRASH_DETECTED", "THROTTLE",
    "ANGLE", "BOOT_GRACE_TIME", "NOPREARM", "LOAD",
    "CALIBRATING", "CLI", "CMS_MENU", "BST",
    "MSP", "PARALYZE", "GPS", "RESC",
    "RPMFILTER", "REBOOT_REQUIRED", "DSHOT_BITBANG", "ACC_CALIB",
    "MOTOR_PROTO", "ARM_SWITCH",
]


def decode_arm_flags(flags):
    if flags == 0:
        return "READY"
    if flags == 0xFFFFFFFF:
        return "(no status yet)"
    names = [ARM_FLAG_NAMES[i] if i < len(ARM_FLAG_NAMES) else "BIT%d" % i
             for i in range(32) if flags & (1 << i)]
    return ", ".join(names) if names else "0x%08X" % flags


# ---------------------------------------------------------------------------
# Control shaping
# ---------------------------------------------------------------------------

MAX_ANGLE_DEG = 25.0     # must not exceed the ESP32's clamp
ANGLE_RATE = 60.0        # deg/s the commanded lean ramps toward the key
ANGLE_RETURN = 120.0     # deg/s it returns to level on release
YAW_RATE = 2.0           # 1/s toward full yaw stick
THROTTLE_RATE = 0.5      # per second while w/s held
THROTTLE_MAX = 0.55      # cap for a bench shakeout -- raise deliberately


def approach(value, target, rate, dt):
    step = rate * dt
    if value < target:
        return min(value + step, target)
    if value > target:
        return max(value - step, target)
    return value


class TeleopState:
    def __init__(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.throttle = 0.0
        self.armed = False

    def update(self, keys, dt):
        roll_target = 0.0
        if keys[pygame.K_LEFT]:
            roll_target -= MAX_ANGLE_DEG
        if keys[pygame.K_RIGHT]:
            roll_target += MAX_ANGLE_DEG
        pitch_target = 0.0
        if keys[pygame.K_UP]:
            pitch_target += MAX_ANGLE_DEG
        if keys[pygame.K_DOWN]:
            pitch_target -= MAX_ANGLE_DEG

        r_rate = ANGLE_RATE if roll_target != 0.0 else ANGLE_RETURN
        p_rate = ANGLE_RATE if pitch_target != 0.0 else ANGLE_RETURN
        self.roll = approach(self.roll, roll_target, r_rate, dt)
        self.pitch = approach(self.pitch, pitch_target, p_rate, dt)

        yaw_target = 0.0
        if keys[pygame.K_a]:
            yaw_target -= 1.0
        if keys[pygame.K_d]:
            yaw_target += 1.0
        self.yaw = approach(self.yaw, yaw_target,
                            YAW_RATE if yaw_target else YAW_RATE * 2, dt)

        if keys[pygame.K_SPACE]:
            self.throttle = 0.0
        elif keys[pygame.K_w]:
            self.throttle = min(self.throttle + THROTTLE_RATE * dt, THROTTLE_MAX)
        elif keys[pygame.K_s]:
            self.throttle = max(self.throttle - THROTTLE_RATE * dt, 0.0)


def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 400), pygame.RESIZABLE)
    pygame.display.set_caption("drone teleop -- z arms/disarms")
    font = pygame.font.SysFont("menlo,consolas,monospace", 18)
    big = pygame.font.SysFont("menlo,consolas,monospace", 34, bold=True)
    clock = pygame.time.Clock()

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", TELEM_PORT))
    rx.setblocking(False)

    st = TeleopState()
    seq = 0
    telem = None
    last = time.perf_counter()
    prev_z = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(
                    (max(event.w, 480), max(event.h, 320)), pygame.RESIZABLE)

        keys = pygame.key.get_pressed()

        now = time.perf_counter()
        dt = now - last
        last = now

        # Plain arm/disarm toggle, edge-triggered so one physical tap of 'z'
        # flips the state exactly once (not every frame the key is held).
        z_now = keys[pygame.K_z]
        if z_now and not prev_z:
            st.armed = not st.armed
            if not st.armed:
                st.throttle = 0.0
        prev_z = z_now

        st.update(keys, dt)

        arm = st.armed
        flags = FLAG_ARM if arm else 0

        seq = (seq + 1) & 0xFFFFFFFF
        tx.sendto(
            struct.pack(CMD_FMT, seq, flags,
                        st.roll, st.pitch, st.yaw, st.throttle, 1),
            (ESP_IP, CMD_PORT),
        )

        try:
            while True:
                data, _ = rx.recvfrom(64)
                if len(data) >= TELEM_SIZE:
                    telem = struct.unpack(TELEM_FMT, data[:TELEM_SIZE])
        except (BlockingIOError, OSError):
            pass

        _draw(screen, font, big, st, arm, telem)
        pygame.display.flip()
        clock.tick(SEND_HZ)

    for _ in range(10):
        seq = (seq + 1) & 0xFFFFFFFF
        tx.sendto(struct.pack(CMD_FMT, seq, 0, 0, 0, 0, 0, 1), (ESP_IP, CMD_PORT))
        time.sleep(0.005)
    pygame.quit()


SEND_HZ = 100
STATE_NAMES = {0: "DISARMED", 1: "ARMED", 2: "FAILSAFE"}
STATE_COLORS = {0: (120, 120, 120), 1: (40, 200, 90), 2: (230, 70, 60)}
MARGIN = 20


def _draw(screen, font, big, st, arm, telem):
    screen.fill((24, 24, 28))
    w, h = screen.get_size()

    if telem is not None:
        state = telem[1]
    else:
        state = 2
    color = STATE_COLORS.get(state, (200, 200, 200))
    label = STATE_NAMES.get(state, "?")

    y = MARGIN
    title = big.render(label, True, color)
    screen.blit(title, (MARGIN, y))
    y += title.get_height() + 16

    lines = [
        "armed    : %s" % ("YES" if arm else "no"),
        "throttle : %5.2f   (cap %.2f)" % (st.throttle, THROTTLE_MAX),
        "roll     : %+6.1f deg" % st.roll,
        "pitch    : %+6.1f deg" % st.pitch,
        "yaw      : %+5.2f" % st.yaw,
    ]
    if telem is not None:
        lines.append("esp loop : %d us max" % telem[6])
        lines.append("FC arming: %s" % decode_arm_flags(telem[7]))
    else:
        lines.append("esp loop : (no telemetry -- check wifi)")
        lines.append("FC arming: (no telemetry)")

    line_h = font.get_linesize() + 8
    for ln in lines:
        screen.blit(font.render(ln, True, (210, 210, 210)), (MARGIN, y))
        y += line_h

    hint = "z=arm/disarm   space=cut   esc=quit"
    hint_surf = font.render(hint, True, (130, 130, 140))
    screen.blit(hint_surf, (MARGIN, h - hint_surf.get_height() - MARGIN))


if __name__ == "__main__":
    main()
