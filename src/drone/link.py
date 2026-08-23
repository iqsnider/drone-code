import socket
import struct
import time

CMD_FMT = "<IBffffB"
TELEM_FMT = "<IBffffHIH"
TELEM_SIZE = struct.calcsize(TELEM_FMT)
FLAG_ARM = 0x01
CMD_PORT = 9000
TELEM_PORT = 9001

ARM_FLAG_NAMES = [
    "NO_GYRO", "FAILSAFE", "RX_FAILSAFE", "BAD_RX_RECOVERY", "BOXFAILSAFE",
    "RUNAWAY_TAKEOFF", "CRASH_DETECTED", "THROTTLE", "ANGLE", "BOOT_GRACE_TIME",
    "NOPREARM", "LOAD", "CALIBRATING", "CLI", "CMS_MENU", "BST", "MSP",
    "PARALYZE", "GPS", "RESC", "RPMFILTER", "REBOOT_REQUIRED", "DSHOT_BITBANG",
    "ACC_CALIB", "MOTOR_PROTO", "ARM_SWITCH",
]


def decode_arm_flags(flags):
    if flags == 0:
        return "ready"
    if flags == 0xFFFFFFFF:
        return "(no reply yet)"
    names = [n for i, n in enumerate(ARM_FLAG_NAMES) if flags & (1 << i)]
    label = ", ".join(names) or f"0x{flags:08x}"

    return label


class Link:
    def __init__(self, esp_ip, send=True):
        self.esp = (esp_ip, CMD_PORT)
        self.send_enabled = send
        self.seq = 0
        self.telem = None
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", TELEM_PORT))
        self.rx.setblocking(False)

    def send(self, arm, roll, pitch, yaw, throttle, angle_mode=1):
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        if not self.send_enabled:
            return
        self.tx.sendto(struct.pack(CMD_FMT, self.seq, FLAG_ARM if arm else 0,
                                   float(roll), float(pitch), float(yaw),
                                   float(throttle), angle_mode), self.esp)

    def poll(self):
        try:
            while True:
                data, _ = self.rx.recvfrom(64)
                if len(data) >= TELEM_SIZE:
                    self.telem = struct.unpack(TELEM_FMT, data[:TELEM_SIZE])
        except (BlockingIOError, OSError):
            pass

        return self.telem

    def disarm_burst(self, n=10):
        for _ in range(n):
            self.send(False, 0, 0, 0, 0)
            time.sleep(0.005)
