#include <Arduino.h>
/*
 * ESP32 DevKitV1 -> iFlight SucceX-F722  (PlatformIO: src/main.cpp)
 *
 * ITERATION 1: open-loop teleop shakeout.
 *
 *     keyboard --UDP--> ESP32 SoftAP --MSP/UART--> FC --> motors
 *
 * Betaflight runs ANGLE mode and does the stabilizing; the arrow keys
 * command lean angles. The ESP32 is a WiFi-to-MSP bridge with a link
 * watchdog failsafe, and it polls MSP_STATUS_EX so the FC's own
 * arming-disable reason shows up in the teleop HUD.
 *
 * Core 0: fixed 100 Hz timer, emits MSP_SET_RAW_RC, polls status.
 * Core 1 (Arduino loop): WiFi + UDP receive + telemetry.
 *
 * WIRING  (both sides 3.3V -- no level shifting)
 *   GPIO17 (TX2) -> F722 RX4
 *   GPIO16 (RX2) <- F722 TX4
 *   GND          -- GND
 *   VIN (5V)     <- FC 5V BEC pad
 *   Do NOT power from USB and the BEC at once.
 *
 * BETAFLIGHT
 *   Ports    : Configuration/MSP ON for UART4, 115200
 *   Receiver : Receiver Mode = MSP RX
 *   Modes    : ARM on AUX1 1700-2100 ; ANGLE on AUX2 1700-2100
 *   Failsafe : stage 2 = Drop
 *   AIR_MODE : OFF
 */
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>
// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const char *AP_SSID = "drone-shakeout";
const char *AP_PASS = "hover1234";
const uint8_t AP_CHANNEL = 6;           // 1, 6, or 11 -- pick the emptiest
const uint16_t LISTEN_PORT = 9000;
const uint16_t TELEM_PORT = 9001;
IPAddress AP_IP(192, 168, 4, 1);
IPAddress AP_GW(192, 168, 4, 1);
IPAddress AP_MASK(255, 255, 255, 0);
const int PIN_TX = 17;          // -> FC RX4
const int PIN_RX = 16;          // <- FC TX4
const uint32_t MSP_BAUD = 115200;
HardwareSerial &FC = Serial2;
const int PIN_LED = 2;
const uint32_t LOOP_HZ = 100;
const uint32_t LOOP_US = 1000000UL / LOOP_HZ;
const uint32_t LINK_TIMEOUT_MS = 200;
const float MAX_ANGLE_DEG = 25.0f;
// ---------------------------------------------------------------------------
// MSP
// ---------------------------------------------------------------------------
const uint8_t MSP_SET_RAW_RC = 200;
const uint8_t MSP_STATUS_EX = 150;  // reply carries armingDisableFlags cleanly
static void mspSend(HardwareSerial &s, uint8_t cmd,
                    const uint8_t *payload, uint8_t size) {
  uint8_t chk = size ^ cmd;
  s.write('$');
  s.write('M');
  s.write('<');
  s.write(size);
  s.write(cmd);
  for (uint8_t i = 0; i < size; i++) {
    s.write(payload[i]);
    chk ^= payload[i];
  }
  s.write(chk);
}
// Incremental parser for MSP v1 replies ($M> ...). We only care about
// MSP_STATUS_EX (150). armingDisableFlags is the last u32 before the
// trailing configState byte, so we read it end-relative:
// payload[size-5 .. size-2]. This is stable regardless of how many
// version-dependent mode bytes preceded it.
class MspStatusReader {
 public:
  bool poll(HardwareSerial &s, uint32_t *flagsOut) {
    bool got = false;
    while (s.available()) {
      uint8_t b = s.read();
      switch (st_) {
        case IDLE:    st_ = (b == '$') ? HDR_M : IDLE; break;
        case HDR_M:   st_ = (b == 'M') ? HDR_D : IDLE; break;
        case HDR_D:   st_ = (b == '>') ? SIZE : IDLE; break;
        case SIZE:    size_ = b; idx_ = 0; chk_ = b; st_ = CMD; break;
        case CMD:     cmd_ = b; chk_ ^= b;
                      st_ = (size_ == 0) ? CHK : DATA; break;
        case DATA:
          if (idx_ < sizeof(buf_)) buf_[idx_] = b;
          chk_ ^= b;
          if (++idx_ >= size_) st_ = CHK;
          break;
        case CHK:
          if (b == chk_ && cmd_ == MSP_STATUS_EX && size_ >= 5 &&
              size_ <= sizeof(buf_)) {
            uint32_t f;
            memcpy(&f, &buf_[size_ - 5], 4);   // little-endian
            *flagsOut = f;
            got = true;
          }
          st_ = IDLE;
          break;
      }
    }
    return got;
  }
 private:
  enum St { IDLE, HDR_M, HDR_D, SIZE, CMD, DATA, CHK };
  St st_ = IDLE;
  uint8_t size_ = 0, cmd_ = 0, chk_ = 0, idx_ = 0;
  uint8_t buf_[96];
};
// ---------------------------------------------------------------------------
// RC channels (Betaflight AETR default)
// ---------------------------------------------------------------------------
const uint8_t CH_ROLL = 0;
const uint8_t CH_PITCH = 1;
const uint8_t CH_THROTTLE = 2;
const uint8_t CH_YAW = 3;
const uint8_t CH_ARM = 4;    // AUX1
const uint8_t CH_ANGLE = 5;  // AUX2
const uint8_t N_CHANNELS = 8;
const uint16_t RC_MID = 1500;
const uint16_t RC_MIN = 1000;
const uint16_t RC_MAX = 2000;
const uint16_t RC_SPAN = 500;
const uint16_t AUX_HIGH = 1800;
const uint16_t AUX_LOW = 1000;
static inline float clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
static inline uint16_t clampu(int v, uint16_t lo, uint16_t hi) {
  return v < lo ? lo : (v > hi ? hi : (uint16_t)v);
}
// ---------------------------------------------------------------------------
// Wire formats  (must match teleop.py)
// ---------------------------------------------------------------------------
struct __attribute__((packed)) CmdPacket {   // Python "<IBffffB"
  uint32_t seq;
  uint8_t flags;
  float roll;
  float pitch;
  float yaw;
  float throttle;
  uint8_t angleMode;
};
const uint8_t FLAG_ARM = 0x01;
struct __attribute__((packed)) TelemPacket { // Python "<IBffffHIH"
  uint32_t seq;
  uint8_t state;
  float roll;
  float pitch;
  float yaw;
  float throttle;
  uint16_t loopMaxUs;
  uint32_t armFlags;   // Betaflight armingDisableFlags (0 = ready to arm)
  uint16_t lossPerMil; // dropped cmd packets per 1000, rolling estimate
};
enum { STATE_DISARMED = 0, STATE_ARMED = 1, STATE_FAILSAFE = 2 };
// ---------------------------------------------------------------------------
// Shared state between cores
// ---------------------------------------------------------------------------
portMUX_TYPE spLock = portMUX_INITIALIZER_UNLOCKED;
volatile CmdPacket g_cmd = {0, 0, 0, 0, 0, 0, 0};
volatile uint32_t g_lastCmdMs = 0;
volatile uint8_t g_state = STATE_DISARMED;
volatile uint16_t g_loopMaxUs = 0;
volatile uint32_t g_armFlags = 0xFFFFFFFF;  // until first status reply
volatile uint16_t g_lossPerMil = 0;         // rolling loss estimate, per 1000
MspStatusReader g_statusReader;
WiFiUDP udp;
IPAddress g_ground;
volatile bool g_haveGround = false;
// ---------------------------------------------------------------------------
// Core 0: the RC loop
// ---------------------------------------------------------------------------
static void sendChannels(const uint16_t *ch) {
  mspSend(FC, MSP_SET_RAW_RC, (const uint8_t *)ch, N_CHANNELS * 2);
}
void rcLoopTask(void *arg) {
  uint16_t ch[N_CHANNELS];
  uint32_t nextUs = micros();
  for (;;) {
    const uint32_t startUs = micros();
    CmdPacket c;
    uint32_t lastMs;
    portENTER_CRITICAL(&spLock);
    c = *(CmdPacket *)&g_cmd;
    lastMs = g_lastCmdMs;
    portEXIT_CRITICAL(&spLock);
    const bool linkOk = (millis() - lastMs) < LINK_TIMEOUT_MS;
    const bool armReq = linkOk && (c.flags & FLAG_ARM);
    for (uint8_t i = 0; i < N_CHANNELS; i++) ch[i] = RC_MID;
    uint8_t state;
    if (!linkOk) {
      ch[CH_THROTTLE] = RC_MIN;
      ch[CH_ARM] = AUX_LOW;
      ch[CH_ANGLE] = AUX_HIGH;
      state = STATE_FAILSAFE;
    } else if (armReq) {
      ch[CH_ROLL] = clampu(
          RC_MID + (int)(RC_SPAN * clampf(c.roll / MAX_ANGLE_DEG, -1, 1)),
          RC_MIN, RC_MAX);
      ch[CH_PITCH] = clampu(
          RC_MID + (int)(RC_SPAN * clampf(c.pitch / MAX_ANGLE_DEG, -1, 1)),
          RC_MIN, RC_MAX);
      ch[CH_YAW] = clampu(
          RC_MID + (int)(RC_SPAN * clampf(c.yaw, -1, 1)), RC_MIN, RC_MAX);
      ch[CH_THROTTLE] = clampu(
          RC_MIN + (int)(2 * RC_SPAN * clampf(c.throttle, 0, 1)),
          RC_MIN, RC_MAX);
      ch[CH_ARM] = AUX_HIGH;
      ch[CH_ANGLE] = c.angleMode ? AUX_HIGH : AUX_LOW;
      state = STATE_ARMED;
    } else {
      ch[CH_THROTTLE] = RC_MIN;
      ch[CH_ARM] = AUX_LOW;
      ch[CH_ANGLE] = AUX_HIGH;
      state = STATE_DISARMED;
    }
    sendChannels(ch);
    // ~10 Hz: ask the FC why it will or won't arm.
    static uint8_t statusDiv = 0;
    if (++statusDiv >= 10) {
      statusDiv = 0;
      mspSend(FC, MSP_STATUS_EX, nullptr, 0);
    }
    uint32_t flags;
    if (g_statusReader.poll(FC, &flags)) {
      portENTER_CRITICAL(&spLock);
      g_armFlags = flags;
      portEXIT_CRITICAL(&spLock);
    }
    const uint16_t elapsed = (uint16_t)(micros() - startUs);
    portENTER_CRITICAL(&spLock);
    g_state = state;
    if (elapsed > g_loopMaxUs) g_loopMaxUs = elapsed;
    portEXIT_CRITICAL(&spLock);
    if (state == STATE_ARMED) digitalWrite(PIN_LED, HIGH);
    else digitalWrite(PIN_LED,
                      (millis() >> (state == STATE_FAILSAFE ? 6 : 9)) & 1);
    nextUs += LOOP_US;
    int32_t slack = (int32_t)(nextUs - micros());
    if (slack > 0) {
      if (slack > 2000) vTaskDelay(pdMS_TO_TICKS(slack / 1000));
      else delayMicroseconds(slack);
    } else {
      nextUs = micros();
    }
  }
}
// ---------------------------------------------------------------------------
// Core 1: WiFi + UDP + telemetry
// ---------------------------------------------------------------------------
void setup() {
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  FC.begin(MSP_BAUD, SERIAL_8N1, PIN_RX, PIN_TX);
  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(AP_IP, AP_GW, AP_MASK);
  // Pin the channel (avoid congestion) and keep the radio awake continuously
  // (WIFI_PS_NONE) -- power-save parks the radio between beacons and adds
  // periodic latency spikes that read as link instability.
  WiFi.softAP(AP_SSID, AP_PASS, AP_CHANNEL);
  esp_wifi_set_ps(WIFI_PS_NONE);
  udp.begin(LISTEN_PORT);
  g_lastCmdMs = millis() - LINK_TIMEOUT_MS - 1;
  xTaskCreatePinnedToCore(rcLoopTask, "rcLoop", 4096, nullptr,
                          configMAX_PRIORITIES - 1, nullptr, 0);
}
void loop() {
  static uint32_t lastTelemMs = 0;
  int len = udp.parsePacket();
  while (len > 0) {
    if (len >= (int)sizeof(CmdPacket)) {
      CmdPacket in;
      udp.read((uint8_t *)&in, sizeof(in));
      bool fresh;
      portENTER_CRITICAL(&spLock);
      uint32_t prevSeq = g_cmd.seq;
      fresh = (prevSeq == 0) || ((int32_t)(in.seq - prevSeq) > 0);
      if (fresh) {
        // Count sequence gaps as loss. Sender increments seq by 1 per
        // packet, so a jump of N means N-1 were dropped in transit. Fold
        // into an exponential rolling average expressed per-1000.
        if (prevSeq != 0) {
          uint32_t gap = in.seq - prevSeq;
          uint32_t lost = (gap > 1) ? (gap - 1) : 0;
          uint32_t inst = (lost * 1000UL) / gap;
          g_lossPerMil = (uint16_t)(((uint32_t)g_lossPerMil * 15 + inst) >> 4);
        }
        in.roll = clampf(in.roll, -MAX_ANGLE_DEG, MAX_ANGLE_DEG);
        in.pitch = clampf(in.pitch, -MAX_ANGLE_DEG, MAX_ANGLE_DEG);
        in.yaw = clampf(in.yaw, -1.0f, 1.0f);
        in.throttle = clampf(in.throttle, 0.0f, 1.0f);
        *(CmdPacket *)&g_cmd = in;
        g_lastCmdMs = millis();
      }
      portEXIT_CRITICAL(&spLock);
      if (fresh) {
        g_ground = udp.remoteIP();
        g_haveGround = true;
      }
    } else {
      uint8_t junk[64];
      udp.read(junk, sizeof(junk));
    }
    len = udp.parsePacket();
  }
  if (g_haveGround && (millis() - lastTelemMs) >= 50) {
    lastTelemMs = millis();
    TelemPacket t;
    portENTER_CRITICAL(&spLock);
    t.seq = g_cmd.seq;
    t.state = g_state;
    t.roll = g_cmd.roll;
    t.pitch = g_cmd.pitch;
    t.yaw = g_cmd.yaw;
    t.throttle = g_cmd.throttle;
    t.loopMaxUs = g_loopMaxUs;
    t.armFlags = g_armFlags;
    t.lossPerMil = g_lossPerMil;
    g_loopMaxUs = 0;
    portEXIT_CRITICAL(&spLock);
    udp.beginPacket(g_ground, TELEM_PORT);
    udp.write((const uint8_t *)&t, sizeof(t));
    udp.endPacket();
  }
  delay(1);
}
