#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>


// wifi network ssid and password
const char *AP_SSID = "drone-shakeout";
const char *AP_PASS = "hover1234";

// channel upon which to transact information
const uint8_t AP_CHANNEL = 6;

// netowork and telemetry ports
const uint16_t LISTEN_PORT = 9000;
const uint16_t TELEM_PORT = 9001;

// esp32 ip adress
IPAddress AP_IP(192, 168, 4, 1);

// gateway adress
IPAddress AP_GW(192, 168, 4, 1);

// subnet mask connection range
IPAddress AP_MASK(255, 255, 255, 0);


// esp32 tx/rx pins
const int PIN_TX = 17; // FC RX4
const int PIN_RX = 16; // FC TX4

// msp usb baud rate
const uint32_t MSP_BAUD = 115200;

// comms via uart pins between fc and esp32
// set the memory address of the fc to uart port
HardwareSerial &FC = Serial2;
const int PIN_LED = 2;

// tx/rx speed between fc and esp32
const uint32_t LOOP_HZ = 100;
const uint32_t LOOP_US = 1000000UL / LOOP_HZ;

// timeout limit
const uint32_t LINK_TIMEOUT_MS = 200;
const float MAX_ANGLE_DEG = 25.0f;

// overriding MSP stick controls with the esp32
const uint8_t MSP_SET_RAW_RC = 200;
const uint8_t MSP_STATUS_EX = 150;

// message structure for the msp protocol
static void mspSend(HardwareSerial &s, uint8_t cmd,
                    const uint8_t *payload, uint8_t size) {uint8_t chk = size ^ cmd;
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
class MspStatusReader {
// parses the msp status message
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

// RC channel IDs
const uint8_t CH_ROLL = 0;
const uint8_t CH_PITCH = 1;
const uint8_t CH_THROTTLE = 2;
const uint8_t CH_YAW = 3;
const uint8_t CH_ARM = 4;
const uint8_t CH_ANGLE = 5;
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

// defines the command packet
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
//defines the telemetry packet
struct __attribute__((packed)) TelemPacket { // Python "<IBffffHIH"
  uint32_t seq;
  uint8_t state;
  float roll;
  float pitch;
  float yaw;
  float throttle;
  uint16_t loopMaxUs;
  uint32_t armFlags;
  uint16_t lossPerMil;
};

// defines states
enum {STATE_DISARMED = 0, STATE_ARMED = 1, STATE_FAILSAFE = 2};

// preventing simultaneous read/write data between the cores with a spinlock
portMUX_TYPE spLock = portMUX_INITIALIZER_UNLOCKED;

// shared variables between the cores
volatile CmdPacket g_cmd = {0, 0, 0, 0, 0, 0, 0}; // latest control packet
volatile uint32_t g_lastCmdMs = 0; // tracks timestamp of last arriving control pakcet
volatile uint8_t g_state = STATE_DISARMED; // starts in the disarmed state
volatile uint16_t g_loopMaxUs = 0; // maximum loop time
volatile uint32_t g_armFlags = 0xFFFFFFFF;  // until first status reply
volatile uint16_t g_lossPerMil = 0; // rolling calculatio nof packet loss per 1000 packets

// communication objects
MspStatusReader g_statusReader; // for processing received telemetry packets
WiFiUDP udp; // open udp network socket
IPAddress g_ground; // ip address of the ground station (my PC)
volatile bool g_haveGround = false; // connection flag

// core 0 is for sending RC commands to the FC
// define the channel send structure
static void sendChannels(const uint16_t *ch) {
  mspSend(FC, MSP_SET_RAW_RC, (const uint8_t *)ch, N_CHANNELS*2);
}
void rcLoopTask(void *arg) {
  uint16_t ch[N_CHANNELS];
  uint32_t nextUs = micros();
  for (;;) {
    // set the loop start timestamp
    const uint32_t startUs = micros();
    CmdPacket c;
    uint32_t lastMs;

    // modifying shared global variables
    portENTER_CRITICAL(&spLock);
    c = *(CmdPacket *)&g_cmd;
    lastMs = g_lastCmdMs;
    portEXIT_CRITICAL(&spLock);

    // check if the link is timing out
    const bool linkOk = (millis() - lastMs) < LINK_TIMEOUT_MS;
    const bool armReq = linkOk && (c.flags & FLAG_ARM);

    // initialize channel values
    for (uint8_t i = 0; i < N_CHANNELS; i++) ch[i] = RC_MID;
    uint8_t state;

    // if the link has timed out set approriate channels to failsafes
    if (!linkOk) {
      ch[CH_THROTTLE] = RC_MIN;
      ch[CH_ARM] = AUX_LOW;
      ch[CH_ANGLE] = AUX_HIGH;
      state = STATE_FAILSAFE;
    } else if (armReq) {
      ch[CH_ROLL] = clampu(
          RC_MID + (int)(RC_SPAN*clampf(c.roll / MAX_ANGLE_DEG, -1, 1)),
          RC_MIN, RC_MAX);
      ch[CH_PITCH] = clampu(
          RC_MID + (int)(RC_SPAN*clampf(c.pitch / MAX_ANGLE_DEG, -1, 1)),
          RC_MIN, RC_MAX);
      ch[CH_YAW] = clampu(
          RC_MID + (int)(RC_SPAN*clampf(c.yaw, -1, 1)), RC_MIN, RC_MAX);
      ch[CH_THROTTLE] = clampu(
          RC_MIN + (int)(2*RC_SPAN*clampf(c.throttle, 0, 1)),
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
    // send the set channels to the FC
    sendChannels(ch);

    // ask the FC why it will or won't arm since I probably missed something
    static uint8_t statusDiv = 0;
    if (++statusDiv >= 10) {
      statusDiv = 0;
      mspSend(FC, MSP_STATUS_EX, nullptr, 0);
    }

    // check the fc status
    uint32_t flags;
    if (g_statusReader.poll(FC, &flags)) {

      // modifying shared globabl arming flags
      portENTER_CRITICAL(&spLock);
      g_armFlags = flags;
      portEXIT_CRITICAL(&spLock);
    }


    // get the elapsed time
    const uint16_t elapsed = (uint16_t)(micros() - startUs);

    // set the globabl state
    portENTER_CRITICAL(&spLock);
    g_state = state;
    if (elapsed > g_loopMaxUs) g_loopMaxUs = elapsed;
    portEXIT_CRITICAL(&spLock);

    // LED for armed state and other states
    if (state == STATE_ARMED) digitalWrite(PIN_LED, HIGH);
    else digitalWrite(PIN_LED,
                      (millis() >> (state == STATE_FAILSAFE ? 6 : 9)) & 1);

    // Check the slack in the loop execution time by comparing the expected time step to the current timestamp
    nextUs += LOOP_US;
    int32_t slack = (int32_t)(nextUs - micros());
    if (slack > 0) {
      // faster the expected we can wait a little bit
      if (slack > 2000) vTaskDelay(pdMS_TO_TICKS(slack / 1000));
      // wait out the clock if we were only slightly faster than expected
      else delayMicroseconds(slack);
    } else {
      // if too slow reset the timeline
      nextUs = micros();
    }
  }
}

void setup() {
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  FC.begin(MSP_BAUD, SERIAL_8N1, PIN_RX, PIN_TX);

  // set the wifi configuration with the specified addresses
  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(AP_IP, AP_GW, AP_MASK);
  WiFi.softAP(AP_SSID, AP_PASS, AP_CHANNEL);
  esp_wifi_set_ps(WIFI_PS_NONE);
  udp.begin(LISTEN_PORT);
  g_lastCmdMs = millis() - LINK_TIMEOUT_MS - 1;

  // start the rc thread
  xTaskCreatePinnedToCore(rcLoopTask, "rcLoop", 4096, nullptr,
                          configMAX_PRIORITIES - 1, nullptr, 0);
}

// main loop
void loop() {
  // start the time and get the number of bytes in the incoming packet
  static uint32_t lastTelemMs = 0;
  int len = udp.parsePacket();

  // read the udp packet
  while (len > 0) {
    // check if we have enough bytes in the data buffer
    if (len >= (int)sizeof(CmdPacket)) {

      // allocate memory to the incoming command packet
      CmdPacket in;

      // read the packet
      udp.read((uint8_t *)&in, sizeof(in));

      // check if the packet is stale
      bool fresh;

      // modifying global variables shared by the cores
      portENTER_CRITICAL(&spLock);
      // get the previous cmd id
      uint32_t prevSeq = g_cmd.seq;
      fresh = (prevSeq == 0) || ((int32_t)(in.seq - prevSeq) > 0);
      if (fresh) {
        // get the number of lost packets
        if (prevSeq != 0) {
          uint32_t gap = in.seq - prevSeq;
          uint32_t lost = (gap > 1) ? (gap - 1) : 0;
          uint32_t inst = (lost * 1000UL) / gap;
          g_lossPerMil = (uint16_t)(((uint32_t)g_lossPerMil * 15 + inst) >> 4);
        }

        // read the attitude command
        in.roll = clampf(in.roll, -MAX_ANGLE_DEG, MAX_ANGLE_DEG);
        in.pitch = clampf(in.pitch, -MAX_ANGLE_DEG, MAX_ANGLE_DEG);
        in.yaw = clampf(in.yaw, -1.0f, 1.0f);

        // read teh thrust
        in.throttle = clampf(in.throttle, 0.0f, 1.0f);
        
        // copy data from local in to global command variable
        *(CmdPacket *)&g_cmd = in;
        g_lastCmdMs = millis();
      }
      portEXIT_CRITICAL(&spLock);

      // set ground station udp and flag based if fresh connection
      if (fresh) {
        g_ground = udp.remoteIP();
        g_haveGround = true;
      }
    } else {
      uint8_t junk[64];
      udp.read(junk, sizeof(junk));
    }

    // get the number of bytes in the next packet
    len = udp.parsePacket();
  }

  // pack the current flight status and transmit to ground
  // check for ground station connection and only execute telemetry every 50ms
  if (g_haveGround && (millis() - lastTelemMs) >= 50) {
    lastTelemMs = millis();
    TelemPacket t;

    // modifying global variables
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

    // transmit the packet back to the ground station
    udp.beginPacket(g_ground, TELEM_PORT);
    udp.write((const uint8_t *)&t, sizeof(t));
    udp.endPacket();
  }
  delay(1);
}
