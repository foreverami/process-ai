/*
 * 진공 건조 챔버 컨트롤러 (아두이노 우노/메가)
 *
 * PC(process_ai/drivers.py)와 115200 baud, 줄 단위 텍스트로 통신한다.
 *   PING                     -> OK
 *   READ                     -> T=25.31 P=101.20 PD=812 DOOR=CLOSED HEAT=0 PUMP=0 VENT=0
 *   DOOR OPEN | DOOR CLOSE   -> OK
 *   PUMP ON   | PUMP OFF     -> OK
 *   VENT OPEN | VENT CLOSE   -> OK
 *   HEAT <설정온도> | HEAT OFF -> OK
 *   거부/오류                 -> ERR <사유>   (사유는 ASCII로만)
 *
 * !! 핀 번호, 센서 변환식, PID 계수는 예시다. 실제 배선과 센서 데이터시트에 맞게 고칠 것 !!
 *
 * PC 소프트웨어와 별개로 펌웨어에서도 한 번 더 막는 안전 장치:
 *  - 워치독: 3초 동안 PC 명령이 없으면 히터와 펌프를 끈다
 *  - 과열 차단: HW_MAX_TEMP_C를 넘으면 히터를 끄고, 충분히 식기 전에는 HEAT 명령을 거부한다
 *  - 열전대 단선: READ에 ERR THERMOCOUPLE로 응답하고 히터를 끈다 (PC는 3회 연속 실패 시 비상 정지)
 *  - 진공 상태에서 문 열기 거부
 *  - 문 이동 제한 시간: 리밋 스위치에 닿지 않으면 모터를 멈춘다 (상태는 UNKNOWN)
 * 히터 전원에는 반드시 별도의 하드웨어 과열 차단 장치(바이메탈 서모스탯, 온도 퓨즈)를 둘 것.
 */
#include <max6675.h>  // 라이브러리 매니저: "MAX6675 library" (Adafruit)

// ---- 핀 배치 (예시) ----
const int PIN_HEATER_SSR = 5;
const int PIN_PUMP_RELAY = 6;
const int PIN_VENT_VALVE = 7;    // 상시 닫힘(NC) 솔레노이드 밸브
const int PIN_DOOR_OPEN = 8;     // 도어 액추에이터 열기 방향 릴레이
const int PIN_DOOR_CLOSE = 9;    // 닫기 방향 릴레이
const int PIN_LIMIT_OPEN = 2;    // 열림 리밋 스위치 (INPUT_PULLUP, 눌리면 LOW)
const int PIN_LIMIT_CLOSED = 3;  // 닫힘 리밋 스위치
const int PIN_TC_SCK = 13, PIN_TC_CS = 10, PIN_TC_SO = 12;
const int PIN_PRESSURE = A0;
const int PIN_PHOTODIODE = A1;   // 트랜스임피던스 증폭기 출력

// 릴레이 모듈이 LOW에서 켜지는 타입이면 LOW로 바꾼다
const int RELAY_ON = HIGH;
const int RELAY_OFF = (RELAY_ON == HIGH) ? LOW : HIGH;

// ---- 한계값 ----
const float HW_MAX_TEMP_C = 210.0;      // PC 쪽 max_temp_c보다 약간 높게
const float OVERTEMP_RESET_C = 190.0;   // 과열 차단 후 이 온도 아래로 식어야 다시 가열 허용
const float DOOR_OPEN_MIN_KPA = 90.0;   // 이보다 낮은 압력(진공)에서는 문 열기 거부
const unsigned long WATCHDOG_MS = 3000;
const unsigned long DOOR_TIMEOUT_MS = 15000;
const unsigned long HEATER_WINDOW_MS = 1000;  // SSR 시간 비례 제어 주기
const unsigned long SENSOR_PERIOD_MS = 250;   // MAX6675는 최소 220ms 간격으로 읽어야 한다

// ---- PID 계수 (예시, 반드시 튜닝) ----
const float KP = 8.0;    // 오차 1°C당 출력 %
const float KI = 0.05;   // 누적 오차(°C·s)당 출력 %
const float KD = 0.0;

MAX6675 thermocouple(PIN_TC_SCK, PIN_TC_CS, PIN_TC_SO);

float tempC = 25.0;
float pressureKpa = 101.3;
int photodiode = 0;
bool thermocoupleFault = false;
bool overTemp = false;

bool heaterEnabled = false;
float setpointC = 0;
float integral = 0, lastError = 0;
float heaterDuty = 0;  // 0~1
unsigned long windowStartMs = 0;

bool pumpOn = false;
bool ventOpen = false;

enum DoorMotion { DOOR_IDLE, DOOR_OPENING, DOOR_CLOSING };
DoorMotion doorMotion = DOOR_IDLE;
unsigned long doorStartMs = 0;

unsigned long lastSensorMs = 0;
unsigned long lastCommandMs = 0;
char line[48];
byte lineLen = 0;

void setup() {
  const int outputs[] = {PIN_PUMP_RELAY, PIN_VENT_VALVE, PIN_DOOR_OPEN, PIN_DOOR_CLOSE};
  for (int pin : outputs) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, RELAY_OFF);
  }
  pinMode(PIN_HEATER_SSR, OUTPUT);
  digitalWrite(PIN_HEATER_SSR, LOW);
  pinMode(PIN_LIMIT_OPEN, INPUT_PULLUP);
  pinMode(PIN_LIMIT_CLOSED, INPUT_PULLUP);
  Serial.begin(115200);
  lastCommandMs = millis();
  windowStartMs = millis();
}

void loop() {
  readSerial();
  unsigned long now = millis();
  if (now - lastSensorMs >= SENSOR_PERIOD_MS) {
    lastSensorMs = now;
    readSensors();
    updatePid();
  }
  driveHeater(now);
  updateDoor(now);
  if (now - lastCommandMs > WATCHDOG_MS) {
    heaterEnabled = false;
    setPump(false);
  }
}

// ---------------------------------------------------------------- 센서

float adcToKpa(int adc) {
  // 예시: 0.5~4.5 V 출력, 0~120 kPa 절대압 센서. 피라니 게이지처럼 로그 출력이면 식을 바꿀 것.
  float volts = adc * 5.0 / 1023.0;
  return constrain((volts - 0.5) / 4.0 * 120.0, 0.0, 200.0);
}

void readSensors() {
  float t = thermocouple.readCelsius();
  thermocoupleFault = isnan(t);
  if (!thermocoupleFault) tempC = t;
  if (thermocoupleFault || tempC > HW_MAX_TEMP_C) {
    if (!thermocoupleFault) overTemp = true;
    heaterEnabled = false;
  }

  pressureKpa = adcToKpa(analogRead(PIN_PRESSURE));

  long sum = 0;
  for (int i = 0; i < 16; i++) sum += analogRead(PIN_PHOTODIODE);
  photodiode = sum / 16;
}

// ---------------------------------------------------------------- 히터

void updatePid() {
  if (!heaterEnabled) {
    heaterDuty = 0;
    integral = 0;
    lastError = 0;
    return;
  }
  const float dt = SENSOR_PERIOD_MS / 1000.0;
  float error = setpointC - tempC;
  integral = constrain(integral + error * dt, -2000.0, 2000.0);
  float derivative = (error - lastError) / dt;
  lastError = error;
  float outputPct = KP * error + KI * integral + KD * derivative;
  heaterDuty = constrain(outputPct, 0.0, 100.0) / 100.0;
}

void driveHeater(unsigned long now) {
  if (now - windowStartMs >= HEATER_WINDOW_MS) windowStartMs = now;
  bool on = heaterEnabled && !overTemp && !thermocoupleFault &&
            (now - windowStartMs) < heaterDuty * HEATER_WINDOW_MS;
  digitalWrite(PIN_HEATER_SSR, on ? HIGH : LOW);
}

// ---------------------------------------------------------------- 펌프·밸브·도어

void setPump(bool on) {
  pumpOn = on;
  digitalWrite(PIN_PUMP_RELAY, on ? RELAY_ON : RELAY_OFF);
}

void setVent(bool open) {
  ventOpen = open;
  digitalWrite(PIN_VENT_VALVE, open ? RELAY_ON : RELAY_OFF);
}

bool limitOpen() { return digitalRead(PIN_LIMIT_OPEN) == LOW; }
bool limitClosed() { return digitalRead(PIN_LIMIT_CLOSED) == LOW; }

void stopDoor() {
  digitalWrite(PIN_DOOR_OPEN, RELAY_OFF);
  digitalWrite(PIN_DOOR_CLOSE, RELAY_OFF);
  doorMotion = DOOR_IDLE;
}

void moveDoor(DoorMotion motion) {
  stopDoor();
  digitalWrite(motion == DOOR_OPENING ? PIN_DOOR_OPEN : PIN_DOOR_CLOSE, RELAY_ON);
  doorMotion = motion;
  doorStartMs = millis();
}

void updateDoor(unsigned long now) {
  if (doorMotion == DOOR_OPENING && limitOpen()) stopDoor();
  else if (doorMotion == DOOR_CLOSING && limitClosed()) stopDoor();
  else if (doorMotion != DOOR_IDLE && now - doorStartMs > DOOR_TIMEOUT_MS) stopDoor();
}

const char* doorState() {
  if (doorMotion != DOOR_IDLE) return "MOVING";
  if (limitClosed()) return "CLOSED";
  if (limitOpen()) return "OPEN";
  return "UNKNOWN";
}

// ---------------------------------------------------------------- 통신

void ok() { Serial.println("OK"); }
void err(const char* reason) {
  Serial.print("ERR ");
  Serial.println(reason);
}

void sendStatus() {
  if (thermocoupleFault) {
    err("THERMOCOUPLE");
    return;
  }
  Serial.print("T=");
  Serial.print(tempC, 2);
  Serial.print(" P=");
  Serial.print(pressureKpa, 2);
  Serial.print(" PD=");
  Serial.print(photodiode);
  Serial.print(" DOOR=");
  Serial.print(doorState());
  Serial.print(" HEAT=");
  Serial.print(heaterEnabled ? 1 : 0);
  Serial.print(" PUMP=");
  Serial.print(pumpOn ? 1 : 0);
  Serial.print(" VENT=");
  Serial.println(ventOpen ? 1 : 0);
}

void handleCommand(const char* cmd) {
  lastCommandMs = millis();
  if (strcmp(cmd, "PING") == 0) {
    ok();
  } else if (strcmp(cmd, "READ") == 0) {
    sendStatus();
  } else if (strcmp(cmd, "DOOR OPEN") == 0) {
    if (pressureKpa < DOOR_OPEN_MIN_KPA) {
      err("VACUUM");
    } else {
      if (!limitOpen()) moveDoor(DOOR_OPENING);
      ok();
    }
  } else if (strcmp(cmd, "DOOR CLOSE") == 0) {
    if (!limitClosed()) moveDoor(DOOR_CLOSING);
    ok();
  } else if (strcmp(cmd, "PUMP ON") == 0) {
    setPump(true);
    ok();
  } else if (strcmp(cmd, "PUMP OFF") == 0) {
    setPump(false);
    ok();
  } else if (strcmp(cmd, "VENT OPEN") == 0) {
    setVent(true);
    ok();
  } else if (strcmp(cmd, "VENT CLOSE") == 0) {
    setVent(false);
    ok();
  } else if (strcmp(cmd, "HEAT OFF") == 0) {
    heaterEnabled = false;
    ok();
  } else if (strncmp(cmd, "HEAT ", 5) == 0) {
    float sp = atof(cmd + 5);
    if (thermocoupleFault) {
      err("THERMOCOUPLE");
    } else if (sp <= 0 || sp > HW_MAX_TEMP_C) {
      err("RANGE");
    } else if (overTemp && tempC > OVERTEMP_RESET_C) {
      err("OVERTEMP");
    } else {
      overTemp = false;
      setpointC = sp;
      heaterEnabled = true;
      ok();
    }
  } else {
    err("UNKNOWN");
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[lineLen] = '\0';
      if (lineLen > 0) handleCommand(line);
      lineLen = 0;
    } else if (lineLen < sizeof(line) - 1) {
      line[lineLen++] = c;
    } else {
      lineLen = 0;  // 너무 긴 줄은 버린다
    }
  }
}
