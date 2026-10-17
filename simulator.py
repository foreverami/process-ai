"""실물 장비 없이 공정을 돌려보기 위한 가상 챔버.

ArduinoDriver와 같은 인터페이스를 가진다. 물리 모델은 시연·테스트용 근사치다.
- 압력: 펌프 배기와 벤트를 1차 지연으로 모사, 밀폐 상태에서는 약한 누설
- 온도: 히터가 설정 온도를 따라가고, 꺼지면 자연 냉각
- 용매: 온도가 높고 압력이 낮을수록 빨리 증발
- 포토다이오드: 용매가 줄면서 신호가 변하고(간섭으로 진동), 다 마르면 평탄해짐

speed: 실제 시간 대비 배속. step_s를 주면 read() 한 번마다 그만큼 시간이 흐른다(테스트용).
faults: 고장 주입. "heater_runaway", "leak", "door_stuck"
"""
import math
import random
import time

from .drivers import DriverError, Telemetry

ATM_KPA = 101.3
AMBIENT_C = 25.0
BASE_PRESSURE_KPA = 0.3
PUMP_TAU_S = 15.0
VENT_TAU_S = 3.0
LEAK_KPA_PER_S = 0.01
HEAT_TAU_S = 15.0
MAX_HEAT_RATE_C_PER_S = 1.0
COOL_TAU_S = 400.0
DOOR_TRAVEL_S = 3.0
SUBSTEP_S = 0.5


class SimulatedChamber:
    def __init__(
        self,
        speed: float = 20.0,
        step_s: float | None = None,
        seed: int | None = None,
        evaporation_k0: float = 3.8e-5,
        noise_pct: float = 0.1,
    ):
        self.speed = speed
        self.step_s = step_s
        self.evaporation_k0 = evaporation_k0
        self.noise_pct = noise_pct
        self.faults: set[str] = set()
        self._rng = random.Random(seed)
        self._t0 = time.monotonic()
        self._t = 0.0
        self.reset()

    def reset(self) -> None:
        """새 시편을 올려놓은 상태로 되돌린다."""
        self.temp_c = AMBIENT_C
        self.pressure_kpa = ATM_KPA
        self.door = "open"
        self._door_target = "open"
        self._door_arrive_t = 0.0
        self.setpoint_c: float | None = None
        self.pump_on = False
        self.vent_open = False
        self.solvent = 1.0  # 남은 용매 비율 (1 → 0)

    def now(self) -> float:
        if self.step_s is not None:
            return self._t
        return (time.monotonic() - self._t0) * self.speed

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def read(self) -> Telemetry:
        if self.step_s is not None:
            target = self._t + self.step_s
        else:
            target = self.now()
        while self._t < target:
            dt = min(SUBSTEP_S, target - self._t)
            self._advance(dt)
            self._t += dt
        return Telemetry(
            t=self._t,
            temp_c=round(self.temp_c, 2),
            pressure_kpa=round(self.pressure_kpa, 3),
            photodiode=round(self._photodiode(), 1),
            door=self.door,
            heater_on=self.setpoint_c is not None,
            pump_on=self.pump_on,
            vent_open=self.vent_open,
        )

    async def set_door(self, open_: bool) -> None:
        target = "open" if open_ else "closed"
        if self.door == target:
            return
        if open_ and self.pressure_kpa < ATM_KPA - 10:
            raise DriverError("챔버가 진공 상태라 문을 열 수 없습니다.")
        self.door = "moving"
        self._door_target = target
        self._door_arrive_t = self._t + DOOR_TRAVEL_S

    async def set_pump(self, on: bool) -> None:
        self.pump_on = on

    async def set_vent(self, open_: bool) -> None:
        self.vent_open = open_

    async def set_heater(self, setpoint_c: float | None) -> None:
        self.setpoint_c = setpoint_c

    def _advance(self, dt: float) -> None:
        if self.door == "moving" and "door_stuck" not in self.faults and self._t >= self._door_arrive_t:
            self.door = self._door_target

        sealed = self.door == "closed"
        if self.vent_open or not sealed:
            self.pressure_kpa += (ATM_KPA - self.pressure_kpa) * (1 - math.exp(-dt / VENT_TAU_S))
        elif self.pump_on:
            floor = 50.0 if "leak" in self.faults else BASE_PRESSURE_KPA
            self.pressure_kpa = floor + (self.pressure_kpa - floor) * math.exp(-dt / PUMP_TAU_S)
        else:
            self.pressure_kpa = min(ATM_KPA, self.pressure_kpa + LEAK_KPA_PER_S * dt)

        passive = -(self.temp_c - AMBIENT_C) / COOL_TAU_S
        if "heater_runaway" in self.faults:
            rate = MAX_HEAT_RATE_C_PER_S
        elif self.setpoint_c is not None and self.temp_c < self.setpoint_c:
            rate = max(min((self.setpoint_c - self.temp_c) / HEAT_TAU_S, MAX_HEAT_RATE_C_PER_S), passive)
        else:
            rate = passive
        self.temp_c += rate * dt

        vacuum_boost = 1 + 3 * (1 - self.pressure_kpa / ATM_KPA)
        k = self.evaporation_k0 * 2 ** ((self.temp_c - AMBIENT_C) / 10) * vacuum_boost
        self.solvent = max(0.0, self.solvent - k * math.sqrt(self.solvent) * dt)

    def _photodiode(self) -> float:
        m = self.solvent
        signal = 600 + 250 * (1 - m) + 60 * math.sin(2 * math.pi * 4 * m)
        return signal * (1 + self._rng.gauss(0, self.noise_pct / 100))
