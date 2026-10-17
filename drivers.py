"""장비 드라이버 인터페이스와 아두이노 시리얼 드라이버.

시리얼 프로토콜 (115200 baud, 줄 단위 ASCII, '\\n'으로 끝남)
firmware/chamber_controller/chamber_controller.ino 와 반드시 같아야 한다.

    PC → 장비                        장비 → PC
    PING                             OK
    READ                             T=25.31 P=101.20 PD=812 DOOR=CLOSED HEAT=0 PUMP=0 VENT=0
    DOOR OPEN | DOOR CLOSE           OK
    PUMP ON   | PUMP OFF             OK
    VENT OPEN | VENT CLOSE           OK
    HEAT <설정온도 °C> | HEAT OFF     OK
    (거부/오류)                       ERR <사유>

    T: 스테이지 온도(°C), P: 챔버 압력(kPa 절대압), PD: 포토다이오드 ADC 값,
    DOOR: OPEN / CLOSED / MOVING / UNKNOWN, HEAT/PUMP/VENT: 1=켜짐(열림) 0=꺼짐(닫힘)

펌웨어는 PC 명령이 일정 시간(3초) 끊기면 히터와 펌프를 끄는 워치독을 가져야 한다.
컨트롤러가 0.5초마다 READ를 보내므로 정상 동작 중에는 워치독이 걸리지 않는다.
"""
import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

DoorState = Literal["open", "closed", "moving", "unknown"]
DOOR_STATES = ("open", "closed", "moving", "unknown")


class DriverError(Exception):
    pass


@dataclass
class Telemetry:
    t: float  # 드라이버 시계 기준 시각(초). 시뮬레이터에서는 배속이 적용된 시간
    temp_c: float
    pressure_kpa: float
    photodiode: float
    door: DoorState
    heater_on: bool
    pump_on: bool
    vent_open: bool

    def to_dict(self) -> dict:
        return asdict(self)


class ChamberDriver(Protocol):
    def now(self) -> float: ...
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def read(self) -> Telemetry: ...
    async def set_door(self, open_: bool) -> None: ...
    async def set_pump(self, on: bool) -> None: ...
    async def set_vent(self, open_: bool) -> None: ...
    async def set_heater(self, setpoint_c: float | None) -> None: ...


def parse_status(line: str, t: float) -> Telemetry:
    try:
        fields = dict(token.split("=", 1) for token in line.split())
        door = fields["DOOR"].lower()
        return Telemetry(
            t=t,
            temp_c=float(fields["T"]),
            pressure_kpa=float(fields["P"]),
            photodiode=float(fields["PD"]),
            door=door if door in DOOR_STATES else "unknown",
            heater_on=fields["HEAT"] == "1",
            pump_on=fields["PUMP"] == "1",
            vent_open=fields["VENT"] == "1",
        )
    except (KeyError, ValueError) as e:
        raise DriverError(f"상태 응답을 해석할 수 없습니다: {line!r}") from e


class ArduinoDriver:
    def __init__(self, port: str, baud: int = 115200, timeout_s: float = 1.0):
        self.port = port
        self.baud = baud
        self.timeout_s = timeout_s
        self._serial = None
        self._lock = asyncio.Lock()
        self._t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._t0

    async def connect(self) -> None:
        import serial  # pyserial

        try:
            self._serial = await asyncio.to_thread(
                serial.Serial, self.port, self.baud, timeout=self.timeout_s
            )
        except serial.SerialException as e:
            raise DriverError(f"시리얼 포트 {self.port}를 열 수 없습니다: {e}") from e
        await asyncio.sleep(2.0)  # 아두이노는 포트가 열리면 리셋된다
        await self._command("PING")

    async def close(self) -> None:
        if self._serial is not None:
            await asyncio.to_thread(self._serial.close)
            self._serial = None

    async def read(self) -> Telemetry:
        line = await self._command("READ", expect_ok=False)
        return parse_status(line, self.now())

    async def set_door(self, open_: bool) -> None:
        await self._command("DOOR OPEN" if open_ else "DOOR CLOSE")

    async def set_pump(self, on: bool) -> None:
        await self._command("PUMP ON" if on else "PUMP OFF")

    async def set_vent(self, open_: bool) -> None:
        await self._command("VENT OPEN" if open_ else "VENT CLOSE")

    async def set_heater(self, setpoint_c: float | None) -> None:
        await self._command("HEAT OFF" if setpoint_c is None else f"HEAT {setpoint_c:.1f}")

    async def _command(self, cmd: str, expect_ok: bool = True) -> str:
        if self._serial is None:
            raise DriverError("장비가 연결되지 않았습니다.")
        async with self._lock:
            line = await asyncio.to_thread(self._roundtrip, cmd)
        if line.startswith("ERR"):
            raise DriverError(f"{cmd} 거부됨: {line[3:].strip()}")
        if expect_ok and line != "OK":
            raise DriverError(f"{cmd}에 예상치 못한 응답: {line!r}")
        return line

    def _roundtrip(self, cmd: str) -> str:
        self._serial.reset_input_buffer()
        self._serial.write((cmd + "\n").encode("ascii"))
        line = self._serial.readline().decode("ascii", errors="replace").strip()
        if not line:
            raise DriverError(f"{cmd}에 응답이 없습니다.")
        return line
