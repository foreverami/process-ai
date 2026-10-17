"""승인된 레시피대로 장비를 단계별로 구동하는 공정 상태 머신.

드라이버에는 이 컨트롤러의 루프 하나만 접근한다. 매 주기마다
센서 읽기 → 안전 점검 → 온도·압력 제어 → 단계 전환 판단 → 기록·방송 순서로 동작한다.
시작/정지/개방 요청은 상태만 바꾸고, 실제 장비 명령은 루프 안에서 내린다.

정상 흐름: 문 닫기 → 진공 배기 → 승온 → 건조(종료점 감시) → 냉각 → 벤트 → 문 열기
이상 발생 시: 히터 끔, 펌프 끔, 벤트 열기. 문은 닫힌 채로 두고 작업자가 안전 개방한다.
"""
import asyncio
import csv
import json
import re
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path

from .config import RUNS_DIR, TICK_S, EquipmentLimits
from .drivers import ChamberDriver, DriverError, Telemetry
from .endpoint import DryEndpointDetector
from .recipe import Issue, Recipe, has_errors


class Phase(str, Enum):
    IDLE = "IDLE"
    CLOSING_DOOR = "CLOSING_DOOR"
    PUMPING = "PUMPING"
    HEATING = "HEATING"
    DRYING = "DRYING"
    COOLING = "COOLING"
    VENTING = "VENTING"
    OPENING_DOOR = "OPENING_DOOR"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


PHASE_LABELS = {
    Phase.IDLE: "대기",
    Phase.CLOSING_DOOR: "문 닫기",
    Phase.PUMPING: "진공 배기",
    Phase.HEATING: "승온",
    Phase.DRYING: "건조",
    Phase.COOLING: "냉각",
    Phase.VENTING: "벤트",
    Phase.OPENING_DOOR: "문 열기",
    Phase.COMPLETE: "완료",
    Phase.ABORTED: "비상 정지",
}

RESTING = {Phase.IDLE, Phase.COMPLETE, Phase.ABORTED}
SEALED = {Phase.PUMPING, Phase.HEATING, Phase.DRYING, Phase.COOLING}  # 문이 닫혀 있어야 하는 단계
HOT = {Phase.CLOSING_DOOR, Phase.PUMPING, Phase.HEATING, Phase.DRYING}  # 전체 공정 시간 제한 대상

TEMP_REACHED_TOLERANCE_C = 1.0
VENTED_MARGIN_KPA = 2.0
PUMP_RESTART_RATIO = 0.8  # 진공 유지 중 목표 압력의 80% 아래로 내려가면 펌프를 쉰다
MAX_READ_FAILURES = 3
MIN_SIGNAL_CHANGE_PCT = 2.0  # 공정 중 포토다이오드 변화가 이보다 작으면 센서 점검 경고


class ProcessError(Exception):
    pass


class RunLog:
    """공정 1회의 레시피, 센서 기록, 이벤트, 결과 요약을 runs/<run_id>/ 에 남긴다."""

    FIELDS = ["t", "phase", "temp_c", "setpoint_c", "pressure_kpa", "photodiode",
              "door", "heater_on", "pump_on", "vent_open"]

    def __init__(self, runs_dir: Path, recipe: Recipe, approved_by: str, issues: list[Issue]):
        slug = re.sub(r"[^\w-]+", "_", recipe.name).strip("_")[:40] or "run"
        base = f"{datetime.now():%Y%m%d-%H%M%S}_{slug}"
        self.run_id, n = base, 1
        while (runs_dir / self.run_id).exists():
            n += 1
            self.run_id = f"{base}-{n}"
        self.dir = runs_dir / self.run_id
        self.dir.mkdir(parents=True)
        self.started_at = datetime.now()

        (self.dir / "recipe.json").write_text(json.dumps({
            "recipe": recipe.model_dump(),
            "approved_by": approved_by,
            "approved_at": self.started_at.isoformat(timespec="seconds"),
            "validation": [i.model_dump() for i in issues],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        self._csv_file = open(self.dir / "telemetry.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(self.FIELDS)
        self._events_file = open(self.dir / "events.jsonl", "a", encoding="utf-8")

    def telemetry(self, tel: Telemetry, phase: Phase, setpoint_c: float | None) -> None:
        row = tel.to_dict() | {"phase": phase.value, "setpoint_c": setpoint_c}
        self._csv.writerow([row[k] for k in self.FIELDS])
        self._csv_file.flush()

    def event(self, event: dict) -> None:
        self._events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events_file.flush()

    def finish(self, summary: dict) -> None:
        summary = summary | {
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
        }
        (self.dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._csv_file.close()
        self._events_file.close()


class ProcessController:
    def __init__(
        self,
        driver: ChamberDriver,
        limits: EquipmentLimits,
        validate: Callable[[Recipe], list[Issue]],
        tick_s: float = TICK_S,
        runs_dir: Path = RUNS_DIR,
    ):
        self.driver = driver
        self.limits = limits
        self.validate = validate  # 온톨로지 SHACL 검증 (KnowledgeBase.validate)
        self.tick_s = tick_s
        self.runs_dir = Path(runs_dir)

        self.phase = Phase.IDLE
        self.plan: list[Phase] = []
        self.recipe: Recipe | None = None
        self.run: RunLog | None = None
        self.last_run_id: str | None = None
        self.last_summary: dict | None = None
        self.events: list[dict] = []
        self.last: Telemetry | None = None
        self.connection_error: str | None = "장비 연결 대기 중"
        self.detector: DryEndpointDetector | None = None

        self._needs_entry = False
        self._manual_open = False
        self._pending_abort: str | None = None
        self._setpoint: float | None = None
        self._ramp: tuple[float, float] | None = None  # (승온 시작 시각, 시작 온도)
        self._hold_pressure = False
        self._phase_started = 0.0
        self._deadline: float | None = None
        self._run_started = 0.0
        self._drying: dict = {}
        self._stats: dict = {}
        self._photodiode_start = 0.0
        self._read_failures = 0
        self._stop = False
        self._subscribers: set[asyncio.Queue] = set()

    # ------------------------------------------------------------------ 요청

    def start(self, recipe: Recipe, approved_by: str) -> str:
        if self.phase not in RESTING:
            raise ProcessError("이미 공정이 진행 중입니다.")
        if not approved_by.strip():
            raise ProcessError("승인자 이름이 필요합니다.")
        issues = self.validate(recipe)
        if has_errors(issues):
            errors = "; ".join(i.message for i in issues if i.level == "error")
            raise ProcessError(f"레시피 검증 오류: {errors}")
        if self.last is None or self.connection_error:
            raise ProcessError(f"장비 상태를 읽을 수 없어 시작할 수 없습니다. ({self.connection_error})")

        middle = [Phase.PUMPING, Phase.HEATING]
        if recipe.order == "heat_first":
            middle.reverse()
        self.plan = [Phase.CLOSING_DOOR, *middle, Phase.DRYING, Phase.COOLING,
                     Phase.VENTING, Phase.OPENING_DOOR]
        self.recipe = recipe
        self.detector = DryEndpointDetector(
            recipe.endpoint.stable_window_s, recipe.endpoint.stable_band_pct
        )
        self.events = []
        self._drying = {}
        self._stats = {"max_temp_c": self.last.temp_c, "min_pressure_kpa": self.last.pressure_kpa}
        self._photodiode_start = self.last.photodiode
        self._manual_open = False
        self._pending_abort = None
        self._run_started = self.last.t
        self.run = RunLog(self.runs_dir, recipe, approved_by, issues)
        self.last_run_id = self.run.run_id
        self.last_summary = None
        self._set_phase(self.plan[0])
        self._event("info", f"'{recipe.name}' 공정 승인 (승인자: {approved_by})")
        return self.run.run_id

    def abort(self, reason: str = "작업자 비상 정지") -> bool:
        if self.phase in RESTING:
            return False
        self._pending_abort = reason
        return True

    def request_open(self) -> None:
        """대기/완료/비상 정지 상태에서 벤트 후 문을 연다."""
        if self.phase not in RESTING:
            raise ProcessError("공정이 진행 중일 때는 문을 열 수 없습니다.")
        if self.last is None or self.connection_error:
            raise ProcessError("장비 상태를 읽을 수 없습니다.")
        if self.last.door == "open":
            raise ProcessError("문이 이미 열려 있습니다.")
        if self.last.temp_c > self.limits.safe_open_temp_c:
            raise ProcessError(
                f"스테이지 온도 {self.last.temp_c:.1f}°C — "
                f"{self.limits.safe_open_temp_c:.0f}°C 이하로 식은 뒤 여세요."
            )
        self.plan = [Phase.VENTING, Phase.OPENING_DOOR]
        self._manual_open = True
        self._pending_abort = None
        self._set_phase(Phase.VENTING)
        self._event("info", "안전 개방 요청")

    def stop(self) -> None:
        self._stop = True

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def status(self) -> dict:
        return {
            "phase": self.phase.value,
            "phase_label": PHASE_LABELS[self.phase],
            "plan": [p.value for p in self.plan],
            "phase_labels": {p.value: label for p, label in PHASE_LABELS.items()},
            "manual_open": self._manual_open,
            "recipe": self.recipe.model_dump() if self.recipe else None,
            "run_id": self.last_run_id,
            "summary": self.last_summary,
            "events": self.events[-100:],
            "telemetry": self._telemetry_message(self.last) if self.last else None,
            "connection_error": self.connection_error,
            "limits": self.limits.model_dump(),
        }

    # ------------------------------------------------------------------ 루프

    async def run_forever(self) -> None:
        await self._connect()
        try:
            while not self._stop:
                await self.tick()
                await asyncio.sleep(self.tick_s)
        finally:
            if self.phase not in RESTING:
                await self._abort("서버 종료")
            await self.driver.close()

    async def _connect(self) -> None:
        while not self._stop:
            try:
                await self.driver.connect()
                self.connection_error = None
                return
            except Exception as e:
                self.connection_error = f"장비 연결 실패: {e}"
                self._publish({"type": "connection", "error": self.connection_error})
                await asyncio.sleep(3)

    async def tick(self) -> None:
        try:
            tel = await self.driver.read()
        except Exception as e:
            self._read_failures += 1
            self.connection_error = f"센서 읽기 실패: {e}"
            self._publish({"type": "connection", "error": self.connection_error})
            if self.phase not in RESTING and self._read_failures >= MAX_READ_FAILURES:
                await self._abort(self.connection_error)
            return
        self._read_failures = 0
        self.connection_error = None
        self.last = tel

        if self.phase not in RESTING:
            try:
                await self._step(tel)
            except DriverError as e:
                await self._abort(f"장비 명령 실패: {e}")

        if self.run is not None:
            self.run.telemetry(tel, self.phase, self._setpoint)
        self._publish(self._telemetry_message(tel))

    async def _step(self, tel: Telemetry) -> None:
        if self._pending_abort:
            await self._abort(self._pending_abort)
            return
        if self.detector is not None and not self._manual_open:
            self.detector.update(tel.t, tel.photodiode)
            self._stats["max_temp_c"] = max(self._stats["max_temp_c"], tel.temp_c)
            self._stats["min_pressure_kpa"] = min(self._stats["min_pressure_kpa"], tel.pressure_kpa)
        if self._needs_entry:
            self._needs_entry = False
            await self._on_enter(tel)
        violation = self._safety_violation(tel)
        if violation:
            await self._abort(violation)
            return
        await self._regulate(tel)
        if self._phase_done(tel):
            await self._advance(tel)

    # ------------------------------------------------------------------ 단계

    def _set_phase(self, phase: Phase) -> None:
        self.phase = phase
        self._needs_entry = True

    async def _advance(self, tel: Telemetry) -> None:
        index = self.plan.index(self.phase)
        if index + 1 < len(self.plan):
            self.phase = self.plan[index + 1]
            await self._on_enter(tel)
        else:
            await self._finish(tel)

    async def _on_enter(self, tel: Telemetry) -> None:
        lim, r, now = self.limits, self.recipe, tel.t
        self._phase_started = now
        self._deadline = None
        match self.phase:
            case Phase.CLOSING_DOOR:
                await self.driver.set_door(False)
                self._deadline = now + lim.door_timeout_s
            case Phase.PUMPING:
                await self.driver.set_vent(False)
                await self.driver.set_pump(True)
                self._deadline = now + r.vacuum.pump_timeout_min * 60
            case Phase.HEATING:
                self._ramp = (now, tel.temp_c)
                ramp_min = max(r.heating.target_c - tel.temp_c, 0) / r.heating.ramp_c_per_min
                self._deadline = now + (ramp_min + lim.heat_timeout_extra_min) * 60
            case Phase.DRYING:
                self._drying = {"started": now}
            case Phase.COOLING:
                self._ramp = None
                self._setpoint = None
                await self.driver.set_heater(None)
                self._deadline = now + lim.cool_timeout_min * 60
            case Phase.VENTING:
                self._hold_pressure = False
                await self.driver.set_heater(None)
                await self.driver.set_pump(False)
                await self.driver.set_vent(True)
                self._deadline = now + lim.vent_timeout_s
            case Phase.OPENING_DOOR:
                if tel.temp_c > lim.safe_open_temp_c:
                    raise DriverError(f"스테이지 온도 {tel.temp_c:.1f}°C가 높아 문을 열지 않습니다.")
                await self.driver.set_door(True)
                self._deadline = now + lim.door_timeout_s
        self._event("info", f"{PHASE_LABELS[self.phase]} 시작")

    def _phase_done(self, tel: Telemetry) -> bool:
        r = self.recipe
        match self.phase:
            case Phase.CLOSING_DOOR:
                return tel.door == "closed"
            case Phase.PUMPING:
                if tel.pressure_kpa <= r.vacuum.target_kpa:
                    self._hold_pressure = True
                    return True
                return False
            case Phase.HEATING:
                return tel.temp_c >= r.heating.target_c - TEMP_REACHED_TOLERANCE_C
            case Phase.DRYING:
                return self._drying_done(tel)
            case Phase.COOLING:
                return tel.temp_c <= r.finish.cool_to_c
            case Phase.VENTING:
                return tel.pressure_kpa >= self.limits.atm_pressure_kpa - VENTED_MARGIN_KPA
            case Phase.OPENING_DOOR:
                return tel.door == "open"
        return False

    def _drying_done(self, tel: Telemetry) -> bool:
        rule = self.recipe.endpoint
        elapsed = tel.t - self._drying["started"]
        if elapsed >= rule.min_hold_min * 60 and self.detector.is_stable():
            self._drying.update(ended=tel.t, endpoint_detected=True)
            self._event("success", "포토다이오드 신호가 안정됨 — 용매 증발 완료로 판단")
            start = self._photodiode_start
            change = abs(tel.photodiode - start) / max(abs(start), 1e-9) * 100
            if change < MIN_SIGNAL_CHANGE_PCT:
                self._event(
                    "warning",
                    f"공정 동안 포토다이오드 신호 변화가 {change:.1f}%뿐입니다. "
                    "센서 위치나 시편 상태를 확인하세요.",
                )
            return True
        if elapsed >= rule.max_hold_min * 60:
            self._drying.update(ended=tel.t, endpoint_detected=False)
            self._event("warning", "최대 유지 시간 도달 — 종료점을 검출하지 못했지만 건조를 마칩니다")
            return True
        return False

    async def _regulate(self, tel: Telemetry) -> None:
        if self._ramp is not None:
            t0, temp0 = self._ramp
            heating = self.recipe.heating
            setpoint = min(heating.target_c, temp0 + heating.ramp_c_per_min * (tel.t - t0) / 60)
            heater_dropped = self._setpoint is not None and not tel.heater_on
            if heater_dropped:
                self._event("warning", "장비에서 히터가 꺼져 있어 다시 켭니다 (통신 지연으로 워치독이 동작했을 수 있음)")
            if heater_dropped or self._setpoint is None or abs(setpoint - self._setpoint) >= 0.1:
                await self.driver.set_heater(round(setpoint, 1))
                self._setpoint = setpoint
        if self.phase == Phase.PUMPING and not tel.pump_on and tel.t > self._phase_started:
            await self.driver.set_pump(True)  # 워치독 등으로 펌프가 꺼졌으면 다시 켠다
        if self._hold_pressure:
            target = self.recipe.vacuum.target_kpa
            if tel.pressure_kpa > target and not tel.pump_on:
                await self.driver.set_pump(True)
            elif tel.pressure_kpa < target * PUMP_RESTART_RATIO and tel.pump_on:
                await self.driver.set_pump(False)

    def _safety_violation(self, tel: Telemetry) -> str | None:
        lim = self.limits
        if tel.temp_c > lim.max_temp_c:
            return f"스테이지 온도 {tel.temp_c:.1f}°C가 장비 최대 온도 {lim.max_temp_c:.0f}°C를 넘었습니다."
        if self.phase in HOT and not self._manual_open:
            limit = self.recipe.heating.target_c + lim.overshoot_margin_c
            if tel.temp_c > limit:
                return f"스테이지 온도 {tel.temp_c:.1f}°C가 허용치 {limit:.0f}°C를 넘었습니다 (과열)."
            if tel.t - self._run_started > lim.max_process_min * 60:
                return f"최대 공정 시간 {lim.max_process_min:.0f}분을 넘었습니다."
        if self.phase in SEALED and tel.door != "closed":
            return f"공정 중 문 상태가 '{tel.door}'로 바뀌었습니다."
        if self._deadline is not None and tel.t > self._deadline:
            return (
                f"{PHASE_LABELS[self.phase]} 단계가 제한 시간 안에 끝나지 않았습니다 "
                f"(온도 {tel.temp_c:.1f}°C, 압력 {tel.pressure_kpa:.2f} kPa, 문 {tel.door})."
            )
        return None

    # ------------------------------------------------------------------ 종료

    async def _finish(self, tel: Telemetry) -> None:
        await self.driver.set_vent(False)
        self._deadline = None
        if self._manual_open:
            self._manual_open = False
            self.phase = Phase.IDLE
            self._event("success", "문 개방 완료")
            return
        self.phase = Phase.COMPLETE
        self._event("success", "공정 완료")
        self._close_run(tel, "COMPLETE", None)

    async def _abort(self, reason: str) -> None:
        self._pending_abort = None
        self._ramp = None
        self._setpoint = None
        self._hold_pressure = False
        self._deadline = None
        for name, command in (
            ("히터 끄기", lambda: self.driver.set_heater(None)),
            ("펌프 끄기", lambda: self.driver.set_pump(False)),
            ("벤트 열기", lambda: self.driver.set_vent(True)),
        ):
            try:
                await command()
            except Exception as e:
                self._event("error", f"{name} 명령 실패: {e}")
        self._event("error", f"비상 정지: {reason}")  # 멈춘 단계가 기록되도록 상태 전환 전에 남긴다
        self.phase = Phase.ABORTED
        self._manual_open = False
        self._close_run(self.last, "ABORTED", reason)

    def _close_run(self, tel: Telemetry | None, result: str, reason: str | None) -> None:
        if self.run is None:
            return
        drying_min = None
        if "started" in self._drying:
            end = self._drying.get("ended", tel.t if tel else self._drying["started"])
            drying_min = round((end - self._drying["started"]) / 60, 2)
        self.last_summary = {
            "run_id": self.run.run_id,
            "result": result,
            "reason": reason,
            "endpoint_detected": self._drying.get("endpoint_detected"),
            "drying_min": drying_min,
            "process_min": round((tel.t - self._run_started) / 60, 2) if tel else None,
            **{k: round(v, 2) for k, v in self._stats.items()},
            "final_photodiode": tel.photodiode if tel else None,
        }
        self.run.finish(self.last_summary)
        self.run = None

    # ------------------------------------------------------------------ 방송

    def _event(self, level: str, message: str) -> None:
        event = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "t": self.last.t if self.last else None,
            "phase": self.phase.value,
            "level": level,
            "message": message,
        }
        self.events.append(event)
        if self.run is not None:
            self.run.event(event)
        self._publish({"type": "event", **event})

    def _telemetry_message(self, tel: Telemetry) -> dict:
        active = self.phase not in RESTING
        return {
            "type": "telemetry",
            **tel.to_dict(),
            "phase": self.phase.value,
            "setpoint_c": self._setpoint,
            "phase_elapsed_s": tel.t - self._phase_started if active else None,
            "run_elapsed_s": tel.t - self._run_started if active and not self._manual_open else None,
            "endpoint": self.detector.snapshot() if self.detector and active else None,
        }

    def _publish(self, message: dict) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass
