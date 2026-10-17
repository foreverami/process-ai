import asyncio
import json

import pytest

from process_ai.controller import Phase, ProcessController, ProcessError
from process_ai.recipe import Recipe
from process_ai.simulator import SimulatedChamber

DONE = (Phase.COMPLETE, Phase.ABORTED, Phase.IDLE)


def make_recipe(**overrides) -> Recipe:
    data = {
        "name": "PEDOT 박막 건조",
        "vacuum": {"target_kpa": 5},
        "heating": {"target_c": 80, "ramp_c_per_min": 10},
        "endpoint": {"min_hold_min": 1, "max_hold_min": 30, "stable_window_s": 30, "stable_band_pct": 0.5},
    }
    data.update(overrides)
    return Recipe.model_validate(data)


def make_controller(tmp_path, kb, **sim_kwargs):
    sim = SimulatedChamber(step_s=2.0, seed=1, **sim_kwargs)
    return sim, ProcessController(sim, kb.limits(), kb.validate, tick_s=0, runs_dir=tmp_path)


async def tick_until(ctrl, phases, max_ticks=5000, on_tick=None):
    for _ in range(max_ticks):
        await ctrl.tick()
        if on_tick:
            on_tick()
        if ctrl.phase in phases:
            return
    raise AssertionError(f"{ctrl.phase}에서 멈춤: {ctrl.events[-3:]}")


def started_phases(ctrl):
    return [e["phase"] for e in ctrl.events if e["message"].endswith(" 시작")]


def last_error(ctrl):
    return next(e["message"] for e in reversed(ctrl.events) if e["level"] == "error")


def test_full_run_closes_pumps_heats_dries_and_opens(tmp_path, kb):
    sim, ctrl = make_controller(tmp_path, kb)
    solvent_at_endpoint = []

    def watch():
        if ctrl.phase == Phase.COOLING and not solvent_at_endpoint:
            solvent_at_endpoint.append(sim.solvent)

    async def scenario():
        await ctrl.tick()
        run_id = ctrl.start(make_recipe(), "홍길동")
        await tick_until(ctrl, DONE, on_tick=watch)
        return run_id

    run_id = asyncio.run(scenario())

    assert ctrl.phase == Phase.COMPLETE, ctrl.events[-3:]
    assert started_phases(ctrl) == [
        "CLOSING_DOOR", "PUMPING", "HEATING", "DRYING", "COOLING", "VENTING", "OPENING_DOOR"
    ]
    assert solvent_at_endpoint[0] < 0.01
    assert sim.door == "open"
    assert sim.setpoint_c is None and not sim.pump_on and not sim.vent_open
    assert sim.temp_c <= 50

    summary = json.loads((tmp_path / run_id / "summary.json").read_text(encoding="utf-8"))
    assert summary["result"] == "COMPLETE"
    assert summary["endpoint_detected"] is True
    assert summary["max_temp_c"] < 80 + 3
    assert summary["min_pressure_kpa"] <= 5
    rows = (tmp_path / run_id / "telemetry.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("t,phase,temp_c") and len(rows) > 100
    saved = json.loads((tmp_path / run_id / "recipe.json").read_text(encoding="utf-8"))
    assert saved["approved_by"] == "홍길동"


def test_heat_first_order_heats_before_pumping(tmp_path, kb):
    _, ctrl = make_controller(tmp_path, kb)

    async def scenario():
        await ctrl.tick()
        ctrl.start(make_recipe(order="heat_first"), "홍길동")
        await tick_until(ctrl, DONE)

    asyncio.run(scenario())
    assert ctrl.phase == Phase.COMPLETE
    assert started_phases(ctrl)[:4] == ["CLOSING_DOOR", "HEATING", "PUMPING", "DRYING"]


def test_vacuum_is_held_near_target_while_drying(tmp_path, kb):
    _, ctrl = make_controller(tmp_path, kb)
    pressures = []

    def watch():
        if ctrl.phase == Phase.DRYING:
            pressures.append(ctrl.last.pressure_kpa)

    async def scenario():
        await ctrl.tick()
        ctrl.start(make_recipe(vacuum={"target_kpa": 20}), "홍길동")
        await tick_until(ctrl, DONE, on_tick=watch)

    asyncio.run(scenario())
    assert pressures and max(pressures) < 20 * 1.05 and min(pressures) > 20 * 0.7


def test_endpoint_timeout_finishes_with_warning(tmp_path, kb):
    _, ctrl = make_controller(tmp_path, kb, noise_pct=3.0)  # 잡음이 커서 신호가 안정되지 않음

    async def scenario():
        await ctrl.tick()
        ctrl.start(make_recipe(endpoint={"min_hold_min": 1, "max_hold_min": 3}), "홍길동")
        await tick_until(ctrl, DONE)

    asyncio.run(scenario())
    assert ctrl.phase == Phase.COMPLETE
    assert ctrl.last_summary["endpoint_detected"] is False
    assert any("최대 유지 시간" in e["message"] for e in ctrl.events if e["level"] == "warning")


def test_flat_photodiode_signal_warns(tmp_path, kb):
    sim, ctrl = make_controller(tmp_path, kb)
    sim.solvent = 0.0  # 이미 마른 시편 (또는 센서가 빛을 못 받는 경우)

    async def scenario():
        await ctrl.tick()
        ctrl.start(make_recipe(), "홍길동")
        await tick_until(ctrl, DONE)

    asyncio.run(scenario())
    assert ctrl.phase == Phase.COMPLETE
    assert any("신호 변화" in e["message"] for e in ctrl.events if e["level"] == "warning")


@pytest.mark.parametrize(
    "fault, phase_expected, message",
    [
        ("heater_runaway", None, "과열"),
        ("leak", "PUMPING", "진공 배기 단계가 제한 시간"),
        ("door_stuck", "CLOSING_DOOR", "문 닫기 단계가 제한 시간"),
    ],
)
def test_faults_abort_to_safe_state(tmp_path, kb, fault, phase_expected, message):
    sim, ctrl = make_controller(tmp_path, kb)
    sim.faults.add(fault)

    async def scenario():
        await ctrl.tick()
        ctrl.start(make_recipe(), "홍길동")
        await tick_until(ctrl, DONE)

    asyncio.run(scenario())
    assert ctrl.phase == Phase.ABORTED
    assert message in last_error(ctrl)
    if phase_expected:
        assert ctrl.events[-1]["phase"] == phase_expected
    assert sim.setpoint_c is None and not sim.pump_on and sim.vent_open
    assert ctrl.last_summary["result"] == "ABORTED"


def test_operator_abort_then_safe_open_after_cooling(tmp_path, kb):
    sim, ctrl = make_controller(tmp_path, kb)

    async def scenario():
        await ctrl.tick()
        ctrl.start(make_recipe(), "홍길동")
        await tick_until(ctrl, (Phase.DRYING,))
        assert ctrl.abort()
        await ctrl.tick()
        assert ctrl.phase == Phase.ABORTED
        assert sim.door == "closed" and sim.setpoint_c is None

        with pytest.raises(ProcessError, match="식은 뒤"):
            ctrl.request_open()
        while ctrl.last.temp_c > 60:
            await ctrl.tick()
        ctrl.request_open()
        await tick_until(ctrl, (Phase.IDLE,))

    asyncio.run(scenario())
    assert sim.door == "open"
    assert "작업자 비상 정지" in last_error(ctrl)


def test_start_rejections(tmp_path, kb):
    _, ctrl = make_controller(tmp_path, kb)

    async def scenario():
        with pytest.raises(ProcessError, match="장비 상태"):
            ctrl.start(make_recipe(), "홍길동")
        await ctrl.tick()
        with pytest.raises(ProcessError, match="최대 사용 온도"):
            ctrl.start(make_recipe(heating={"target_c": 500}), "홍길동")
        with pytest.raises(ProcessError, match="승인자"):
            ctrl.start(make_recipe(), "  ")
        ctrl.start(make_recipe(), "홍길동")
        with pytest.raises(ProcessError, match="진행 중"):
            ctrl.start(make_recipe(), "홍길동")
        with pytest.raises(ProcessError, match="진행 중"):
            ctrl.request_open()

    asyncio.run(scenario())


def test_heater_is_reenabled_if_device_dropped_it(tmp_path, kb):
    sim, ctrl = make_controller(tmp_path, kb)

    async def scenario():
        await ctrl.tick()
        ctrl.start(make_recipe(), "홍길동")
        await tick_until(ctrl, (Phase.DRYING,))
        sim.setpoint_c = None  # 펌웨어 워치독이 히터를 끈 상황
        await ctrl.tick()
        await ctrl.tick()

    asyncio.run(scenario())
    assert sim.setpoint_c == 80
    assert any("다시 켭니다" in e["message"] for e in ctrl.events)


def test_repeated_read_failures_abort(tmp_path, kb):
    sim, ctrl = make_controller(tmp_path, kb)

    async def scenario():
        await ctrl.tick()
        ctrl.start(make_recipe(), "홍길동")
        await tick_until(ctrl, (Phase.HEATING,))

        async def broken_read():
            raise OSError("USB 연결 끊김")

        sim.read = broken_read
        for _ in range(3):
            await ctrl.tick()

    asyncio.run(scenario())
    assert ctrl.phase == Phase.ABORTED
    assert "USB 연결 끊김" in last_error(ctrl)
    assert sim.setpoint_c is None and not sim.pump_on
