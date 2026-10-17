"""공정 레시피 스키마.

양식 입력과 AI 채팅이 똑같이 만드는 공통 형식이다. 스키마는 타입만 강제하고,
값의 범위와 논리 검사는 온톨로지의 SHACL 규칙(ontology/vd-shapes.ttl)이 맡는다.
각 필드의 의미·단위·동의어는 ontology/vd-core.ttl에서 vd:jsonField로 연결된다.
"""
from typing import Literal

from pydantic import BaseModel, Field

AMBIENT_C = 25.0


class VacuumStep(BaseModel):
    target_kpa: float = Field(description="목표 챔버 압력 (kPa, 절대압)")
    pump_timeout_min: float = Field(10.0, description="목표 압력 도달 제한 시간 (분)")


class HeatingStep(BaseModel):
    target_c: float = Field(description="가열 스테이지 목표 온도 (°C)")
    ramp_c_per_min: float = Field(10.0, description="승온 속도 (°C/분)")


class EndpointRule(BaseModel):
    """포토다이오드 신호가 일정 시간 평탄하면 용매 증발 완료로 판단한다."""

    min_hold_min: float = Field(1.0, description="목표 온도 도달 후 최소 유지 시간 (분)")
    max_hold_min: float = Field(60.0, description="종료점 미검출 시 최대 유지 시간 (분)")
    stable_window_s: float = Field(30.0, description="신호 안정 판정 구간 (초)")
    stable_band_pct: float = Field(0.5, description="판정 구간 안에서 허용하는 신호 변동폭 (%)")


class FinishStep(BaseModel):
    cool_to_c: float = Field(50.0, description="벤트와 문 열기 전 냉각 목표 온도 (°C)")


class Recipe(BaseModel):
    name: str = Field(description="실험 이름")
    sample_id: str = ""
    solvent: str = ""
    substrate: str = ""
    order: Literal["vacuum_first", "heat_first"] = "vacuum_first"
    vacuum: VacuumStep
    heating: HeatingStep
    endpoint: EndpointRule = Field(default_factory=EndpointRule)
    finish: FinishStep = Field(default_factory=FinishStep)
    notes: str = ""


class Issue(BaseModel):
    level: Literal["error", "warning", "info"]
    field: str
    message: str


def field_paths(model: type[BaseModel] = Recipe, prefix: str = "") -> list[str]:
    """레시피의 값 필드 경로 목록 (예: heating.target_c)을 선언 순서대로 돌려준다."""
    paths = []
    for name, info in model.model_fields.items():
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            paths += field_paths(annotation, f"{prefix}{name}.")
        else:
            paths.append(prefix + name)
    return paths


def default_for(path: str):
    """필드 경로의 기본값. 기본값이 없는 필수 필드면 None."""
    model: type[BaseModel] = Recipe
    *parents, leaf = path.split(".")
    for name in parents:
        model = model.model_fields[name].annotation
    info = model.model_fields[leaf]
    return None if info.is_required() else info.get_default(call_default_factory=True)


def active_minutes_upper_bound(r: Recipe) -> float:
    """문이 닫힌 뒤 건조가 끝날 때까지 걸릴 수 있는 최대 시간(분)."""
    ramp_min = max(r.heating.target_c - AMBIENT_C, 0) / r.heating.ramp_c_per_min
    return r.vacuum.pump_timeout_min + ramp_min + r.endpoint.max_hold_min


def has_errors(issues: list[Issue]) -> bool:
    return any(i.level == "error" for i in issues)
