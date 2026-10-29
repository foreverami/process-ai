"""실행 설정.

장비 한계값은 코드에 없다. ontology/vd-equipment.ttl에 있고, KnowledgeBase.limits()가 읽어
EquipmentLimits로 만들어 공정 제어기에 넘긴다.
"""
import os
from pathlib import Path

from pydantic import BaseModel


class EquipmentLimits(BaseModel):
    max_temp_c: float          # 구성품 최대 사용 온도 중 가장 낮은 값
    max_ramp_c_per_min: float
    min_pressure_kpa: float    # 진공 펌프 최저 도달 압력
    atm_pressure_kpa: float
    safe_open_temp_c: float
    overshoot_margin_c: float
    max_process_min: float     # 문 닫힘부터 건조 종료까지 (가열이 켜진 구간)
    door_timeout_s: float
    heat_timeout_extra_min: float
    vent_timeout_s: float
    cool_timeout_min: float


DRIVER = os.getenv("PROCESS_DRIVER", "sim")  # "sim" 또는 "arduino"
SERIAL_PORT = os.getenv("PROCESS_SERIAL_PORT", "/dev/tty.usbmodem1101")
SERIAL_BAUD = int(os.getenv("PROCESS_SERIAL_BAUD", "115200"))
SIM_SPEED = float(os.getenv("PROCESS_SIM_SPEED", "20"))  # 시뮬레이터 배속

TICK_S = 0.5  # 센서 읽기/제어 주기 (실제 시간 기준)
RUNS_DIR = Path(os.getenv("PROCESS_RUNS_DIR", Path(__file__).parent / "runs"))

EQUIPMENT_ID = os.getenv("PROCESS_EQUIPMENT", "Chamber01")  # vd-equipment.ttl의 장비 개체 이름

# 로컬 LLM 서버 (OpenAI 호환 API). LM Studio 기본 주소. Ollama는 http://localhost:11434/v1
# "rules"로 적으면 LLM 없이 규칙 기반 추출기(rules.py)를 쓴다 — LLM을 돌릴 수 없는 PC에서 채팅 흐름을 시험할 때
LLM_BASE_URL = os.getenv("PROCESS_LLM_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("PROCESS_LLM_MODEL", "")  # 비우면 서버에 올라와 있는 첫 번째 모델
LLM_TIMEOUT_S = float(os.getenv("PROCESS_LLM_TIMEOUT", "120"))
# 추론형 모델의 "생각 과정"을 끈다. 값 추출에는 필요 없고 응답만 느려진다. 빈 값이면 보내지 않음
LLM_REASONING_EFFORT = os.getenv("PROCESS_LLM_REASONING", "none")

# 작업자가 승인한 (문장 → 값) 기록. 기록은 항상 쌓인다.
# FEWSHOT_K>0이면 비슷한 과거 문장을 프롬프트에 되먹인다. 기본 0(끔): Gemma 4 E2B는 비슷한 문장을 이전 턴으로
# 보여 주면 새 문장의 값(예: 90도)을 빼 버리는 일이 있어, 평가로 효과를 확인한 뒤 켜야 한다 (README 참고).
EXAMPLES_PATH = Path(os.getenv("PROCESS_EXAMPLES_PATH", Path(__file__).parent / "examples" / "examples.jsonl"))
FEWSHOT_K = int(os.getenv("PROCESS_FEWSHOT_K", "0"))
