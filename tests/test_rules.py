"""규칙 기반 추출기 (rules.py): LLM과 같은 계약을 지키는지 — 적힌 값만, 적힌 단위 그대로, 문장 종류 표시."""
import asyncio

import pytest

from process_ai.llm import LocalLLM
from process_ai.planner import DraftState, Planner
from process_ai.rules import RuleExtractor, is_rules_url, make_extractor


@pytest.fixture(scope="module")
def rules(kb):
    return RuleExtractor(kb)


def q(value, unit):
    return {"value": value, "unit": unit}


def test_reads_values_and_units_as_written(rules):
    out = rules.extract("PET 기판 위 톨루엔 시편. 10 Torr까지 배기하고 분당 5도로 90도까지 승온, 최대 30분 유지")
    assert out["targetPressure"] == q(10, "Torr")  # 변환하지 않는다 (온톨로지가 kPa로 바꾼다)
    assert out["targetTemperature"] == q(90, "도")
    assert out["rampRate"] == q(5, "도/분")
    assert out["maxHoldDuration"] == q(30, "분")
    assert out["solventName"] == "톨루엔" and out["substrateName"] == "PET 기판"  # 적힌 그대로. PET 필름으로 바꾸는 건 온톨로지
    assert out["intent"] == "plan"


def test_untouched_items_stay_null(rules):
    out = rules.extract("아세톤 시편, 3 kPa로 건조")
    assert out["targetTemperature"] is None and out["rampRate"] is None and out["recipeName"] is None


def test_cue_words_pick_the_item(rules):
    out = rules.extract("IPA, 5 kPa, 80도. 판정 구간 20초, 변동 허용폭 1%, 최소 2분 최대 40분 유지, 배기 제한 시간 10분")
    assert out["stableWindow"] == q(20, "초")
    assert out["stableBand"] == q(1, "%")
    assert out["minHoldDuration"] == q(2, "분")
    assert out["maxHoldDuration"] == q(40, "분")
    assert out["pumpTimeout"] == q(10, "분")


def test_cooling_temperature_and_compound_time(rules):
    out = rules.extract("120도로 건조하고 40도까지 식힌 뒤 꺼내기. 2시간 30분 유지")
    assert out["targetTemperature"] == q(120, "도")
    assert out["coolToTemperature"] == q(40, "도")
    assert out["maxHoldDuration"] == q(150, "min")


def test_names_and_sample_id(rules):
    out = rules.extract("실험명 PEDOT-07, 시편 번호 S-12. 5 kPa, 120도")
    assert out["recipeName"] == "PEDOT-07" and out["sampleId"] == "S-12"


def test_unknown_substrate_as_written(rules):
    assert rules.extract("구리 포일 위 NMP 시편, 2 kPa, 150도")["substrateName"] == "구리 포일"


def test_process_order_with_negation(rules):
    assert rules.extract("진공 먼저 말고 가열 먼저 해줘. 3 kPa, 100도")["processOrder"] == "heat_first"
    assert rules.extract("진공 먼저 4 kPa까지 잡고 그 다음 75도로 가열")["processOrder"] == "vacuum_first"


def test_followup_bare_number_goes_to_asked_item(rules):
    assert rules.extract("85", asked=["targetTemperature"])["targetTemperature"] == q(85, "")
    assert rules.extract("85")["targetTemperature"] is None  # 물어본 항목이 없으면 어디 값인지 모른다


@pytest.mark.parametrize("text, intent", [
    ("왜 압력을 그렇게 잡았어?", "question"),
    ("200도까지 올려도 돼?", "question"),
    ("온도 10도 올려줘", "revise"),
    ("압력은 5 kPa로 바꿔줘", "revise"),
    ("안녕하세요, 내일 실험 준비 중입니다", "other"),
    ("IPA 5 kPa 80도", "plan"),
])
def test_intent(rules, text, intent):
    assert rules.extract(text)["intent"] == intent


def test_complete_json_parses_planner_prompt(rules, kb):
    planner = Planner(kb, rules)
    result = asyncio.run(planner.chat("아세톤 시편, 3 kPa로 건조"))
    assert result.draft.asked == ["targetTemperature"]
    result = asyncio.run(planner.chat("85", result.draft))
    assert result.recipe is not None and result.recipe.heating.target_c == 85
    assert result.model == "rule-extractor"


def test_status_marks_rules_mode(rules):
    status = asyncio.run(rules.status())
    assert status["ok"] and status["kind"] == "rules"


def test_make_extractor_switches_on_url(kb):
    assert is_rules_url("rules") and is_rules_url("RULES://") and not is_rules_url("http://localhost:1234/v1")
    assert isinstance(make_extractor(kb, "rules"), RuleExtractor)
    assert isinstance(make_extractor(kb, "http://localhost:1234/v1"), LocalLLM)
