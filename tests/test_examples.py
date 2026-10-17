import asyncio
import json

from fakes import FakeLLM, extraction

from process_ai.examples import ExampleStore, recipe_values
from process_ai.planner import DraftState, Planner
from process_ai.recipe import Recipe

SENTENCE = "PET 기판 위 톨루엔 시편. 10 Torr까지 배기하고 분당 5도로 90도까지 승온"


def turn(kb, message, **raw):
    """채팅 한 턴: LLM 원출력과 해석한 값."""
    ex = extraction(kb, **raw)
    planner = Planner(kb, FakeLLM(ex))
    interp = asyncio.run(planner.interpret(message, DraftState()))
    return {"message": message, "extracted": interp.extracted, "values": interp.values}


def test_record_keeps_operator_corrections(kb, tmp_path):
    store = ExampleStore(tmp_path / "ex.jsonl", kb)
    t = turn(kb, SENTENCE, solventName="톨루엔", substrateName="PET 기판",
             targetPressure={"value": 10, "unit": "Torr"}, targetTemperature={"value": 90, "unit": "도"},
             rampRate={"value": 5, "unit": "도/분"})
    ai = Recipe.model_validate({"name": "x", "solvent": "톨루엔", "substrate": "PET 필름",
                                "vacuum": {"target_kpa": 1.333}, "heating": {"target_c": 90, "ramp_c_per_min": 5}})
    final = ai.model_copy(deep=True)
    final.heating.target_c = 85  # 작업자가 검토 화면에서 90 → 85로 고쳤다
    example = store.record([t], recipe_values(kb, ai), recipe_values(kb, final), "홍길동", "run-1")

    assert example["corrected"] and example["changed"] == {"targetTemperature": 85}
    assert store.stats() == {"count": 1, "single_turn": 1, "corrected": 1}
    shown = store.example_json(example)
    assert shown["targetPressure"] == {"value": 10.0, "unit": "Torr"}  # 맞았던 값은 원문 단위 그대로
    assert shown["targetTemperature"] == {"value": 85, "unit": "°C"}  # 고친 값은 기준 단위로
    assert shown["substrateName"] == "PET 기판" and shown["intent"] == "plan"
    assert "maxHoldDuration" not in shown  # 기본값으로 채워진 항목은 예시에 넣지 않는다


def test_similar_prefers_shared_wording_and_materials(kb, tmp_path):
    store = ExampleStore(tmp_path / "ex.jsonl", kb)
    for message, raw in [
        (SENTENCE, dict(solventName="톨루엔", targetPressure={"value": 10, "unit": "Torr"}, targetTemperature={"value": 90, "unit": "도"})),
        ("실험명 PEDOT-07, 시편 번호 S-12. 5 kPa, 120도", dict(recipeName="PEDOT-07", sampleId="S-12",
                                                        targetPressure={"value": 5, "unit": "kPa"}, targetTemperature={"value": 120, "unit": "도"})),
    ]:
        t = turn(kb, message, **raw)
        recipe = Planner(kb, FakeLLM())._build_recipe(t["values"] | {"targetPressure": t["values"].get("targetPressure", 5)})[0]
        store.record([t], recipe_values(kb, recipe), recipe_values(kb, recipe), "홍길동", "run")
    # 여러 턴짜리 기록은 예시로 쓰지 않는다
    store.record([turn(kb, "80도", targetTemperature={"value": 80, "unit": "도"}), turn(kb, "5 kPa", targetPressure={"value": 5, "unit": "kPa"})],
                 {}, {"targetPressure": 5, "targetTemperature": 80}, "홍길동", "run")

    found = store.similar("PET 기판에 톨루엔 시편, 8 Torr까지 뽑고 90도로 건조", k=2)
    assert [e["messages"][0] for e in found] == [SENTENCE]
    assert store.similar("안녕하세요", k=2) == []
    block = store.prompt_block(found)
    assert block.startswith("참고 예시") and f"문장: {SENTENCE}" in block and '"targetPressure": {"value": 10.0, "unit": "Torr"}' in block


def test_cases_export_uses_operator_values(kb, tmp_path):
    store = ExampleStore(tmp_path / "ex.jsonl", kb)
    t = turn(kb, "IPA 시편 5 kPa 80도", solventName="IPA", targetPressure={"value": 5, "unit": "kPa"}, targetTemperature={"value": 80, "unit": "도"})
    recipe = Planner(kb, FakeLLM())._build_recipe(t["values"])[0]
    store.record([t], recipe_values(kb, recipe), recipe_values(kb, recipe), "홍길동", "run")
    cases = store.to_cases()
    assert cases[0]["message"] == "IPA 시편 5 kPa 80도" and cases[0]["intent"] == "plan"
    assert cases[0]["expect"] == {"solventName": "이소프로판올", "targetPressure": 5.0, "targetTemperature": 80.0}
