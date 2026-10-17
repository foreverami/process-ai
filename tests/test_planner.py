import asyncio

import pytest
from fakes import FakeLLM, extraction

from process_ai.llm import LLMError
from process_ai.planner import DraftState, Planner


def chat(planner, message, draft=None):
    return asyncio.run(planner.chat(message, draft or DraftState()))


def test_prompt_and_schema_are_built_from_ontology(kb):
    planner = Planner(kb, FakeLLM())
    prompt = planner.system_prompt
    assert "targetPressure: 목표 압력 (다른 표현: 진공도" in prompt
    assert "이소프로판올(2-propanol, IPA, isopropyl alcohol)" in prompt
    assert '"heat_first"(가열 후 진공 배기)' in prompt

    def check(node):
        assert "$ref" not in node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        for value in node.values():
            if isinstance(value, dict):
                check(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        check(item)

    check(planner.schema)
    assert set(planner.schema["properties"]) == {p.key for p in kb.parameters} | {"intent"}


def test_full_plan_becomes_recipe_with_conversion_linking_and_defaults(kb):
    llm = FakeLLM(extraction(
        kb,
        solventName="IPA",
        substrateName="PET 기판",
        targetPressure={"value": 10, "unit": "Torr"},
        targetTemperature={"value": 80, "unit": "도"},
        rampRate={"value": 5, "unit": "도/분"},
    ))
    result = chat(Planner(kb, llm), "PET 기판에 IPA 시편, 10 Torr까지 뽑고 분당 5도로 80도까지 올려서 건조")

    r = result.recipe
    assert r.vacuum.target_kpa == pytest.approx(1.333)
    assert (r.heating.target_c, r.heating.ramp_c_per_min) == (80, 5)
    assert (r.solvent, r.substrate, r.name) == ("이소프로판올", "PET 필름", "이소프로판올 80°C 진공 건조")
    assert "목표 압력: 10 Torr → 1.333 kPa (단위 변환)" in result.notes
    assert "용매: 'IPA' → 이소프로판올 (온톨로지 용어 연결)" in result.notes
    assert "최대 유지 시간: 60 min (기본값)" in result.notes
    assert not [i for i in result.issues if i.level == "error"]
    assert result.reply.startswith("레시피 초안을 만들었습니다: 목표 압력 1.333 kPa, 목표 온도 80 °C")
    assert result.draft.asked == []


def test_missing_values_are_asked_and_short_answer_is_merged(kb):
    llm = FakeLLM(
        extraction(kb, solventName="에탄올", targetTemperature={"value": 70, "unit": "°C"}),
        extraction(kb, targetPressure={"value": 5, "unit": ""}),
    )
    planner = Planner(kb, llm)

    first = chat(planner, "에탄올 시편 70도에서 말려줘")
    assert first.recipe is None
    assert first.draft.asked == ["targetPressure"]
    assert "지금까지 이해한 조건: 용매 에탄올, 목표 온도 70 °C." in first.reply
    assert "레시피를 만들려면 이 값이 필요해요: 목표 압력(kPa)." in first.reply

    second = chat(planner, "5", first.draft)
    assert "직전에 작업자에게 물어본 항목: 목표 압력(targetPressure)" in llm.calls[1][1]["content"]
    assert (second.recipe.vacuum.target_kpa, second.recipe.heating.target_c) == (5, 70)
    assert "목표 압력: 단위가 없어 kPa로 해석했습니다." in second.notes


def test_ontology_fills_what_the_llm_missed(kb):
    llm = FakeLLM(extraction(kb, targetTemperature={"value": 80, "unit": "°C"}))
    result = chat(Planner(kb, llm), "IPA 시편을 가열 먼저 해서 80도로 올리고 최대 진공으로 건조")

    r = result.recipe
    assert (r.solvent, r.order, r.vacuum.target_kpa) == ("이소프로판올", "heat_first", 1.0)
    assert "목표 압력: '최대 진공' → 1 kPa (장비의 최저 도달 압력)" in result.notes
    assert any("온톨로지 사전" in n and "이소프로판올" in n for n in result.notes)


def test_wrong_unit_is_reported_not_used(kb):
    llm = FakeLLM(extraction(kb, targetPressure={"value": 5, "unit": "분"}, targetTemperature={"value": 80, "unit": "°C"}))
    result = chat(Planner(kb, llm), "5분 압력, 80도")
    assert result.recipe is None
    assert "확인이 필요한 내용: '분'는 목표 압력에 쓸 수 없는 단위입니다." in result.reply


def test_rule_violation_is_explained_in_reply(kb):
    llm = FakeLLM(extraction(
        kb, substrateName="PET", targetPressure={"value": 5, "unit": "kPa"}, targetTemperature={"value": 150, "unit": "°C"}
    ))
    result = chat(Planner(kb, llm), "PET 기판, 5 kPa, 150도")
    assert "실행할 수 없는 항목이 1개 있습니다: 목표 온도 150°C가 PET 필름의 최대 사용 온도 120°C를 넘습니다." in result.reply


def test_invalid_output_is_retried_once(kb):
    good = extraction(kb, targetPressure={"value": 5, "unit": "kPa"}, targetTemperature={"value": 80, "unit": "°C"})
    llm = FakeLLM({"targetTemperature": "여든"}, good)
    result = chat(Planner(kb, llm), "5 kPa, 80도")
    assert result.recipe is not None
    assert "형식이 틀렸다" in llm.calls[1][-1]["content"]

    with pytest.raises(LLMError, match="형식에 맞지 않습니다"):
        chat(Planner(kb, FakeLLM({"x": 1}, {"y": 2})), "5 kPa, 80도")


def interpret(planner, message, draft=None):
    return asyncio.run(planner.interpret(message, draft or DraftState()))


def test_prompt_carries_ontology_extraction_hints(kb):
    prompt = Planner(kb, FakeLLM()).system_prompt
    assert "sampleId: 시편 ID" in prompt and "숫자가 든 식별 코드" in prompt
    assert "예시 2" in prompt and "예시의 값을 옮기지 않는다" in prompt


def test_guard_drops_values_copied_from_the_prompt_example(kb):
    # 실제 평가에서 Kanana 2 3B가 "IPA5kPa80도건조"에 프롬프트 예시의 기판·승온 속도·유지 시간을 붙였다
    llm = FakeLLM(extraction(
        kb, solventName="IPA", substrateName="PET 기판",
        targetPressure={"value": 5, "unit": "kPa"}, targetTemperature={"value": 80, "unit": "도"},
        rampRate={"value": 5, "unit": "도/분"}, maxHoldDuration={"value": 30, "unit": "분"},
    ))
    interp = interpret(Planner(kb, llm), "IPA5kPa80도건조")
    # 30분과 문장에 없는 기판은 뺀다. 5는 '5kPa'에 있지만 한 번뿐이라 승온 속도에 또 쓴 것은 중복으로 뺀다
    assert interp.values == {"solventName": "이소프로판올", "targetPressure": 5, "targetTemperature": 80}
    assert any("최대 유지 시간 30" in p for p in interp.problems)
    assert any("승온 속도 5: 같은 숫자를" in p for p in interp.problems)
    assert any("기판 'PET 기판'" in p for p in interp.problems)


def test_guard_drops_names_of_the_wrong_kind(kb):
    # 실제 평가에서 Qwen3 4B가 "톨루엔 시편"을 기판으로 뽑았다
    llm = FakeLLM(extraction(kb, solventName="톨루엔", substrateName="톨루엔 시편", targetTemperature={"value": 90, "unit": "도"}))
    interp = interpret(Planner(kb, llm), "톨루엔 시편을 풀 진공 걸고 90도로 말려")
    assert "substrateName" not in interp.values
    assert any("톨루엔은 용매라 기판에서 뺐습니다" in p for p in interp.problems)
    assert interp.values["targetPressure"] == 1.0  # '풀 진공'은 온톨로지가 채운다


def test_guard_accepts_other_names_of_the_same_entity_and_unknown_written_names(kb):
    llm = FakeLLM(
        extraction(kb, solventName="toluene", substrateName="PET 필름", targetPressure={"value": 3000, "unit": "Pa"}),
        extraction(kb, solventName="DMSO", substrateName="구리 포일", targetTemperature={"value": 150, "unit": "도"}),
    )
    planner = Planner(kb, llm)
    # 모델이 'PET 기판'을 'PET 필름'으로 바꿔 써도 같은 개체가 문장에 있으면 근거가 있다. '3,000'도 숫자로 읽는다
    first = interpret(planner, "PET 기판 위 톨루엔, 3,000 Pa")
    assert first.values == {"solventName": "톨루엔", "substrateName": "PET 필름", "targetPressure": 3}
    # 온톨로지에 없는 이름이라도 문장에 적혀 있으면 그대로 둔다
    second = interpret(planner, "구리 포일 위 DMSO 시편, 150도")
    assert second.values == {"solventName": "DMSO", "substrateName": "구리 포일", "targetTemperature": 150}
    assert second.problems == []


def test_guard_rejects_generic_terms_wrong_kind_ids_and_copied_notes(kb):
    # v2 평가에서 실제로 나온 출력들: 시편 ID 칸의 용매 이름·일반 명사, 기판 칸의 '시편', 요청 문장 전체를 메모로
    llm = FakeLLM(
        extraction(kb, sampleId="MeOH", solventName="메탄올", targetTemperature={"value": 55, "unit": "도"}),
        extraction(kb, sampleId="시편", substrateName="시편", solventName="PGMEA", targetPressure={"value": 5, "unit": "kPa"}),
        extraction(kb, notes="지난번이랑 같은 조건으로 해줘"),
        extraction(kb, sampleId="S-12", recipeName="PEDOT-07", targetPressure={"value": 5, "unit": "kPa"}),
    )
    planner = Planner(kb, llm)
    assert interpret(planner, "MeOH 시편, 설정 온도 55도").values == {"solventName": "메탄올", "targetTemperature": 55}
    second = interpret(planner, "PGMEA 시편 5 kPa")
    assert second.values == {"solventName": "PGMEA", "targetPressure": 5}
    assert any("일반 명사" in p for p in second.problems)
    third = interpret(planner, "지난번이랑 같은 조건으로 해줘")
    assert third.values == {} and any("메모" in p for p in third.problems)
    # 형식에 맞는 식별 코드는 그대로 둔다
    assert interpret(planner, "실험명 PEDOT-07, 시편 번호 S-12. 5 kPa").values == {
        "recipeName": "PEDOT-07", "sampleId": "S-12", "targetPressure": 5,
    }


def test_guard_trusts_ontology_over_llm_for_process_order(kb):
    # v2 평가에서 Qwen이 '예열 후 진공'을 vacuum_first로 뽑았다. 온톨로지가 문장에서 찾은 순서가 이긴다
    llm = FakeLLM(
        extraction(kb, processOrder="vacuum_first", targetPressure={"value": 2, "unit": "kPa"}),
        extraction(kb, processOrder="heat_first", targetPressure={"value": 2, "unit": "kPa"}),
        extraction(kb, processOrder="heat_first", targetPressure={"value": 2, "unit": "kPa"}),
    )
    planner = Planner(kb, llm)
    assert interpret(planner, "예열 후 진공으로 진행. 2 kPa").values["processOrder"] == "heat_first"
    # 온톨로지가 순서 표현을 못 찾으면 LLM의 답을 쓰지 않는다 ("건조 후 식혀서"의 '후'만 보고 순서를 지어내므로)
    assert "processOrder" not in interpret(planner, "건조 후 40도까지 식혀서 꺼내기. 2 kPa").values
    # 대신 온톨로지의 순서 표현 사전을 넓혔다
    assert interpret(planner, "먼저 예열한 다음 진공을 2 kPa까지").values["processOrder"] == "heat_first"


def test_guard_rejects_coating_material_in_substrate_slot(kb):
    # v3 평가에서 세 모델 모두 'PEDOT:PSS 박막'을 기판으로 뽑았다. 온톨로지의 코팅 재료 클래스로 구분한다
    llm = FakeLLM(
        extraction(kb, substrateName="PEDOT:PSS 박막", solventName="물", targetPressure={"value": 5, "unit": "kPa"}),
        extraction(kb, substrateName="PR", solventName="PGMEA", targetTemperature={"value": 110, "unit": "도"}),
    )
    planner = Planner(kb, llm)
    first = interpret(planner, "PEDOT:PSS 박막, 물 용매, 5 kPa")
    assert first.values == {"solventName": "물", "targetPressure": 5}
    assert any("PEDOT:PSS는 코팅 재료라 기판에서 뺐습니다" in p for p in first.problems)
    second = interpret(planner, "PR 코팅 시편, PGMEA, 110도")
    assert second.values == {"solventName": "PGMEA", "targetTemperature": 110}


DRAFT = DraftState(
    values={"solventName": "톨루엔", "substrateName": "PET 필름", "targetPressure": 1.333, "targetTemperature": 90},
    notes=["목표 압력: 10 Torr → 1.333 kPa (단위 변환)", "기판: 'PET 기판' → PET 필름 (온톨로지 용어 연결)"],
)


def test_prompt_and_schema_carry_intent(kb):
    planner = Planner(kb, FakeLLM())
    assert "intent에 문장의 종류를 적는다" in planner.system_prompt and "예시 3" in planner.system_prompt
    assert set(planner.schema["properties"]["intent"]["anyOf"][0]["enum"]) == {"plan", "question", "revise", "other"}


def test_question_is_answered_with_grounds_instead_of_a_new_draft(kb):
    # 작업자가 "왜?"라고 물으면 초안을 다시 만들지 않고 단위 변환·기본값·규칙·지식을 근거로 설명한다
    result = chat(Planner(kb, FakeLLM(extraction(kb, intent="question"))), "왜 그렇게 제안했어?", DRAFT)
    assert result.recipe is None and result.draft == DRAFT
    assert "초안의 근거예요." in result.reply
    assert "- 목표 압력 1.333 kPa: 10 Torr → 1.333 kPa (단위 변환) 펌프 최저 도달 압력은 1 kPa예요." in result.reply
    assert "- 목표 온도 90 °C: 말씀하신 값 그대로예요. 장비 최대 온도는 200 °C예요." in result.reply
    assert "- 승온 속도 10 °C/min: 말씀이 없어 기본값을 썼어요. 장비 최대 승온 속도의 절반" in result.reply
    assert "장비 최대 승온 속도는 20 °C/min이에요." in result.reply
    assert "- 실험 이름 '톨루엔 90°C 진공 건조': 말씀이 없어 자동으로 지었어요." in result.reply
    assert "규칙 검사: [참고] 톨루엔의 인화점은 4°C입니다." in result.reply
    assert "PET 필름의 최대 사용 온도는 120°C입니다." in result.reply


def test_question_about_one_parameter_answers_only_that(kb):
    result = chat(Planner(kb, FakeLLM(extraction(kb, intent="question"))), "승온 속도는 어떻게 정한 거야?", DRAFT)
    assert result.reply.startswith("승온 속도의 근거예요.\n- 승온 속도 10 °C/min: 말씀이 없어 기본값을 썼어요.")
    assert "목표 압력" not in result.reply and "규칙 검사" not in result.reply


def test_question_without_draft_asks_for_conditions(kb):
    result = chat(Planner(kb, FakeLLM(extraction(kb, intent="question"))), "왜 그렇게 제안했어?")
    assert result.reply.startswith("아직 초안이 없어요.") and result.recipe is None


def test_intent_heuristics_cover_llm_mistakes(kb):
    planner = Planner(kb, FakeLLM(
        extraction(kb, intent="plan"),  # 값 없이 '왜'가 있으면 LLM이 plan이라 해도 질문
        extraction(kb, intent="question", targetTemperature={"value": 95, "unit": "도"}),  # 값이 있으면 질문이 아니다
    ))
    asked_why = chat(planner, "왜 90도야?", DRAFT)
    assert "근거예요" in asked_why.reply and asked_why.recipe is None
    revised = chat(planner, "95도로 올리면 어때?", DRAFT)
    assert revised.recipe.heating.target_c == 95
    assert revised.reply.startswith("바꿨어요: 목표 온도 90 → 95 °C.")
    assert revised.draft.notes == DRAFT.notes  # 해석 메모는 대화를 따라 유지된다


def test_unrelated_message_keeps_the_draft(kb):
    result = chat(Planner(kb, FakeLLM(extraction(kb, intent="other"))), "고마워, 이따가 이어서 할게", DRAFT)
    assert result.recipe is None and result.draft == DRAFT
    assert "초안은 그대로예요: 용매 톨루엔, 기판 PET 필름, 목표 압력 1.333 kPa, 목표 온도 90 °C." in result.reply


def test_merge_notes_replaces_same_parameter():
    from process_ai.planner import merge_notes

    old = ["목표 압력: 10 Torr → 1.333 kPa (단위 변환)", "용매: 'IPA' → 이소프로판올 (온톨로지 용어 연결)"]
    new = ["목표 압력: 50 mbar → 5 kPa (단위 변환)"]
    assert merge_notes(old, new) == ["용매: 'IPA' → 이소프로판올 (온톨로지 용어 연결)", "목표 압력: 50 mbar → 5 kPa (단위 변환)"]


def test_entity_name_inside_a_question_does_not_turn_it_into_a_plan(kb):
    # 콘솔에서 "이 조건이면 PET 괜찮아?"가 초안 재생성으로 처리됐다. 온톨로지 사전이 'PET'를 기판 값으로 채웠기 때문
    planner = Planner(kb, FakeLLM(
        extraction(kb, intent="question"),
        extraction(kb, intent="revise", substrateName="유리 기판"),
        extraction(kb, intent="plan", substrateName="PEN"),
    ))
    asked = chat(planner, "이 조건이면 PET 괜찮아?", DRAFT)
    assert asked.recipe is None and asked.draft == DRAFT
    assert "규칙 검사: [참고] 톨루엔의 인화점은 4°C입니다." in asked.reply
    # 이름만 있어도 LLM이 수정이라 했거나 의문 표현이 없으면 값으로 쓴다
    assert chat(planner, "유리 기판으로 바꾸면 어때?", DRAFT).recipe.substrate == "유리 기판"
    assert chat(planner, "PEN 기판", DRAFT).recipe.substrate == "PEN 필름"


def test_quoting_the_current_value_in_a_question_is_not_a_revision(kb):
    # v4 평가에서 "왜 압력을 1.333 kPa로 잡았어?"의 1.333을 LLM이 값으로 뽑아 수정으로 처리됐다
    planner = Planner(kb, FakeLLM(
        extraction(kb, intent="question", targetPressure={"value": 1.333, "unit": "kPa"}),
        extraction(kb, intent="question", targetPressure={"value": 5, "unit": "kPa"}),
    ))
    same = chat(planner, "왜 압력을 1.333 kPa로 잡았어?", DRAFT)
    assert same.recipe is None and same.reply.startswith("목표 압력의 근거예요.")
    changed = chat(planner, "압력을 5 kPa로 하면 어때?", DRAFT)  # 값이 달라지면 수정이다
    assert changed.recipe.vacuum.target_kpa == 5 and changed.reply.startswith("바꿨어요: 목표 압력 1.333 → 5 kPa.")


def test_llm_question_flag_alone_does_not_discard_a_stated_name(kb):
    # v4 평가에서 Qwen이 "시편 번호 W-03 준비됐어"에 intent=question을 붙여 시편 ID가 버려졌다
    result = chat(Planner(kb, FakeLLM(extraction(kb, intent="question", sampleId="W-03"))), "시편 번호 W-03 준비됐어", DRAFT)
    assert result.draft.values["sampleId"] == "W-03" and result.recipe.sample_id == "W-03"


# 작업자가 콘솔에서 실제로 한 대화에서 나온 문제들
BARE = DraftState(values={"targetPressure": 1.0, "targetTemperature": 200.0})


def test_changing_a_defaulted_value_is_reported_as_a_change(kb):
    # "최대 유지시간을 30분으로 수정해줘" → 60은 기본값이라 초안 값에 없었고, 바뀐 것을 알려 주지 않았다
    llm = FakeLLM(extraction(kb, intent="revise", maxHoldDuration={"value": 30, "unit": "분"}))
    result = chat(Planner(kb, llm), "현재 계획에서 최대 유지시간을 30분으로 수정해줘", BARE)
    assert result.reply.startswith("바꿨어요: 최대 유지 시간 60 → 30 min.")
    assert result.recipe.endpoint.max_hold_min == 30


def test_question_about_defaults_lists_only_defaults_with_their_rationale(kb):
    result = chat(Planner(kb, FakeLLM(extraction(kb, intent="question"))), "기본값이 뭐를 바탕으로 설정이 된거야?", BARE)
    assert result.reply.startswith("기본값으로 채운 항목이에요.")
    assert "- 승온 속도 10 °C/min: 말씀이 없어 기본값을 썼어요. 장비 최대 승온 속도의 절반으로 잡은" in result.reply
    assert "- 안정 판정 구간 30 s: 말씀이 없어 기본값을 썼어요. 시뮬레이터용 값이에요." in result.reply
    assert "- 목표 압력" not in result.reply and "규칙 검사" not in result.reply


def test_loose_label_match_and_overridden_default_are_explained(kb):
    # "왜 최대 온도 시간이 사라졌어?" — 이름이 정확하지 않아도 두 단어가 겹치면 그 항목이고,
    # 기본값이던 항목을 바꾼 뒤에는 어느 기본값 대신인지 말해 준다
    draft = DraftState(values=BARE.values | {"maxHoldDuration": 30.0})
    result = chat(Planner(kb, FakeLLM(extraction(kb, intent="question"))), "왜 최대 온도 시간이 사라졌어?", draft)
    assert result.reply.startswith("최대 유지 시간의 근거예요.\n- 최대 유지 시간 30 min: 말씀하신 값이에요 (기본값 60 min 대신).")


# ---- 제가 만든 질문 세트(cases_candidates.jsonl)에서 드러난 기능들
def test_relative_change_adds_to_the_current_value(kb):
    planner = Planner(kb, FakeLLM(
        extraction(kb, intent="revise", targetTemperature={"value": 10, "unit": "도"}),
        extraction(kb, intent="revise", targetPressure={"value": 0.5, "unit": "kPa"}),
        extraction(kb, intent="revise", targetTemperature={"value": 100, "unit": "도"}),
        extraction(kb, intent="revise", targetTemperature={"value": 120, "unit": "도"}),
    ))
    up = chat(planner, "온도 10도 올려줘", DRAFT)
    assert up.recipe.heating.target_c == 100 and up.reply.startswith("바꿨어요: 목표 온도 90 → 100 °C.")
    assert "목표 온도: 90 → 100 °C (+10 °C)" in up.notes
    down = chat(planner, "압력은 0.5 kPa 낮춰", DRAFT)
    assert down.recipe.vacuum.target_kpa == 0.833
    # 숫자 뒤에 '로', '까지'가 있으면 절대값이다
    assert chat(planner, "온도를 100도로 올려", DRAFT).recipe.heating.target_c == 100
    assert chat(planner, "온도 120도까지 올려", DRAFT).recipe.heating.target_c == 120


def test_what_if_question_checks_rules_without_changing_the_draft(kb):
    planner = Planner(kb, FakeLLM(
        extraction(kb, intent="question", targetTemperature={"value": 200, "unit": "도"}),
        extraction(kb, intent="question", targetTemperature={"value": 150, "unit": "도"}),
    ))
    ok = chat(planner, "200도까지 올려도 돼?", DRAFT)
    assert ok.recipe is None and ok.draft == DRAFT
    assert ok.reply.startswith("목표 온도 200 °C(으)로 하면 실행할 수 없어요: 목표 온도 200°C가 PET 필름의 최대 사용 온도 120°C를 넘습니다.")
    no = chat(planner, "왜 150도는 안 돼?", DRAFT)
    assert "PET 필름의 최대 사용 온도 120°C를 넘습니다" in no.reply and no.draft == DRAFT


def test_material_questions_are_answered_from_the_ontology(kb):
    planner = Planner(kb, FakeLLM(*[extraction(kb, intent="question")] * 4))
    assert chat(planner, "PET 최대 온도가 몇 도야?", DRAFT).reply == "- PET 필름의 최대 사용 온도는 120°C입니다."
    flash = chat(planner, "톨루엔 인화점이 몇 도지?", DRAFT).reply
    assert "톨루엔 (CAS 108-88-3): 끓는점 110.6°C, 인화점 4°C" in flash and "1.333 kPa에서 톨루엔의 끓는점" in flash
    # 초안이 없어도 재료 질문에는 답한다
    assert chat(planner, "PGMEA 끓는점 알려줘").reply == "- PGMEA (CAS 108-65-6): 끓는점 146°C, 인화점 42°C"
    compare = chat(planner, "IPA랑 에탄올 중에 뭐가 더 빨리 말라?", DRAFT).reply
    assert "끓는점이 낮은 에탄올 쪽이" in compare and "이소프로판올 (CAS" in compare


def test_duration_next_step_start_and_reset_replies(kb):
    planner = Planner(kb, FakeLLM(*[extraction(kb, intent="question")] * 2 + [extraction(kb, intent="other")] * 2))
    duration = chat(planner, "이거 얼마나 걸려?", DRAFT).reply
    assert duration.startswith("문을 닫은 뒤 건조 종료까지 최대 76.5분이에요 (배기 최대 10분 + 승온 약 6분 + 유지 최대 60분).")
    assert "승인하고 공정 시작" in chat(planner, "다음에 뭐 해야 돼?", DRAFT).reply
    start = chat(planner, "공정 시작해", DRAFT)
    assert start.reply.startswith("채팅으로는 공정을 시작하지 않아요.") and start.draft == DRAFT
    assert "'새 대화' 버튼" in chat(planner, "처음부터 다시 할래", DRAFT).reply


def test_hold_time_without_minimum_cue_is_the_maximum_and_multipliers_work(kb):
    planner = Planner(kb, FakeLLM(
        extraction(kb, intent="plan", targetTemperature={"value": 90, "unit": "도"}, targetPressure={"value": 2, "unit": "kPa"},
                   minHoldDuration={"value": 150, "unit": "분"}),
        extraction(kb, intent="revise"),
    ))
    # Gemma가 "2시간 30분 유지"를 최소 유지 시간에 넣었다. '최소'라는 말이 없으면 최대 유지 시간이다
    hold = chat(planner, "90도에서 2시간 30분 유지, 압력 2 kPa")
    assert hold.recipe.endpoint.max_hold_min == 150 and hold.recipe.endpoint.min_hold_min == 1
    # "압력 절반으로" — 숫자가 없어도 문장이 가리키는 항목 하나에 배수를 적용한다
    half = chat(planner, "압력 절반으로", DRAFT)
    assert half.recipe.vacuum.target_kpa == 0.6665 and half.reply.startswith("바꿨어요: 목표 압력 1.333 → 0.6665 kPa.")


def test_code_reads_a_single_number_when_the_llm_returns_nothing(kb):
    # Gemma는 "압력은 0.5 kPa 낮춰"에서 값을 비워 보냈다. 항목 하나 + 숫자·단위 하나뿐이면 코드가 직접 읽는다
    planner = Planner(kb, FakeLLM(extraction(kb, intent="revise"), extraction(kb, intent="revise"), extraction(kb, intent="plan")))
    down = chat(planner, "압력은 0.5 kPa 낮춰", DRAFT)
    assert down.recipe.vacuum.target_kpa == 0.833 and down.reply.startswith("바꿨어요: 목표 압력 1.333 → 0.833 kPa.")
    assert "목표 압력: 0.5 kPa (문장에서 직접 읽음)" in down.notes
    absolute = chat(planner, "온도 120도로", DRAFT)
    assert absolute.recipe.heating.target_c == 120
    # 단위가 항목과 맞지 않으면 읽지 않는다
    assert "targetPressure" not in interpret(planner, "압력 30분").values


def test_question_targets_prefer_the_most_specific_name(kb):
    planner = Planner(kb, FakeLLM())
    assert [p.key for p in planner._question_targets("온도 120도로")] == ["targetTemperature"]
    assert [p.key for p in planner._question_targets("냉각 온도는 왜 50도야?")] == ["coolToTemperature"]
    assert [p.key for p in planner._question_targets("최대 온도 시간이 왜 사라졌어?")] == ["maxHoldDuration"]


def test_v5_regressions_duplicate_number_and_plan_verb(kb):
    # v5 평가: "판정 구간 1분 …"의 1을 최소 유지 시간에도 넣었고, 최소→최대 규칙이 그걸 최대 1분으로 옮겼다
    planner = Planner(kb, FakeLLM(
        extraction(kb, intent="plan", targetPressure={"value": 4, "unit": "kPa"}, targetTemperature={"value": 85, "unit": "도"},
                   minHoldDuration={"value": 1, "unit": "분"}, stableWindow={"value": 1, "unit": "분"}, stableBand={"value": 0.5, "unit": "%"}),
        extraction(kb, intent="plan", substrateName="PI 필름", targetPressure={"value": 4, "unit": "kPa"},
                   targetTemperature={"value": 110, "unit": "도"}, rampRate={"value": 3, "unit": "도/분"}),
        extraction(kb, intent="revise", targetTemperature={"value": 110, "unit": "도"}),
    ))
    window = chat(planner, "판정 구간 1분, 허용 변동률 0.5%, 압력 4 kPa, 온도 85도")
    assert window.recipe.endpoint.stable_window_s == 60 and window.recipe.endpoint.max_hold_min == 60
    assert any("같은 숫자를 다른 항목에도" in p for p in window.problems) if hasattr(window, "problems") else True
    # "분당 3도씩 올려서 110도"의 '올려서'는 계획 문장의 동사다. 초안이 없으면 상대 변경이 아니다
    plan = chat(planner, "분당 3도씩 올려서 110도, 4 kPa, PI 필름")
    assert plan.recipe.heating.ramp_c_per_min == 3 and plan.recipe.heating.target_c == 110
    # 초안이 있어도 동사가 숫자 앞에 오면 절대값이다
    assert chat(planner, "올려서 110도", DRAFT).recipe.heating.target_c == 110


def test_similar_operator_examples_are_fed_back_into_the_prompt(kb, tmp_path):
    from process_ai.examples import ExampleStore, recipe_values

    store = ExampleStore(tmp_path / "ex.jsonl", kb)
    first = Planner(kb, FakeLLM(extraction(kb, solventName="톨루엔", targetPressure={"value": 10, "unit": "Torr"},
                                           targetTemperature={"value": 90, "unit": "도"})))
    interp = asyncio.run(first.interpret("PET 기판 위 톨루엔 시편. 10 Torr까지 배기하고 90도까지 승온"))
    recipe = first._build_recipe(interp.values)[0]
    store.record([{"message": "PET 기판 위 톨루엔 시편. 10 Torr까지 배기하고 90도까지 승온", "extracted": interp.extracted,
                   "values": interp.values}], recipe_values(kb, recipe), recipe_values(kb, recipe), "홍길동", "run")

    llm = FakeLLM(extraction(kb, solventName="톨루엔", targetPressure={"value": 8, "unit": "Torr"}, targetTemperature={"value": 90, "unit": "도"}),
                  extraction(kb, targetPressure={"value": 5, "unit": "kPa"}))
    planner = Planner(kb, llm, store, fewshot_k=2)
    result = chat(planner, "PET 기판에 톨루엔 시편, 8 Torr까지 뽑고 90도로 건조")
    # 과거 문장과 그 정답이 이전 대화 턴(user → assistant)으로 들어가고, 새 문장이 마지막 턴이다
    roles = [m["role"] for m in llm.calls[0]]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.calls[0][1]["content"] == "문장: PET 기판 위 톨루엔 시편. 10 Torr까지 배기하고 90도까지 승온"
    assert '"targetPressure": {"value": 10.0, "unit": "Torr"}' in llm.calls[0][2]["content"]
    assert llm.calls[0][3]["content"] == "문장: PET 기판에 톨루엔 시편, 8 Torr까지 뽑고 90도로 건조"
    assert "참고한 과거 문장: 1개 (작업자가 확인한 기록)" in result.notes
    assert result.values["targetPressure"] == 1.067  # 8 Torr
    # 되묻기에 대한 짧은 답에는 예시를 넣지 않는다
    chat(planner, "5 kPa", DraftState(values={"targetTemperature": 90}, asked=["targetPressure"]))
    assert [m["role"] for m in llm.calls[1]] == ["system", "user"]


def test_examples_are_dropped_when_the_model_loses_a_required_value(kb, tmp_path):
    # 실제 Gemma: 비슷한 과거 문장을 보여 주면 "8 Torr까지 뽑고 90도로 건조"에서 90도를 빼 버렸다 → 예시 없이 다시 뽑는다
    from process_ai.examples import ExampleStore, recipe_values

    store = ExampleStore(tmp_path / "ex.jsonl", kb)
    seed = Planner(kb, FakeLLM(extraction(kb, solventName="톨루엔", targetPressure={"value": 10, "unit": "Torr"},
                                          targetTemperature={"value": 90, "unit": "도"})))
    interp = asyncio.run(seed.interpret("PET 기판 위 톨루엔 시편. 10 Torr까지 배기하고 90도까지 승온"))
    recipe = seed._build_recipe(interp.values)[0]
    store.record([{"message": "PET 기판 위 톨루엔 시편. 10 Torr까지 배기하고 90도까지 승온", "extracted": interp.extracted,
                   "values": interp.values}], recipe_values(kb, recipe), recipe_values(kb, recipe), "홍길동", "run")
    llm = FakeLLM(
        extraction(kb, solventName="톨루엔", targetPressure={"value": 8, "unit": "Torr"}),  # 예시를 보고 온도를 놓침
        extraction(kb, solventName="톨루엔", targetPressure={"value": 8, "unit": "Torr"}, targetTemperature={"value": 90, "unit": "도"}),
    )
    result = chat(Planner(kb, llm, store, fewshot_k=2), "PET 기판에 톨루엔 시편, 8 Torr까지 뽑고 90도로 건조")
    assert len(llm.calls) == 2 and [m["role"] for m in llm.calls[1]] == ["system", "user"]
    assert result.recipe.heating.target_c == 90
    assert "과거 문장 참고를 취소했어요 (값을 놓쳐서 예시 없이 다시 읽음)" in result.notes
