import asyncio

from fakes import FakeLLM, extraction

from process_ai.evaluation.run_eval import evaluate, format_report, load_cases, matches, rescore
from process_ai.llm import LLMError
from process_ai.planner import Planner


def test_cases_file_is_valid(kb):
    cases = load_cases()
    keys = {p.key for p in kb.parameters}
    assert len(cases) >= 10 and len({c["id"] for c in cases}) == len(cases)
    for case in cases:
        assert set(case["expect"]) <= keys, case["id"]


def test_scoring_counts_correct_hallucinated_and_errors(kb):
    cases = [
        {"id": "ok", "message": "IPA, 5 kPa, 80도", "expect": {"solventName": "이소프로판올", "targetPressure": 5, "targetTemperature": 80}},
        {"id": "made-up", "message": "아세톤 3 kPa", "expect": {"solventName": "아세톤", "targetPressure": 3}},
        {"id": "down", "message": "80도", "expect": {"targetTemperature": 80}},
    ]
    llm = FakeLLM(
        extraction(kb, solventName="IPA", targetPressure={"value": 5, "unit": "kPa"}, targetTemperature={"value": 80, "unit": "도"}),
        extraction(kb, solventName="아세톤", targetPressure={"value": 3, "unit": "kPa"}, targetTemperature={"value": 80, "unit": "도"}),
        LLMError("연결 안 됨"),
    )
    report = asyncio.run(evaluate(Planner(kb, llm), cases))

    # 문장에 없는 80도는 근거 검사가 빼므로 환각으로 남지 않는다. 검사를 빼고 세면 1건이다
    assert (report["passed"], report["hallucinations"], report["errors"]) == (2, 0, 1)
    assert report["hallucinations_no_guard"] == 1
    assert report["rows"][1]["hallucinated_no_guard"] == ["targetTemperature"]
    assert report["field_accuracy"] == round(5 / 6, 3)
    text = format_report([report])
    assert "| fake-4b | 2/3 | 83.3% | 0 | 1 |" in text and "- down: 연결 안 됨" in text


def test_no_fill_score_and_categories_show_ontology_contribution(kb):
    cases = [
        {"id": "full", "category": "qualitative", "message": "톨루엔 시편을 풀 진공 걸고 90도로 말려",
         "expect": {"solventName": "톨루엔", "targetPressure": 1.0, "targetTemperature": 90}},
        {"id": "plain", "category": "basic", "message": "IPA, 5 kPa, 80도",
         "expect": {"solventName": "이소프로판올", "targetPressure": 5, "targetTemperature": 80}},
    ]
    # 첫 사례는 LLM이 '풀 진공'(숫자 없음)을 null로 두고, 온톨로지 사전이 펌프 최저 압력으로 채운다
    llm = FakeLLM(
        extraction(kb, solventName="톨루엔", targetTemperature={"value": 90, "unit": "도"}),
        extraction(kb, solventName="IPA", targetPressure={"value": 5, "unit": "kPa"}, targetTemperature={"value": 80, "unit": "도"}),
    )
    report = asyncio.run(evaluate(Planner(kb, llm), cases))

    assert (report["field_accuracy"], report["field_accuracy_no_fill"]) == (1.0, round(5 / 6, 3))
    assert (report["passed"], report["passed_no_fill"]) == (2, 1)
    assert report["by_category"]["qualitative"]["field_accuracy_no_fill"] == round(2 / 3, 3)
    assert report["by_category"]["basic"]["field_accuracy_no_fill"] == 1.0
    assert "| 정성 표현·공정 순서 | 1 | 100.0% (66.7%) |" in format_report([report])


def test_cases_have_known_categories():
    from process_ai.evaluation.run_eval import CATEGORY_LABELS

    assert {c.get("category") for c in load_cases()} <= set(CATEGORY_LABELS)


def test_matches():
    assert matches(1.333, 1.3332) and not matches(1.333, 1.4)
    assert matches("IPA", "ipa") and not matches(5, "5")


def test_rescore_reuses_saved_llm_output_without_calling_the_llm(kb):
    cases = [{"id": "hold", "category": "unit", "message": "압력 50 mbar, 가열 온도 70°C, 최대 1시간 유지",
              "expect": {"targetPressure": 5, "targetTemperature": 70, "maxHoldDuration": 60}}]
    # 모델이 규칙과 달리 '1시간'을 '60분'으로 바꿔 냈다. 값은 맞으므로 근거 검사가 받아 줘야 한다
    llm = FakeLLM(extraction(kb, targetPressure={"value": 50, "unit": "mbar"}, targetTemperature={"value": 70, "unit": "°C"},
                             maxHoldDuration={"value": 60, "unit": "분"}))
    planner = Planner(kb, llm)
    saved = asyncio.run(evaluate(planner, cases))
    again = rescore(Planner(kb, llm=None), cases, saved | {"tag": "v1"})

    assert saved["passed"] == again["passed"] == 1 and again["rescored_from"] == "v1"
    assert again["rows"][0]["latency_s"] == saved["rows"][0]["latency_s"]


def test_intent_is_scored_as_one_field(kb):
    draft = {"values": {"targetPressure": 5, "targetTemperature": 80}}
    cases = [
        {"id": "q", "category": "question", "message": "왜 80도야?", "draft": draft, "expect": {}, "intent": "question"},
        {"id": "r", "category": "question", "message": "90도로", "draft": draft, "expect": {"targetTemperature": 90}, "intent": "revise"},
    ]
    llm = FakeLLM(extraction(kb, intent="plan"), extraction(kb, intent="plan", targetTemperature={"value": 90, "unit": "도"}))
    report = asyncio.run(evaluate(Planner(kb, llm), cases))
    assert report["passed"] == 2 and report["field_accuracy"] == 1.0
    assert [r["intent"] for r in report["rows"]] == ["question", "revise"]


def test_values_in_a_question_are_not_counted_as_hallucinations(kb):
    # "이 조건이면 PET 괜찮아?"에서 온톨로지 사전이 채운 기판 값은 채팅에서 버려지므로 환각으로 세지 않는다
    draft = {"values": {"substrateName": "PET 필름", "targetPressure": 5, "targetTemperature": 80}}
    cases = [{"id": "q", "category": "question", "message": "이 조건이면 PET 괜찮아?", "draft": draft, "expect": {}, "intent": "question"}]
    report = asyncio.run(evaluate(Planner(kb, FakeLLM(extraction(kb, intent="question"))), cases))
    assert report["passed"] == 1 and report["hallucinations"] == 0 and report["field_accuracy_no_fill"] == 1.0


def test_evaluate_can_resume_from_a_partial_file(kb, tmp_path):
    cases = [
        {"id": "a", "message": "IPA, 5 kPa, 80도", "expect": {"solventName": "이소프로판올", "targetPressure": 5, "targetTemperature": 80}},
        {"id": "b", "message": "아세톤 3 kPa 60도", "expect": {"solventName": "아세톤", "targetPressure": 3, "targetTemperature": 60}},
    ]
    partial = tmp_path / "partial.jsonl"
    # 첫 실행: 사례 a 뒤에 LLM이 끊긴다 (오류 행도 저장된다)
    first = FakeLLM(extraction(kb, solventName="IPA", targetPressure={"value": 5, "unit": "kPa"}, targetTemperature={"value": 80, "unit": "도"}),
                    LLMError("끊김"))
    asyncio.run(evaluate(Planner(kb, first), cases, partial))
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 2
    # 오류 행은 지우고 다시 실행하면 a는 건너뛰고 b만 LLM을 부른다
    lines = [l for l in partial.read_text(encoding="utf-8").splitlines() if '"error"' not in l]
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    second = FakeLLM(extraction(kb, solventName="아세톤", targetPressure={"value": 3, "unit": "kPa"}, targetTemperature={"value": 60, "unit": "도"}))
    report = asyncio.run(evaluate(Planner(kb, second), cases, partial))
    assert report["passed"] == 2 and len(second.calls) == 1
