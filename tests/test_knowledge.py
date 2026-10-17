import json

import pytest
from rdflib import RDF, Graph

from process_ai.knowledge import VD, KnowledgeBase, UnitError
from process_ai.recipe import Recipe, field_paths


def recipe(**overrides) -> Recipe:
    data = {"name": "테스트", "vacuum": {"target_kpa": 5}, "heating": {"target_c": 80}}
    data.update(overrides)
    return Recipe.model_validate(data)


def test_ontology_passes_integrity_rules(kb):
    assert kb.check_integrity() == []


def test_integrity_rules_catch_missing_equipment_value(tmp_path):
    broken = KnowledgeBase(runs_dir=tmp_path)
    broken.graph.remove((VD.StageHeater01, VD.maxRampRate, None))
    assert any("최대 승온 속도" in m for m in broken.check_integrity())


def test_limits_come_from_equipment_ontology(kb):
    limits = kb.limits()
    assert (limits.max_temp_c, limits.max_ramp_c_per_min, limits.min_pressure_kpa) == (200, 20, 1.0)
    assert (limits.safe_open_temp_c, limits.door_timeout_s, limits.max_process_min) == (60, 20, 240)
    assert kb.max_temperature() == (200.0, "가열 스테이지 히터")


def test_every_recipe_field_is_defined_in_ontology(kb):
    assert [p.json_field for p in kb.parameters] == field_paths()
    assert {p.key for p in kb.parameters if p.required} == {"targetPressure", "targetTemperature"}


def test_entity_linking_prefers_longest_label_and_respects_word_boundaries(kb):
    found = kb.link_entities("IPA를 쓴 PET 기판 시편을 최대 진공으로 건조하고 ITO 유리와 비교, 물성은 따로 확인")
    assert [(e.label, e.surface) for e in found] == [
        ("이소프로판올", "IPA"),
        ("PET 필름", "PET 기판"),
        ("시편", "시편"),  # 일반 명사(vd:GenericTerm)도 찾는다. 근거 검사가 재료 이름과 구분할 때 쓴다
        ("최대 진공", "최대 진공"),
        ("ITO 유리", "ITO 유리"),
    ]
    assert kb.link_name("isopropyl alcohol", VD.Solvent).label == "이소프로판올"
    assert kb.link_name("PGMEA", VD.Solvent).label == "PGMEA"
    assert kb.link_name("DMSO", VD.Solvent) is None


@pytest.mark.parametrize(
    "key, value, unit, expected",
    [
        ("targetPressure", 10, "Torr", 1.333),
        ("targetPressure", 50, "mbar", 5.0),
        ("targetPressure", 1, "기압", 101.3),
        ("targetPressure", 5, "kPa", 5.0),
        ("targetTemperature", 80, "℃", 80.0),
        ("maxHoldDuration", 1.5, "시간", 90.0),
        ("stableWindow", 1, "분", 60.0),
        ("rampRate", 120, "°C/h", 2.0),
    ],
)
def test_unit_conversion_uses_ontology_units(kb, key, value, unit, expected):
    converted, _ = kb.to_canonical(kb.param[key], value, unit)
    assert converted == pytest.approx(expected, rel=1e-3)


def test_unit_errors(kb):
    with pytest.raises(UnitError, match="쓸 수 없는 단위"):
        kb.to_canonical(kb.param["targetPressure"], 5, "분")
    with pytest.raises(UnitError, match="알 수 없습니다"):
        kb.to_canonical(kb.param["targetPressure"], 5, "psi")
    assert "단위가 없어" in kb.to_canonical(kb.param["targetTemperature"], 80, "")[1]


def test_default_recipe_has_no_errors(kb):
    assert [i for i in kb.validate(recipe()) if i.level == "error"] == []


@pytest.mark.parametrize(
    "overrides, level, field, fragment",
    [
        ({"heating": {"target_c": 250}}, "error", "heating.target_c", "가열 스테이지 히터의 최대 사용 온도 200°C"),
        ({"substrate": "PET", "heating": {"target_c": 130}}, "error", "heating.target_c", "PET 필름의 최대 사용 온도 120°C"),
        ({"vacuum": {"target_kpa": 0.5}}, "error", "vacuum.target_kpa", "최저 도달 압력 1 kPa"),
        ({"vacuum": {"target_kpa": 101.3}}, "error", "vacuum.target_kpa", "진공이 아닙니다"),
        ({"heating": {"target_c": 80, "ramp_c_per_min": 30}}, "error", "heating.ramp_c_per_min", "최대 승온 속도 20°C/분"),
        ({"endpoint": {"min_hold_min": 10, "max_hold_min": 5}}, "error", "endpoint.min_hold_min", "최소 유지 시간보다"),
        ({"endpoint": {"max_hold_min": 300}}, "error", "endpoint.max_hold_min", "장비 허용 시간 240분"),
        ({"endpoint": {"stable_window_s": 600, "max_hold_min": 5}}, "error", "endpoint.stable_window_s", "최대 유지 시간 5분보다"),
        ({"finish": {"cool_to_c": 70}}, "error", "finish.cool_to_c", "개방 허용 온도 60°C"),
        ({"name": "  "}, "error", "name", "실험 이름"),
        ({"endpoint": {"stable_window_s": 5}}, "warning", "endpoint.stable_window_s", "10초보다 짧으면"),
        ({"heating": {"target_c": 28}}, "warning", "heating.target_c", "실온"),
        ({"solvent": "DMSO"}, "warning", "solvent", "온톨로지에 없는 용매"),
        ({"substrate": "사파이어"}, "warning", "substrate", "온톨로지에 없는 기판"),
        ({"solvent": "IPA"}, "info", "heating.target_c", "이소프로판올의 인화점은 12°C"),
    ],
)
def test_shacl_rules(kb, overrides, level, field, fragment):
    issues = kb.validate(recipe(**overrides))
    assert any(i.level == level and i.field == field and fragment in i.message for i in issues), issues


def test_facts_include_solvent_properties_and_boiling_point_at_pressure(kb):
    facts = kb.facts(recipe(solvent="IPA", substrate="PET"))
    assert facts[0] == "이 장비의 최대 온도는 200°C입니다 (가열 스테이지 히터 기준)."
    assert "이소프로판올 (CAS 67-63-0): 끓는점 82.5°C, 인화점 12°C" in facts
    assert any("PET 필름의 최대 사용 온도는 120°C" in f for f in facts)
    boiling = kb.boiling_point_at(VD.Isopropanol, 101.325)
    assert boiling == pytest.approx(82.5, abs=0.5)
    assert kb.boiling_point_at(VD.Isopropanol, 5) == pytest.approx(22.6, abs=1)


def write_run(runs_dir, run_id, solvent, target_c, result="COMPLETE", drying=4.0, detected=True):
    run = runs_dir / run_id
    run.mkdir(parents=True)
    body = recipe(solvent=solvent, heating={"target_c": target_c}).model_dump()
    (run / "recipe.json").write_text(json.dumps({"recipe": body, "approved_by": "테스트"}), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({
        "result": result, "endpoint_detected": detected, "drying_min": drying, "process_min": 20.0,
        "max_temp_c": target_c, "min_pressure_kpa": 4.0, "reason": None,
        "started_at": f"2026-09-{run_id[-2:]}T10:00:00", "ended_at": f"2026-09-{run_id[-2:]}T10:20:00",
    }), encoding="utf-8")


def test_similar_runs_and_knowledge_graph_export(tmp_path):
    kb = KnowledgeBase(runs_dir=tmp_path)
    write_run(tmp_path, "run-01", "IPA", 80, drying=4.0)
    write_run(tmp_path, "run-02", "이소프로판올", 90, drying=6.0)
    write_run(tmp_path, "run-03", "IPA", 150)  # 온도 차이가 커서 제외
    write_run(tmp_path, "run-04", "에탄올", 80)  # 용매가 달라 제외
    write_run(tmp_path, "run-05", "IPA", 75, result="ABORTED", drying=None, detected=None)

    similar = kb.similar_runs(recipe(solvent="isopropanol"))
    assert [r["run_id"] for r in similar["runs"]] == ["run-05", "run-02", "run-01"]
    assert "완료 2건" in similar["summary"] and "평균 건조 5.0분" in similar["summary"]
    assert kb.similar_runs(recipe(solvent="톨루엔"))["count"] == 0

    exported = Graph().parse(data=kb.export_turtle(), format="turtle")
    assert len(list(exported.subjects(RDF.type, VD.ProcessRun))) == 5


def test_ontology_defaults_match_recipe_defaults(kb):
    from process_ai.recipe import default_for

    with_default = [p for p in kb.parameters if p.default_value is not None]
    assert len(with_default) == 7
    for p in with_default:
        assert default_for(p.json_field) == p.default_value, p.key
        assert p.default_rationale, p.key
