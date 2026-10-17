import time

from fakes import FakeLLM, extraction
from fastapi.testclient import TestClient
from rdflib import Graph

from process_ai.controller import ProcessController
from process_ai.llm import LLMError
from process_ai.planner import Planner
from process_ai.server import create_app
from process_ai.simulator import SimulatedChamber

RECIPE = {"name": "IPA 건조", "solvent": "IPA", "vacuum": {"target_kpa": 5}, "heating": {"target_c": 80}}


def make_client(tmp_path, kb, *llm_responses):
    sim = SimulatedChamber(speed=200)
    controller = ProcessController(sim, kb.limits(), kb.validate, tick_s=0.01, runs_dir=tmp_path)
    planner = Planner(kb, FakeLLM(*llm_responses))
    return TestClient(create_app(controller, planner, kb, warm_up=False))  # 예열은 가짜 응답을 소모하므로 끈다


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("시간 초과")


def test_validate_returns_ontology_review(tmp_path, kb):
    with make_client(tmp_path, kb) as client:
        review = client.post("/api/recipe/validate", json=RECIPE).json()
        assert review["ok"] and review["max_active_min"] > 0
        assert review["recipe"]["endpoint"]["max_hold_min"] == 60
        assert any("CAS 67-63-0" in f for f in review["facts"])
        assert any(i["level"] == "info" for i in review["issues"])
        assert review["similar_runs"]["count"] == 0

        bad = RECIPE | {"substrate": "PET", "heating": {"target_c": 150}}
        review = client.post("/api/recipe/validate", json=bad).json()
        assert not review["ok"]


def test_start_and_abort_run(tmp_path, kb):
    with make_client(tmp_path, kb) as client:
        wait_for(lambda: client.get("/api/status").json()["telemetry"] is not None)
        bad = RECIPE | {"heating": {"target_c": 999}}
        assert client.post("/api/runs", json={"recipe": bad, "approved_by": "홍길동"}).status_code == 409

        r = client.post("/api/runs", json={"recipe": RECIPE, "approved_by": "홍길동"})
        assert r.status_code == 200 and r.json()["run_id"]
        assert client.post("/api/runs", json={"recipe": RECIPE, "approved_by": "홍길동"}).status_code == 409
        wait_for(lambda: client.get("/api/status").json()["phase"] not in ("IDLE", "CLOSING_DOOR"))
        assert client.post("/api/runs/abort").json() == {"aborted": True}
        wait_for(lambda: client.get("/api/status").json()["phase"] == "ABORTED")


def test_chat_uses_local_llm_and_keeps_draft(tmp_path, kb):
    responses = (
        extraction(kb, targetTemperature={"value": 80, "unit": "°C"}),
        extraction(kb, targetPressure={"value": 5, "unit": "kPa"}),
        LLMError("로컬 LLM 서버에 연결할 수 없습니다."),
    )
    with make_client(tmp_path, kb, *responses) as client:
        first = client.post("/api/chat", json={"message": "80도"}).json()
        assert first["recipe"] is None and first["draft"]["asked"] == ["targetPressure"]
        second = client.post("/api/chat", json={"message": "5 kPa", "draft": first["draft"]}).json()
        assert second["recipe"]["heating"]["target_c"] == 80 and second["model"] == "fake-4b"
        r = client.post("/api/chat", json={"message": "다시"})
        assert r.status_code == 502 and "연결할 수 없습니다" in r.json()["detail"]
        assert client.get("/api/llm/status").json()["ok"] is True


def test_knowledge_endpoints(tmp_path, kb):
    with make_client(tmp_path, kb) as client:
        materials = client.get("/api/knowledge/materials").json()
        assert "이소프로판올" in materials["solvents"] and "PET 필름" in materials["substrates"]
        r = client.get("/api/knowledge/export")
        assert r.headers["content-type"].startswith("text/turtle")
        assert len(Graph().parse(data=r.text, format="turtle")) > 500


def test_websocket_sends_status_then_updates(tmp_path, kb):
    with make_client(tmp_path, kb) as client:
        with client.websocket_connect("/ws/process") as ws:
            assert ws.receive_json()["type"] == "status"
            assert ws.receive_json()["type"] in ("telemetry", "event")


def test_approval_records_operator_feedback(tmp_path, kb):
    from process_ai.examples import ExampleStore

    sim = SimulatedChamber(speed=200)
    controller = ProcessController(sim, kb.limits(), kb.validate, tick_s=0.01, runs_dir=tmp_path)
    store = ExampleStore(tmp_path / "examples.jsonl", kb)
    planner = Planner(kb, FakeLLM(extraction(kb, solventName="IPA", targetPressure={"value": 5, "unit": "kPa"},
                                             targetTemperature={"value": 80, "unit": "도"})), store)
    with TestClient(create_app(controller, planner, kb, warm_up=False)) as client:
        chat = client.post("/api/chat", json={"message": "IPA 시편 5 kPa 80도"}).json()
        assert chat["values"] == {"solventName": "이소프로판올", "targetPressure": 5.0, "targetTemperature": 80.0}
        final = dict(chat["recipe"]); final["heating"] = dict(final["heating"], target_c=75)  # 작업자가 80 → 75로 고침
        started = client.post("/api/runs", json={
            "recipe": final, "approved_by": "홍길동",
            "feedback": {"turns": [{"message": "IPA 시편 5 kPa 80도", "extracted": chat["extracted"], "values": chat["values"]}],
                         "ai_recipe": chat["recipe"]},
        }).json()
        assert started["example_id"] and started["corrected"] is True
        assert client.get("/api/examples/stats").json() == {"count": 1, "single_turn": 1, "corrected": 1}
        client.post("/api/runs/abort")
    saved = store.load()[0]
    assert saved["changed"] == {"targetTemperature": 75.0} and saved["approved_by"] == "홍길동"
