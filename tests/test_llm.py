import asyncio
import json

import httpx
import pytest

from process_ai.llm import LLMError, LLMFormatError, LocalLLM, parse_json_object


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def make_llm(handler, model=""):
    return LocalLLM("http://llm.test/v1", model=model, transport=httpx.MockTransport(handler))


def test_complete_json_uses_schema_and_first_chat_model():
    seen = []

    def handler(request: httpx.Request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "text-embedding-nomic"}, {"id": "qwen3-4b"}]})
        seen.append(json.loads(request.content))
        return chat_response('<think>음</think>\n```json\n{"a": 1}\n```')

    data, model = asyncio.run(make_llm(handler).complete_json([{"role": "user", "content": "x"}], {"type": "object"}))
    assert (data, model) == ({"a": 1}, "qwen3-4b")
    assert seen[0]["model"] == "qwen3-4b"
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["temperature"] == 0
    assert seen[0]["reasoning_effort"] == "none"


def test_falls_back_when_server_rejects_structured_output():
    sent = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        sent.append(("response_format" in body, body.get("reasoning_effort")))
        if "response_format" in body:
            return httpx.Response(400, json={"error": "response_format not supported"})
        return chat_response('{"a": 2}')

    llm = make_llm(handler, model="gemma-3-4b")
    assert asyncio.run(llm.complete_json([], {}))[0] == {"a": 2}
    assert asyncio.run(llm.complete_json([], {}))[0] == {"a": 2}
    # 생각 끄기 옵션은 문제가 아니었으므로 유지하고, 구조화 출력만 뺀다
    assert sent == [(True, "none"), (True, None), (False, "none"), (False, "none")]


def test_falls_back_when_server_rejects_reasoning_option():
    sent = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        sent.append(("response_format" in body, body.get("reasoning_effort")))
        if "reasoning_effort" in body:
            return httpx.Response(422, json={"error": "unknown field reasoning_effort"})
        return chat_response('{"a": 3}')

    llm = make_llm(handler, model="llama-3.2-3b")
    assert asyncio.run(llm.complete_json([], {}))[0] == {"a": 3}
    assert asyncio.run(llm.complete_json([], {}))[0] == {"a": 3}
    assert sent == [(True, "none"), (True, None), (True, None)]


def test_connection_and_server_errors_are_explained():
    def refuse(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="연결할 수 없습니다"):
        asyncio.run(make_llm(refuse, model="m").complete_json([], {}))

    status = asyncio.run(make_llm(refuse).status())
    assert status["ok"] is False and "연결할 수 없습니다" in status["error"]

    def no_models(request):
        return httpx.Response(200, json={"data": []})

    with pytest.raises(LLMError, match="불러온 모델이 없습니다"):
        asyncio.run(make_llm(no_models).complete_json([], {}))

    def listed(request):
        return httpx.Response(200, json={"data": [{"id": "qwen3-4b"}]})

    status = asyncio.run(make_llm(listed, model="llama-8b").status())
    assert status["ok"] is False and "llama-8b" in status["error"]


def test_parse_json_object():
    assert parse_json_object('앞말 {"x": [1, 2]} 뒷말') == {"x": [1, 2]}
    with pytest.raises(LLMFormatError):
        parse_json_object("JSON 없음")
    with pytest.raises(LLMFormatError):
        parse_json_object("{깨진 json}")
