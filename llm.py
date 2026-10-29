"""로컬 LLM 서버 클라이언트.

OpenAI 호환 Chat Completions API(/v1/chat/completions)를 쓰므로 LM Studio, Ollama,
llama.cpp server, vLLM 어디에나 붙는다. 개발은 노트북의 LM Studio로 하고,
나중에 GPU 서버의 vLLM으로 옮겨도 주소(PROCESS_LLM_URL)만 바꾸면 된다.

응답은 JSON 스키마로 강제한다(구조화 출력). 서버가 스키마 강제를 지원하지 않으면
프롬프트 지시만으로 다시 요청하고, 응답에서 JSON 객체를 찾아 쓴다.

추론형 모델(Gemma 4 등)은 답하기 전에 "생각 과정"을 수백 토큰 쓰는데, 값 추출에는 필요 없고
응답만 몇 배 느려진다. 그래서 reasoning_effort를 기본 "none"으로 보낸다. 이 옵션을 모르는
서버가 4xx로 거부하면 빼고 다시 보낸다.
"""
import json
import re

import httpx


class LLMError(Exception):
    pass


class LLMHTTPError(LLMError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class LLMFormatError(LLMError):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def parse_json_object(text: str) -> dict:
    """모델 출력에서 JSON 객체를 꺼낸다. 추론 태그(<think>)와 코드 블록 표시는 무시한다."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise LLMFormatError("응답에 JSON 객체가 없습니다.", text)
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMFormatError(f"JSON을 해석할 수 없습니다: {e}", text) from e
    if not isinstance(data, dict):
        raise LLMFormatError("응답이 JSON 객체가 아닙니다.", text)
    return data


def _is_chat_model(model_id: str) -> bool:
    return "embed" not in model_id.lower()


class LocalLLM:
    def __init__(
        self,
        base_url: str,
        model: str = "",
        timeout_s: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
        reasoning_effort: str = "none",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort  # 빈 문자열이면 보내지 않는다
        self._schema_supported = True
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s, transport=transport)

    async def list_models(self) -> list[str]:
        response = await self._request("GET", "/models")
        return [m["id"] for m in response.json().get("data", [])]

    async def resolve_model(self) -> str:
        if self.model:
            return self.model
        models = [m for m in await self.list_models() if _is_chat_model(m)]
        if not models:
            raise LLMError("로컬 LLM 서버에 불러온 모델이 없습니다. LM Studio에서 모델을 불러오세요.")
        return models[0]

    async def status(self) -> dict:
        try:
            models = await self.list_models()
        except LLMError as e:
            return {"ok": False, "url": self.base_url, "model": self.model or None, "models": [], "error": str(e), "kind": "llm"}
        model = self.model or next((m for m in models if _is_chat_model(m)), None)
        error = None
        if model is None:
            error = "서버에 불러온 모델이 없습니다."
        elif self.model and self.model not in models:
            error = f"서버에 '{self.model}' 모델이 없습니다."
        return {"ok": error is None, "url": self.base_url, "model": model, "models": models, "error": error, "kind": "llm"}

    async def complete_json(self, messages: list[dict], schema: dict, name: str = "extraction") -> tuple[dict, str]:
        model = await self.resolve_model()
        payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 2048}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self._schema_supported:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            }
        # 선택 옵션을 모르는 서버가 4xx로 거부하면 옵션을 하나씩, 그다음 둘 다 빼고 다시 요청한다.
        # 성공한 조합을 기억해서 다음 요청부터는 처음부터 그렇게 보낸다.
        optional = [k for k in ("reasoning_effort", "response_format") if k in payload]
        variants = [payload] + [{k: v for k, v in payload.items() if k != key} for key in optional]
        if len(optional) == 2:
            variants.append({k: v for k, v in payload.items() if k not in optional})
        for i, body in enumerate(variants):
            try:
                response = await self._request("POST", "/chat/completions", json=body)
                break
            except LLMHTTPError as e:
                if not 400 <= e.status_code < 500 or i == len(variants) - 1:
                    raise
        if "reasoning_effort" not in body:
            self.reasoning_effort = ""
        if "response_format" not in body:
            # 구조화 출력을 지원하지 않는 서버: 프롬프트 지시만으로 요청한다
            self._schema_supported = False
        try:
            content = response.json()["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError) as e:
            raise LLMError(f"로컬 LLM 응답 형식을 알 수 없습니다: {response.text[:200]}") from e
        return parse_json_object(content), model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise LLMError(
                f"로컬 LLM 서버({self.base_url})에 연결할 수 없습니다. "
                "LM Studio에서 서버를 켜거나 'ollama serve'를 실행하세요."
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError(
                f"로컬 LLM이 {self.timeout_s:g}초 안에 답하지 않았습니다. "
                "더 작은 모델을 쓰거나 PROCESS_LLM_TIMEOUT을 늘리세요."
            ) from e
        except httpx.HTTPError as e:
            raise LLMError(f"로컬 LLM 서버와 통신하지 못했습니다: {e}") from e
        if response.status_code >= 400:
            raise LLMHTTPError(
                f"로컬 LLM 서버 오류 ({response.status_code}): {response.text[:300]}", response.status_code
            )
        return response
