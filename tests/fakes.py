"""테스트용 가짜 로컬 LLM. 정해 둔 추출 결과를 순서대로 돌려준다."""


class FakeLLM:
    def __init__(self, *responses, model: str = "fake-4b"):
        self.responses = list(responses)
        self.model = model
        self.calls: list[list[dict]] = []

    async def complete_json(self, messages, schema, name="extraction"):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, self.model

    async def status(self):
        return {"ok": True, "url": "fake://", "model": self.model, "models": [self.model], "error": None}


def extraction(kb, **values) -> dict:
    """모든 항목이 null인 추출 결과에 주어진 값만 채운다."""
    return {p.key: None for p in kb.parameters} | values
