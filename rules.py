"""규칙 기반 추출기 — 로컬 LLM 서버가 없는 컴퓨터에서 쓰는 대체 추출기.

LocalLLM과 같은 인터페이스(complete_json, status, aclose)를 갖고, LLM에게 맡긴 것과 같은 일만 한다:
문장에서 값과 단위를 적힌 그대로 뽑고 문장의 종류(plan·revise·question·other)를 표시한다.
단위 변환, 용어 연결, 근거 검사, 규칙 검증은 그대로 온톨로지와 Planner.normalize가 한다.
읽을 단위 이름·항목 이름·재료 이름도 코드에 적지 않고 온톨로지에서 가져온다.

쓰는 때
- LLM을 돌릴 수 없는 PC(메모리 부족, LM Studio 미설치, 클라우드 컨테이너)에서 채팅 흐름 전체를 시험할 때:
  PROCESS_LLM_URL=rules
- 논문의 비교 기준(규칙 기반 vs 로컬 LLM) : python3 start.py eval --url rules

한계: 미리 정한 표현만 읽는다. 처음 보는 말투, 긴 문단, 영어 문장은 LLM보다 많이 놓친다.
그래서 자동으로 켜지지 않는다. 주소를 'rules'로 적었을 때만 쓰고 화면 배지에도 "규칙 기반"이라고 표시한다.
"""
import re
from dataclasses import dataclass

from rdflib.namespace import SKOS

from .knowledge import VD, KnowledgeBase

MODEL_ID = "rule-extractor"
RULES_URLS = ("rules", "rules://", "builtin", "none", "off", "")

NUMBER = r"\d+(?:\.\d+)?"
QUESTION = re.compile(r"\?|왜|이유|근거|어떻게|어째서|무슨|뭐|어떤|설명|괜찮|맞아|맞나|몇|얼마나|알려|되나|될까|가능|\b(?:why|how|what|can)\b", re.I)
WHATIF = re.compile(r"돼\s*\?|돼요|되나|될까|되는지|가능|안\s*돼|넘어도|해도\s*되")
REVISE = re.compile(r"올려|올리|높여|높이|늘려|늘리|증가|키워|내려|내리|낮춰|낮추|줄여|줄이|감소|바꿔|바꾸|변경|수정|고쳐|"
                    r"절반|반으로|두\s*배|2배|세\s*배|3배|걸로|(?:으로|로)\s*해")
RAMP_BEFORE = re.compile(r"(?:분당|매분|per\s*min(?:ute)?)\s*$", re.I)
RAMP_AFTER = re.compile(r"^\s*씩")
COOL = re.compile(r"냉각|식히|식혀|식힌|꺼내|cool", re.I)
MIN_CUE = re.compile(r"최소|at\s*least|min(?:imum)?\s*hold", re.I)
PUMP_CUE = re.compile(r"배기|펌핑|pump", re.I)
WINDOW_CUE = re.compile(r"판정|구간|window", re.I)
NAME_CUE = re.compile(r"(?:실험\s*(?:이름|명)|레시피\s*(?:이름|명))\s*[:은는]?\s*([A-Za-z0-9가-힣_#./-]+)")
SAMPLE_CUE = re.compile(r"(?:시편\s*(?:번호|ID|아이디)|샘플\s*(?:ID|번호)|sample\s*id)\s*[:은는]?\s*([A-Za-z0-9가-힣_#./-]+)", re.I)
SUBSTRATE_ON = re.compile(r"^\s*([가-힣A-Za-z]+(?:\s[가-힣A-Za-z]+)?)\s+위에?\s")
ORDER_NEGATED = re.compile(r"\s*(?:말고|대신|보다)")
ASKED = re.compile(r"\((\w+)\)")
BARE_NUMBER = re.compile(rf"(?<![\w.\-]){NUMBER}(?![\w.])")


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


@dataclass
class Quantity:
    value: float
    unit_text: str
    kind: object  # 온톨로지의 물리량 종류 (qudt QuantityKind IRI)
    start: int
    end: int


def is_rules_url(url: str) -> bool:
    return url.strip().lower() in RULES_URLS


class RuleExtractor:
    """LocalLLM 대신 쓸 수 있는 정규식 추출기. 온톨로지의 단위·항목·재료 사전으로 문장을 읽는다."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.model = MODEL_ID
        self.base_url = "rules"
        labels: list[str] = []
        self._unit_by_label = {}
        for unit in kb.units.values():
            for label in [unit.symbol, *kb.graph.objects(unit.iri, SKOS.altLabel)]:
                self._unit_by_label[squash(str(label))] = unit
                labels.append(str(label))
        labels.sort(key=len, reverse=True)  # '°C/min'이 '°C'보다, 'mTorr'가 'Torr'보다 먼저 잡히게
        self._quantity = re.compile(rf"({NUMBER})\s*({'|'.join(map(re.escape, labels))})(?![A-Za-z])", re.I)
        self._number_params = [p for p in kb.parameters if p.kind == "number"]
        self._kind_of = {p.key: kb.units[p.unit].quantity_kind for p in self._number_params}
        self._cues = {p.key: sorted({squash(n) for n in (p.label, *p.synonyms)}, key=len, reverse=True)
                      for p in self._number_params}

    # ------------------------------------------------------------------ LocalLLM 인터페이스

    async def list_models(self) -> list[str]:
        return [self.model]

    async def resolve_model(self) -> str:
        return self.model

    async def status(self) -> dict:
        return {"ok": True, "url": self.base_url, "model": self.model, "models": [self.model], "error": None,
                "kind": "rules", "note": "로컬 LLM 없이 규칙(정규식)으로 값을 뽑습니다. 정해 둔 표현만 읽습니다."}

    async def complete_json(self, messages: list[dict], schema: dict, name: str = "extraction") -> tuple[dict, str]:
        prompt = messages[-1]["content"]
        head, _, sentence = prompt.rpartition("문장:")
        asked = ASKED.findall(head) if "물어본 항목" in head else []
        return self.extract(sentence.strip(), asked), self.model

    async def aclose(self) -> None:
        pass

    # ------------------------------------------------------------------ 추출

    def extract(self, text: str, asked: list[str] = ()) -> dict:
        """문장 → LLM과 같은 형식의 JSON (모든 항목 키 + intent). 숫자는 {"value", "unit"}으로 적힌 그대로."""
        values: dict = {p.key: None for p in self.kb.parameters}
        quantities = self._merge_compound_time(self._quantities(text))
        for i, q in enumerate(quantities):
            before = text[quantities[i - 1].end if i else 0:q.start]
            after = text[q.end:quantities[i + 1].start if i + 1 < len(quantities) else len(text)]
            if key := self._param_for(q, before, after):
                unit = q.unit_text
                if key == "rampRate" and self._kind_of["rampRate"] != q.kind:
                    unit = "도/분" if unit == "도" else f"{unit}/min"  # '분당 5도' → 승온 속도 단위로
                values[key] = {"value": q.value, "unit": unit}
        if not any(isinstance(v, dict) for v in values.values()):
            self._bare_numbers(text, asked, values)
        self._names(text, values)
        values["intent"] = self._intent(text, values)
        return values

    def _quantities(self, text: str) -> list[Quantity]:
        return [
            Quantity(float(m.group(1)), m.group(2), self._unit_by_label[squash(m.group(2))].quantity_kind, m.start(), m.end())
            for m in self._quantity.finditer(text)
        ]

    def _merge_compound_time(self, quantities: list[Quantity]) -> list[Quantity]:
        """'2시간 30분'처럼 붙어 있는 시간 두 개를 분 단위 하나로 합친다."""
        time_kind = self._kind_of["maxHoldDuration"]
        p = self.kb.param["maxHoldDuration"]
        out: list[Quantity] = []
        for q in quantities:
            prev = out[-1] if out else None
            if prev and prev.kind == time_kind == q.kind and not q.unit_text.strip().startswith(prev.unit_text):
                if q.start - prev.end <= 1 and self._to_base(p, prev) > self._to_base(p, q):
                    total = self._to_base(p, prev) + self._to_base(p, q)
                    out[-1] = Quantity(round(total, 4), p.unit_symbol, time_kind, prev.start, q.end)
                    continue
            out.append(q)
        return out

    def _to_base(self, p, q: Quantity) -> float:
        return self.kb.to_canonical(p, q.value, q.unit_text)[0]

    def _param_for(self, q: Quantity, before: str, after: str) -> str | None:
        candidates = [p for p in self._number_params if self._kind_of[p.key] == q.kind]
        temperature = self._kind_of["targetTemperature"]
        if q.kind == temperature and (RAMP_BEFORE.search(before) or RAMP_AFTER.match(after)):
            return "rampRate"
        # 숫자 앞에 항목 이름이나 동의어가 적혀 있으면 그 항목 (가장 긴 이름이 우선)
        squashed = squash(before)
        best = None
        for p in candidates:
            for cue in self._cues[p.key]:
                pos = squashed.rfind(cue)
                if pos >= 0 and (best is None or len(cue) > len(best[1])):
                    best = (p.key, cue)
                    break
        if best:
            return best[0]
        if len(candidates) == 1:
            return candidates[0].key
        if q.kind == temperature:
            return "coolToTemperature" if COOL.search(before[-12:]) or COOL.search(after[:12]) else "targetTemperature"
        if q.kind == self._kind_of["maxHoldDuration"]:
            if MIN_CUE.search(before):
                return "minHoldDuration"
            if PUMP_CUE.search(before):
                return "pumpTimeout"
            if WINDOW_CUE.search(before) or squash(q.unit_text) in ("s", "초", "sec", "second", "seconds"):
                return "stableWindow"
            return "maxHoldDuration"
        return None

    def _bare_numbers(self, text: str, asked: list[str], values: dict) -> None:
        """단위 없는 숫자: 직전에 물어본 항목의 답('85')이거나, 항목 이름 뒤의 숫자('온도 90')."""
        numbers = BARE_NUMBER.findall(text)
        if not numbers:
            return
        asked = [k for k in asked if k in self.kb.param and self.kb.param[k].kind == "number"]
        if asked and len(numbers) == len(asked) and squash(re.sub(NUMBER, "", text)) in ("", "요", "입니다", "이요", "로", "으로"):
            for key, number in zip(asked, numbers):
                values[key] = {"value": float(number), "unit": ""}
            return
        for m in BARE_NUMBER.finditer(text):
            squashed = squash(text[:m.start()])
            for p in self._number_params:
                if any(squashed.endswith(cue) or squashed.endswith(cue + "는") or squashed.endswith(cue + "은")
                       for cue in self._cues[p.key]):
                    values[p.key] = {"value": float(m.group()), "unit": ""}
                    break

    def _names(self, text: str, values: dict) -> None:
        entities = self.kb.link_entities(text)
        # 'IPA5kPa80도'처럼 붙여 쓴 문장은 글자와 숫자 사이를 띄운 뒤 다시 찾는다
        spaced = self.kb.link_entities(re.sub(r"(?<=[A-Za-z가-힣])(?=\d)|(?<=\d)(?=[A-Za-z가-힣])", " ", text))
        for key, entity_type in (("solventName", VD.Solvent), ("substrateName", VD.Substrate)):
            found = [e for e in entities if e.type == entity_type] or [e for e in spaced if e.type == entity_type]
            if len(found) == 1:
                values[key] = found[0].surface
        if values["substrateName"] is None and (m := SUBSTRATE_ON.match(text)):
            # '구리 포일 위 NMP 시편'처럼 온톨로지에 없는 기판은 적힌 그대로 (재료·일반 명사가 아닐 때만)
            if not self.kb.link_entities(m.group(1)):
                values["substrateName"] = m.group(1)
        orders = [e for e in entities if e.type == VD.ProcessOrder]
        codes = [str(self.kb.graph.value(e.iri, VD.code)) for e in orders]
        if len(set(codes)) == 1:
            values["processOrder"] = codes[0]
        elif len(orders) == 2 and ORDER_NEGATED.match(text[text.find(orders[0].surface) + len(orders[0].surface):]):
            values["processOrder"] = codes[1]  # '진공 먼저 말고 가열 먼저'
        if m := NAME_CUE.search(text):
            values["recipeName"] = m.group(1).rstrip(".,")
        if m := SAMPLE_CUE.search(text):
            sample = m.group(1).rstrip(".,")
            pattern = self.kb.param["sampleId"].value_pattern
            if not pattern or re.fullmatch(pattern, sample):
                values["sampleId"] = sample

    def _intent(self, text: str, values: dict) -> str:
        has_values = any(v is not None for k, v in values.items() if k != "intent")
        numeric = any(isinstance(v, dict) for v in values.values())
        if QUESTION.search(text) and (not numeric or WHATIF.search(text)):
            return "question"
        if has_values and REVISE.search(text):
            return "revise"
        return "plan" if has_values else "other"


def make_extractor(kb: KnowledgeBase, url: str, model: str = "", timeout_s: float = 120.0, reasoning_effort: str = "none"):
    """설정에 따라 로컬 LLM 클라이언트나 규칙 기반 추출기를 만든다. 서버와 평가 도구가 함께 쓴다."""
    if is_rules_url(url):
        return RuleExtractor(kb)
    from .llm import LocalLLM

    return LocalLLM(url, model, timeout_s, reasoning_effort=reasoning_effort)
