"""채팅으로 받은 실험 계획을 로컬 LLM과 온톨로지로 레시피 초안으로 옮긴다.

역할 분담 — 작은 모델(3~4B)로도 안정적으로 돌리기 위해 LLM에게는 판단이나 계산을 맡기지 않는다.
- 로컬 LLM : 문장에서 값과 단위를 적힌 그대로 뽑고, 문장의 종류(조건 입력·질문·수정·기타)를 표시한다
- 온톨로지 : 추출 항목 정의, 용어 연결, 단위 변환, "최대 진공" 같은 표현 해석, 검증 규칙
- 코드     : 대화 상태 병합, 필수값 확인, 기본값 채우기, 답변 작성, 질문에는 초안의 근거를 설명

작업자가 "왜 그렇게 잡았어?"라고 물으면 초안을 다시 만들지 않고, 단위 변환·용어 연결·기본값·규칙 검사·
온톨로지 지식을 근거로 설명한다. 설명은 LLM이 아니라 코드가 실행 중 생긴 정보로 만들므로 지어낼 수 없다.

AI는 초안만 만든다. 레시피는 검증을 거쳐 작업자가 승인해야 공정 제어기로 넘어간다.
"""
import json
import re
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .knowledge import MATERIAL_TYPES, VD, KnowledgeBase, UnitError
from .llm import LLMError, LLMFormatError
from .recipe import AMBIENT_C, Issue, Recipe, active_minutes_upper_bound, default_for

ENTITY_PARAMS = {"solventName": VD.Solvent, "substrateName": VD.Substrate}
NUMBER = re.compile(r"\d+(?:\.\d+)?")
INTENTS = ("plan", "question", "revise", "other")
# LLM이 종류를 틀려도 잡히도록, 값이 없는 문장에 이런 표현이 있으면 질문으로 본다
QUESTION_PATTERN = re.compile(r"\?|왜|이유|근거|어떻게|어째서|무슨|뭐|어떤|설명|괜찮|맞아|맞나|몇|얼마나|알려|되나|될까|가능")
# 값을 바꾸자는 게 아니라 "그래도 되는지" 묻는 표현. 숫자가 있어도 초안을 바꾸지 않고 가정해서 검사만 한다
WHATIF_PATTERN = re.compile(r"돼\s*\?|돼요|되나|될까|되는지|가능|안\s*돼|넘어도|해도\s*되")
# 상대 변경: "10도 올려줘"는 10도로가 아니라 +10도. 숫자 뒤에 '로/으로/까지'가 붙으면 절대값이다
RELATIVE_UP = re.compile(r"올려|올리|높여|높이|늘려|늘리|더\s|증가|키워|키우")
RELATIVE_DOWN = re.compile(r"내려|내리|낮춰|낮추|줄여|줄이|감소")
MULTIPLIERS = ((r"절반|반으로|반 으로|half", 0.5), (r"두\s*배|2배|double", 2.0), (r"세\s*배|3배", 3.0))
# LLM이 값을 하나도 못 뽑았을 때 "압력 0.5 kPa 낮춰"처럼 숫자+단위가 하나뿐인 문장을 코드가 직접 읽는다
NUMBER_WITH_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(°C/min|℃/분|도/분|[A-Za-z°℃%]+(?:/[A-Za-z]+)?|도|분|초|시간|기압|토르|파스칼|밀리바)")
START_PATTERN = re.compile(r"시작|실행|돌려|가동|run|start", re.IGNORECASE)
RESET_PATTERN = re.compile(r"처음부터|초기화|리셋|새로\s*시작|다시\s*할")
DURATION_PATTERN = re.compile(r"얼마나|걸려|걸리|소요|몇\s*분|몇\s*시간")
NEXT_PATTERN = re.compile(r"다음|순서|어떻게\s*진행|뭐\s*해야|무엇을\s*해야")
# 재료 이름이 나와도 초안 자체를 묻는 질문이면 물성만 답하지 않고 근거까지 설명한다
DRAFT_FOCUS = re.compile(r"조건|초안|레시피|괜찮|맞아|맞나|왜|이유|근거|어떻게|기본값|규칙|한계|안전")


def numbers_in(text: str) -> list[float]:
    """문장에 적힌 숫자들. '3,000'처럼 천 단위 쉼표가 있는 표기도 함께 읽는다."""
    found = [float(n) for n in NUMBER.findall(text)]
    found += [float(n.replace(",", "")) for n in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", text)]
    return found


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def topic(word: str) -> str:
    """'은/는' 조사. 받침이 있으면 '은'."""
    return word + ("은" if (ord(word[-1]) - 0xAC00) % 28 and "가" <= word[-1] <= "힣" else "는")


def copula(word: str) -> str:
    """'이라/라' 연결. 받침이 있거나 한글이 아니면 '이라'."""
    return word + ("라" if "가" <= word[-1] <= "힣" and not (ord(word[-1]) - 0xAC00) % 28 else "이라")


def merge_notes(old: list[str], new: list[str]) -> list[str]:
    """해석 메모를 합친다. 같은 항목('목표 압력: …')의 새 메모가 이전 메모를 대신한다."""
    label = lambda n: n.split(":", 1)[0]
    replaced = {label(n) for n in new}
    return [n for n in old if label(n) not in replaced] + list(new)


LEVEL_LABEL = {"error": "불가", "warning": "경고", "info": "참고"}


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-3 * max(1.0, abs(a), abs(b))


class Quantity(BaseModel):
    value: float
    unit: str


class DraftState(BaseModel):
    """대화 사이에 화면이 들고 있는 상태. 서버는 대화를 저장하지 않는다."""

    values: dict[str, float | str] = Field(default_factory=dict)  # 파라미터 키 → 기준 단위 값
    asked: list[str] = Field(default_factory=list)  # 직전에 작업자에게 물어본 파라미터 키
    notes: list[str] = Field(default_factory=list)  # 값을 어떻게 해석했는지 (단위 변환, 용어 연결 등). 질문에 답할 때 쓴다


class Interpretation(BaseModel):
    values: dict[str, float | str]  # 이번 메시지에서 얻은 값 (기준 단위)
    examples: list[str] = []  # 프롬프트에 예시로 넣은 과거 문장 (작업자 확인)
    notes: list[str]  # 단위 변환, 용어 연결처럼 코드가 해석한 내용
    problems: list[str]  # 이해하지 못했거나 근거 검사로 뺀 내용
    extracted: dict  # LLM 원출력
    model: str
    latency_s: float
    intent: str | None = None  # LLM이 표시한 문장 종류. 최종 판단은 Planner.resolve_intent


class PlanResult(BaseModel):
    reply: str
    draft: DraftState
    recipe: Recipe | None
    values: dict[str, float | str] = Field(default_factory=dict)  # 이번 턴에서 얻은 값. 화면이 승인 때 학습 기록용으로 보낸다
    notes: list[str]
    issues: list[Issue]
    extracted: dict
    model: str


def strict_schema(model: type[BaseModel]) -> dict:
    """구조화 출력용 JSON 스키마: 참조를 펼치고, 모든 필드를 필수(null 허용)로 만든다."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].split("/")[-1]])
        out = {k: walk(v) for k, v in node.items() if k not in ("title", "default")}
        if out.get("type") == "object":
            out["required"] = list(out.get("properties", {}))
            out["additionalProperties"] = False
        return out

    return walk(schema)


class Planner:
    def __init__(self, kb: KnowledgeBase, llm, examples=None, fewshot_k: int = 2):
        self.kb = kb
        self.llm = llm
        self.examples = examples  # ExampleStore: 작업자가 확인한 과거 문장. None이면 쓰지 않는다
        self.fewshot_k = fewshot_k
        self.extraction_model = self._build_extraction_model()
        self.schema = strict_schema(self.extraction_model)
        self.system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------ 온톨로지 → 스키마·프롬프트

    def _build_extraction_model(self) -> type[BaseModel]:
        fields = {}
        for p in self.kb.parameters:
            if p.kind == "number":
                fields[p.key] = (Quantity | None, None)
            elif p.kind == "choice":
                fields[p.key] = (Literal[tuple(code for code, _ in p.choices)] | None, None)
            else:
                fields[p.key] = (str | None, None)
        fields["intent"] = (Literal[INTENTS] | None, None)
        # 모르는 키는 거부해서, 모델이 항목 이름을 틀리면 재요청하게 한다
        return create_model("Extraction", __config__=ConfigDict(extra="forbid"), **fields)

    def _build_system_prompt(self) -> str:
        lines = []
        for p in self.kb.parameters:
            line = f"- {p.key}: {p.label}"
            if p.synonyms:
                line += f" (다른 표현: {', '.join(p.synonyms)})"
            if p.kind == "number":
                line += f" — 숫자와 단위, 기준 단위 {p.unit_symbol}"
            elif p.kind == "choice":
                line += " — " + " 또는 ".join(f'"{code}"({label})' for code, label in p.choices)
            if p.hint:
                line += f". {p.hint}"
            lines.append(line)
        empty = self.extraction_model().model_dump()
        example_full = empty | {
            "intent": "plan",
            "solventName": "IPA",
            "substrateName": "PET 기판",
            "targetPressure": {"value": 5, "unit": "Torr"},
            "targetTemperature": {"value": 80, "unit": "도"},
            "rampRate": {"value": 5, "unit": "도/분"},
            "maxHoldDuration": {"value": 30, "unit": "분"},
        }
        example_short = empty | {
            "intent": "plan",
            "solventName": "메탄올",
            "targetPressure": {"value": 12, "unit": "kPa"},
            "targetTemperature": {"value": 45, "unit": "도"},
        }
        example_question = empty | {"intent": "question"}
        example_revise = empty | {"intent": "revise", "targetTemperature": {"value": 10, "unit": "도"}}
        return f"""너는 진공 가열 건조 장비의 실험 계획 문장에서 공정 조건을 뽑아 JSON으로 옮기는 정보 추출기다.

규칙
1. 문장에 적힌 값만 옮긴다. 언급되지 않은 항목은 null로 둔다. 추측하거나 기본값을 넣지 않는다.
2. 숫자 항목은 {{"value": 숫자, "unit": "문장에 적힌 단위"}}로 쓴다. 단위를 바꾸거나 계산하지 않는다. 단위가 없으면 unit은 "".
3. 문장의 숫자를 하나도 빠뜨리지 말고, 숫자마다 알맞은 항목 하나에만 넣는다. 같은 숫자를 두 항목에 넣지 않는다.
4. "최대 진공"처럼 숫자가 없는 표현은 null로 둔다.
5. 용매와 기판은 문장에 적힌 이름 그대로 쓴다.
6. 예시는 형식만 보여준다. 예시의 값을 옮기지 않는다.
7. intent에 문장의 종류를 적는다. "plan": 공정 조건을 말함, "revise": 이미 말한 값을 바꿈, "question": 지금 초안이나 값에 대해 물음, "other": 인사·잡담 등. 질문이면 값 항목은 모두 null.
8. JSON만 출력한다.

항목
{chr(10).join(lines)}

알려진 용매: {', '.join(self.kb.entity_names(VD.Solvent))}
알려진 기판: {', '.join(self.kb.entity_names(VD.Substrate))}

예시 1
문장: PET 기판에 IPA 용매 시편, 5 Torr까지 뽑고 분당 5도로 80도까지 올려서 최대 30분 건조
JSON: {json.dumps(example_full, ensure_ascii=False)}

예시 2
문장: 메탄올 12 kPa 45도
JSON: {json.dumps(example_short, ensure_ascii=False)}

예시 3
문장: 왜 압력을 그렇게 잡았어?
JSON: {json.dumps(example_question, ensure_ascii=False)}

예시 4
문장: 온도 10도 올려줘
JSON: {json.dumps(example_revise, ensure_ascii=False)}"""

    def _user_prompt(self, message: str, asked: list[str]) -> str:
        if not asked:
            return f"문장: {message}"
        labels = ", ".join(f"{self.kb.param[k].label}({k})" for k in asked if k in self.kb.param)
        return f"직전에 작업자에게 물어본 항목: {labels}. 짧은 답은 이 항목의 값이다.\n문장: {message}"

    # ------------------------------------------------------------------ 해석

    async def interpret(self, message: str, draft: DraftState | None = None) -> Interpretation:
        """메시지 하나를 LLM으로 추출하고 온톨로지로 정규화한다."""
        draft = draft or DraftState()
        started = time.perf_counter()
        # 작업자가 확인한 비슷한 과거 문장을 이전 대화 턴으로 넣는다 (되묻기 답변에는 넣지 않는다)
        examples = self.examples.similar(message, self.fewshot_k) if self.examples and not draft.asked else []
        demos = [(e["messages"][0], self.examples.example_json(e)) for e in examples]
        extraction, model = await self._extract(message, draft.asked, demos)
        interp = self.normalize(message, extraction, model, draft=draft)
        if demos and self._lost_required_value(message, draft, interp):
            # 작은 모델은 비슷한 과거 문장을 보면 새 문장의 값을 빼 버리기도 한다. 그러면 예시 없이 다시 뽑는다
            extraction, model = await self._extract(message, draft.asked)
            interp = self.normalize(message, extraction, model, draft=draft)
            interp.notes.append("과거 문장 참고를 취소했어요 (값을 놓쳐서 예시 없이 다시 읽음)")
        else:
            interp.examples = [e["messages"][0] for e in examples]
            if examples:
                interp.notes.append(f"참고한 과거 문장: {len(examples)}개 (작업자가 확인한 기록)")
        interp.latency_s = round(time.perf_counter() - started, 2)
        return interp

    def normalize(
        self, message: str, extraction: BaseModel, model: str = "", ontology_fill: bool = True, guard: bool = True,
        draft: DraftState | None = None,
    ) -> Interpretation:
        """LLM 추출 결과를 온톨로지로 기준 단위·표준 용어로 바꾼다.

        guard: 문장에 근거가 없는 값(예시를 베낀 값 등)과 종류가 맞지 않는 이름(기판 칸의 용매)을 뺀다.
        ontology_fill=False, guard=False는 평가에서 각 단계의 기여를 재는 비교 실험용이다.
        """
        values: dict[str, float | str] = {}
        notes: list[str] = []
        problems: list[str] = []
        message_numbers = numbers_in(message)
        linked = self.kb.link_entities(message)
        message_entities = {e.iri for e in linked}
        message_orders = {str(self.kb.graph.value(e.iri, VD.code)) for e in linked if e.type == VD.ProcessOrder}

        for p in self.kb.parameters:
            raw = getattr(extraction, p.key)
            if raw is None:
                continue
            if p.kind == "number":
                if guard and not self._number_grounded(p, raw, message_numbers):
                    problems.append(f"{p.label} {raw.value:g} {raw.unit}: 문장에 없는 값이라 뺐습니다.")
                    continue
                try:
                    value, note = self.kb.to_canonical(p, raw.value, raw.unit)
                except UnitError as e:
                    problems.append(str(e))
                    continue
                values[p.key] = value
                if note:
                    notes.append(note)
            elif p.kind == "choice":
                # 공정 순서는 온톨로지가 문장에서 순서 표현을 찾았을 때만 받는다. 모델은 "건조 후 식혀서"의
                # '후'만 보고도 순서를 지어내므로, 표현이 없으면 기본 순서(진공 먼저)로 두고 검토 화면에서 고치게 한다
                if guard and p.key == "processOrder" and raw not in message_orders:
                    problems.append(f"{p.label} '{dict(p.choices).get(raw, raw)}': 문장에 순서 표현이 없어 뺐습니다."
                                    if not message_orders else f"{p.label} '{raw}': 문장의 표현과 달라 뺐습니다.")
                    continue
                values[p.key] = raw
            elif text := raw.strip():
                if p.key in ENTITY_PARAMS:
                    entity = self.kb.link_name(text, ENTITY_PARAMS[p.key])
                    if guard and (problem := self._entity_problem(p, text, entity, message, message_entities)):
                        problems.append(problem)
                        continue
                    if entity and entity.label != text:
                        notes.append(f"{p.label}: '{text}' → {entity.label} (온톨로지 용어 연결)")
                        text = entity.label
                elif guard and (problem := self._text_problem(p, text, message)):
                    problems.append(problem)
                    continue
                values[p.key] = text

        if guard:
            self._drop_duplicate_numbers(message, values, extraction.model_dump(), problems)
        if guard and "minHoldDuration" in values and "maxHoldDuration" not in values:
            # 온톨로지에서 최소 유지 시간의 이름은 모두 '최소'로 시작한다. 그 말이 없는 "유지 시간"은 최대 유지 시간이다
            cues = {n.split()[0] for n in (self.kb.param["minHoldDuration"].label, *self.kb.param["minHoldDuration"].synonyms)}
            if not any(c in message for c in cues) and not re.search(r"at least|minimum|min\.? hold", message, re.I):
                values["maxHoldDuration"] = values.pop("minHoldDuration")
                notes.append(f"최대 유지 시간: '최소'라는 말이 없어 최대 유지 시간으로 해석했습니다.")
        if ontology_fill:
            self._fill_from_ontology(message, values, notes)
        if not any(self.kb.param[k].kind == "number" for k in values):
            self._backup_extract(message, values, notes)
        if draft is not None:
            self._apply_relative(message, values, notes, extraction.model_dump(), draft)
        return Interpretation(
            values=values,
            notes=notes,
            problems=problems,
            extracted=extraction.model_dump(),
            model=model,
            latency_s=0.0,
            intent=getattr(extraction, "intent", None),
        )

    def resolve_intent(self, message: str, interp: Interpretation, draft: DraftState) -> str:
        """문장의 종류를 정한다.

        - LLM이 직접 뽑은 숫자·선택 값이 바뀐 값이면 LLM이 뭐라 했든 조건 입력(plan)이나 수정(revise)이다.
        - 이름만 있거나 온톨로지 사전이 채운 값뿐인데 의문 표현이 있으면 질문이다
          ("이 조건이면 PET 괜찮아?"의 'PET'는 기판을 바꾸려는 게 아니다).
        - 수정(revise)은 초안의 실제 값(기본값 포함)이 달라질 때다. 기본값이던 항목을 바꾸는 것도 수정이다.
        """
        given = {k for k in interp.values if interp.extracted.get(k) is not None}  # LLM이 직접 뽑은 값
        given |= {k for k in interp.values
                  if any(n.startswith(f"{self.kb.param[k].label}: ") and ("→" in n or "직접 읽음" in n) for n in interp.notes)}
        asks_text = bool(QUESTION_PATTERN.search(message))  # 문장에 의문 표현이 있다
        asks = asks_text or interp.intent == "question"
        # 바뀐 숫자·선택 값이 있으면 조건 입력이다. 지금 초안의 값을 그대로 되묻는 것("왜 1.333 kPa로 잡았어?")은 아니다
        strong = any(self.kb.param[k].kind != "text" and interp.values[k] != draft.values.get(k) for k in given)
        if strong and WHATIF_PATTERN.search(message):
            return "question"  # "200도까지 올려도 돼?" — 바꾸지 말고 가정해서 검사만
        if strong:
            plan = True
        elif given:
            # LLM이 직접 뽑은 이름만 있을 때는 문장에 의문 표현이 있어야 질문이다. LLM의 표시만으로는
            # "시편 번호 W-03 준비됐어" 같은 진술을 질문으로 오해해 값을 버리게 된다
            plan = not asks_text or interp.intent == "revise"
        else:
            plan = bool(interp.values) and not asks  # 온톨로지 사전이 채운 값뿐이면 의문 표현이 없을 때만
        if plan:
            return "revise" if draft.values and self._changed(draft.values, interp.values) else "plan"
        if asks:
            return "question"
        return "other"

    def _number_grounded(self, p, raw: Quantity, message_numbers: list[float]) -> bool:
        """숫자가 문장에 그대로 있거나, 문장의 숫자를 같은 물리량의 다른 단위로 환산한 값이면 근거가 있다.

        규칙상 LLM은 단위를 바꾸지 않아야 하지만, '1시간'을 '60분'으로, '2시간 30분'을 '150분'으로 합쳐 내는 일이
        있다. 값이 맞으면 받아 준다 (숫자 하나의 환산, 또는 숫자 두 개의 환산 합).
        """
        if any(n == raw.value for n in message_numbers):
            return True
        try:
            value, _ = self.kb.to_canonical(p, raw.value, raw.unit)
        except UnitError:
            return False
        units = self.kb.compatible_units(p)
        converted = [self.kb.to_canonical(p, n, unit)[0] for n in message_numbers for unit in units]
        if any(close(c, value) for c in converted):
            return True
        return any(close(a + b, value) for i, a in enumerate(converted) for b in converted[i + 1:])

    def _text_problem(self, p, text: str, message: str) -> str | None:
        """이름·메모 같은 문자열 값이 문장에 있고, 형식이 맞고, 재료 이름이나 일반 명사가 아닌지 확인한다."""
        if squash(text) not in squash(message):
            return f"{p.label} '{text}': 문장에 없는 내용이라 뺐습니다."
        if p.key == "notes" and squash(text) == squash(message):
            return f"{p.label}: 요청 문장 전체를 메모로 옮겨서 뺐습니다."
        if self.kb.is_generic_term(text):
            return f"{p.label} '{text}': 일반 명사라 뺐습니다."
        if p.value_pattern and not re.fullmatch(p.value_pattern, text):
            return f"{p.label} '{text}': 형식이 맞지 않아 뺐습니다."
        if p.key != "notes":
            # 이름 전체가 재료 이름일 때만 뺀다 ('PEDOT-07'처럼 재료 이름이 일부로 들어간 실험명은 둔다)
            for entity_type in MATERIAL_TYPES:
                linked = self.kb.link_name(text, entity_type)
                if linked and linked.surface == text.strip():
                    return f"{p.label} '{text}': {topic(linked.label)} {self.kb.label(entity_type)} 이름이라 뺐습니다."
        return None

    def _entity_problem(self, p, text: str, entity, message: str, message_entities: set) -> str | None:
        """용매·기판 이름이 문장에 근거가 있고 종류가 맞는지 온톨로지로 확인한다."""
        if self.kb.is_generic_term(text):
            return f"{p.label} '{text}': 재료 이름이 아닌 일반 명사라 뺐습니다."
        if entity is None:
            # 연결되지 않은 이름 안에 다른 종류의 재료가 있으면 칸을 잘못 고른 것이다
            # (예: 기판 칸의 '톨루엔 시편', 'PEDOT:PSS 박막')
            other = next((e for e in self.kb.link_entities(text)
                          if e.type in MATERIAL_TYPES and e.type != ENTITY_PARAMS[p.key]), None)
            if other:
                return f"{p.label} '{text}': {topic(other.label)} {copula(self.kb.label(other.type))} {p.label}에서 뺐습니다."
        # 적힌 그대로이거나, 같은 개체의 다른 이름이 문장에 있으면 근거가 있다 (예: 'PET 기판' → PET 필름)
        if squash(text) in squash(message) or (entity and entity.iri in message_entities):
            return None
        return f"{p.label} '{text}': 문장에 없는 이름이라 뺐습니다."

    def _fill_from_ontology(self, message: str, values: dict, notes: list[str]) -> None:
        """LLM이 놓친 것을 온톨로지 용어 사전으로 보완한다."""
        entities = self.kb.link_entities(message)
        for key, entity_type in ENTITY_PARAMS.items():
            found = [e for e in entities if e.type == entity_type]
            if key not in values and len(found) == 1:
                values[key] = found[0].label
                notes.append(f"{self.kb.param[key].label}: 문장의 '{found[0].surface}'를 {found[0].label}(으)로 인식 (온톨로지 사전)")
        g = self.kb.graph
        for entity in entities:
            if entity.type == VD.QualitativeSetpoint:
                param = next(p for p in self.kb.parameters if p.iri == g.value(entity.iri, VD.setpointFor))
                source = g.value(entity.iri, VD.resolvesTo)
                value = self.kb.component_value(source)
                if param.key not in values and value is not None:
                    values[param.key] = value
                    notes.append(
                        f"{param.label}: '{entity.surface}' → {value:g} {param.unit_symbol} "
                        f"(장비의 {self.kb.label(source)})"
                    )
            elif entity.type == VD.ProcessOrder and "processOrder" not in values:
                values["processOrder"] = str(g.value(entity.iri, VD.code))
                notes.append(f"공정 순서: '{entity.surface}' → {entity.label} (온톨로지 사전)")

    async def _extract(self, message: str, asked: list[str], demos: list[tuple[str, dict]] = ()) -> tuple[BaseModel, str]:
        """demos는 작업자가 확인한 (문장, JSON) 쌍. 규칙 속 '예시'가 아니라 이전 대화 턴으로 넣는다.

        시스템 프롬프트의 "예시 값을 옮기지 않는다" 규칙 때문에, 예시라고 이름 붙여 넣으면 작은 모델이
        예시와 같은 값(예: 90도)을 문장에서도 빼 버린다. 이전 턴으로 넣으면 그 혼동이 없다.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        for sentence, answer in demos:
            messages.append({"role": "user", "content": f"문장: {sentence}"})
            messages.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
        messages.append({"role": "user", "content": self._user_prompt(message, asked)})
        for _ in range(2):
            try:
                data, model = await self.llm.complete_json(messages, self.schema)
            except LLMFormatError as e:
                error, answer = e, e.raw
            else:
                try:
                    return self.extraction_model.model_validate(data), model
                except ValidationError as e:
                    error, answer = e, json.dumps(data, ensure_ascii=False)
            messages = messages + [
                {"role": "assistant", "content": answer or "(빈 응답)"},
                {"role": "user", "content": f"형식이 틀렸다 ({str(error)[:300]}). 스키마에 맞는 JSON만 다시 출력하라."},
            ]
        raise LLMError(f"로컬 LLM 응답이 형식에 맞지 않습니다. 다른 모델을 써 보세요. ({str(error)[:200]})")

    # ------------------------------------------------------------------ 대화

    async def chat(self, message: str, draft: DraftState | None = None) -> PlanResult:
        draft = draft or DraftState()
        interp = await self.interpret(message, draft)
        intent = self.resolve_intent(message, interp, draft)
        if intent == "question":
            return self._answer(message, draft, interp)
        values = draft.values | interp.values
        missing = [p for p in self.kb.parameters if p.required and p.key not in values]
        if intent == "other" and draft.values and not missing:
            return self._unchanged(draft, interp, message)
        notes = merge_notes(draft.notes, interp.notes)
        changed = self._changes(draft.values, interp.values) if intent == "revise" else ""
        if missing:
            return PlanResult(
                reply=changed + self._ask_reply(values, missing, interp.problems),
                draft=DraftState(values=values, asked=[p.key for p in missing], notes=notes),
                recipe=None,
                notes=interp.notes,
                issues=[],
                values=interp.values, extracted=interp.extracted,
                model=interp.model,
            )
        recipe, fill_notes = self._build_recipe(values)
        issues = self.kb.validate(recipe)
        return PlanResult(
            reply=changed + self._draft_reply(recipe, issues, interp.problems),
            draft=DraftState(values=values, notes=notes),
            recipe=recipe,
            notes=interp.notes + fill_notes,
            issues=issues,
            values=interp.values, extracted=interp.extracted,
            model=interp.model,
        )

    # ------------------------------------------------------------------ 질문에 근거로 답하기

    def _answer(self, message: str, draft: DraftState, interp: Interpretation) -> PlanResult:
        """질문에 답한다. 초안은 바꾸지 않는다 (recipe=None이면 화면도 검토 패널을 그대로 둔다).

        답의 재료는 모두 실행 중 생긴 정보와 온톨로지다: 가정 검사(SHACL), 재료 물성, 소요 시간, 해석 메모, 기본값 근거.
        """
        result = dict(draft=draft, recipe=None, notes=[], issues=[], values=interp.values, extracted=interp.extracted, model=interp.model)
        missing = [p for p in self.kb.parameters if p.required and p.key not in draft.values]
        complete = bool(draft.values) and not missing
        recipe, fill_notes = self._build_recipe(draft.values) if complete else (None, [])
        lines: list[str] = []

        # 1) "200도까지 올려도 돼?" — 가정한 값으로 규칙 검사만 한다. 지금 값을 되묻는 것("왜 1.333 kPa로?")은 제외
        effective = self._effective(draft.values)
        hypothetical = {k: v for k, v in interp.values.items()
                        if interp.extracted.get(k) is not None and self.kb.param[k].kind != "text"
                        and v != effective.get(k)}
        if hypothetical and complete:
            trial, _ = self._build_recipe(draft.values | hypothetical)
            issues = self.kb.validate(trial)
            result["issues"] = issues
            what = ", ".join(f"{self.kb.param[k].label} {v:g} {self.kb.param[k].unit_symbol}".strip()
                             if self.kb.param[k].kind == "number" else f"{self.kb.param[k].label} {v}"
                             for k, v in hypothetical.items())
            errors = [i for i in issues if i.level == "error"]
            if errors:
                lines.append(f"{what}(으)로 하면 실행할 수 없어요: " + " / ".join(i.message for i in errors))
            else:
                lines.append(f"{what}(으)로 해도 규칙상 괜찮아요." + (
                    " 다만: " + " / ".join(f"[{LEVEL_LABEL[i.level]}] {i.message}" for i in issues) if issues else ""))
                lines.append(f"바꾸려면 '{what}로'라고 말씀해 주세요. 지금 초안은 그대로예요.")
            return PlanResult(reply="\n".join(lines), **result)

        # 2) 재료 물성 질문 — 초안이 없어도 답한다
        pressure = draft.values.get("targetPressure")
        materials = [e for e in self.kb.link_entities(message) if e.type in MATERIAL_TYPES]
        for entity in materials:
            lines += [f"- {fact}" for fact in self.kb.entity_facts(entity, pressure)]
        solvents = [e for e in materials if e.type == VD.Solvent]
        if len(solvents) >= 2:
            g = self.kb.graph
            lowest = min(solvents, key=lambda e: float(g.value(e.iri, VD.boilingPoint)))
            lines.append(f"끓는점이 낮은 {lowest.label} 쪽이 같은 조건에서 더 빨리 증발하는 경향이 있어요. "
                         "실제 건조 시간은 포토다이오드가 판정해요.")
        if not draft.values:
            if not lines:
                lines.append("아직 초안이 없어요. 먼저 공정 조건을 말씀해 주세요 (예: IPA 시편 5 kPa 80도).")
            return PlanResult(reply="\n".join(lines), **result)

        # 3) 소요 시간, 다음 단계
        if complete and DURATION_PATTERN.search(message):
            ramp = max(recipe.heating.target_c - AMBIENT_C, 0) / recipe.heating.ramp_c_per_min
            lines.append(f"문을 닫은 뒤 건조 종료까지 최대 {active_minutes_upper_bound(recipe):g}분이에요 "
                         f"(배기 최대 {recipe.vacuum.pump_timeout_min:g}분 + 승온 약 {ramp:.0f}분 + 유지 최대 "
                         f"{recipe.endpoint.max_hold_min:g}분). 건조 완료가 일찍 감지되면 더 짧고, 냉각·벤트는 따로 걸려요.")
        if complete and NEXT_PATTERN.search(message):
            lines.append("검토 화면에서 조건을 확인하고 승인자 이름을 적은 뒤 '승인하고 공정 시작'을 누르세요. "
                         "그러면 문 닫기 → 진공 배기 → 승온 → 건조 → 냉각 → 벤트 → 문 열기 순서로 진행돼요.")
        if lines and not DRAFT_FOCUS.search(message):
            return PlanResult(reply="\n".join(lines), **result)
        context = lines  # 재료 물성 등은 근거 설명 앞에 붙인다

        # 4) 초안의 근거
        targets = self._question_targets(message)
        about_defaults = any(w in message for w in ("기본값", "디폴트", "default"))
        params = targets or list(self.kb.parameters)
        if about_defaults:
            params = [p for p in params if p.key not in draft.values]
        if targets:
            lines = [f"{'·'.join(p.label for p in targets)}의 근거예요."]
        elif about_defaults:
            lines = ["기본값으로 채운 항목이에요. 모두 말씀하시면 바뀌어요."]
        else:
            lines = ["초안의 근거예요."]
        if missing:
            lines[0] = "아직 초안이 완성되지 않았어요. 지금까지의 근거예요."
        lines = context + lines
        for p in params:
            if line := self._explain_parameter(p, draft, fill_notes, recipe):
                lines.append(f"- {line}")
        if missing:
            lines.append("아직 없는 값: " + ", ".join(f"{p.label}({p.unit_symbol})" for p in missing))
        elif (not targets and not about_defaults) or any(w in message for w in ("규칙", "한계", "안전", "괜찮", "맞")):
            issues = self.kb.validate(recipe)
            result["issues"] = issues
            if issues:
                lines.append("규칙 검사: " + " / ".join(f"[{LEVEL_LABEL[i.level]}] {i.message}" for i in issues))
            else:
                lines.append("규칙 검사: 장비·재료 한계를 모두 통과했어요.")
            lines += [f"- {fact}" for fact in self.kb.facts(recipe)]
        return PlanResult(reply="\n".join(lines), **result)

    def _question_targets(self, message: str) -> list:
        """문장이 특정 항목을 가리키면 그 항목들만.

        이름이나 동의어가 그대로 있으면 그 항목이다 ('승온 속도는 어떻게 정했어?'). 이름이 두 단어 이상이면
        그중 두 단어가 보여도 그 항목으로 본다 ('최대 온도 시간이 왜 사라졌어?' → 최대 유지 시간).
        한 항목의 표현이 다른 항목의 표현 안에 들어 있으면('냉각 온도' 안의 '온도') 긴 쪽만 센다.
        """
        text = squash(message)
        spans = {}
        for p in self.kb.parameters:
            names = sorted((squash(n) for n in (p.label, *p.synonyms) if squash(n) in text), key=len, reverse=True)
            if names:
                start = text.find(names[0])
                spans[p.key] = (start, start + len(names[0]))
                continue
            words = [w for w in p.label.split() if w in text]
            if len(p.label.split()) >= 2 and len(words) >= 2:
                positions = [text.find(w) for w in words]
                spans[p.key] = (min(positions), max(pos + len(w) for pos, w in zip(positions, words)))
        keys = [k for k, (a, b) in spans.items()
                if not any(k != o and (oa <= a and b <= ob) and (ob - oa) > (b - a) for o, (oa, ob) in spans.items())]
        return [p for p in self.kb.parameters if p.key in keys]

    def _explain_parameter(self, p, draft: DraftState, fill_notes: list[str], recipe: Recipe | None) -> str | None:
        prefix = f"{p.label}: "
        default = default_for(p.json_field) if p.kind == "number" else None
        if p.key in draft.values:
            value = draft.values[p.key]
            shown = f"{value:g} {p.unit_symbol}".strip() if p.kind == "number" else dict(p.choices).get(value, value)
            note = next((n[len(prefix):] for n in draft.notes if n.startswith(prefix)), None)
            if note:
                reason = note
            elif default is not None and default != value:
                reason = f"말씀하신 값이에요 (기본값 {default:g} {p.unit_symbol} 대신)."
            else:
                reason = "말씀하신 값 그대로예요."
        elif (note := next((n for n in fill_notes if n.startswith(prefix)), None)) is not None:
            # 채운 값 메모는 "목표 압력: 5 kPa (기본값)", "실험 이름: '…' (자동 생성)" 꼴이다
            shown, _, how = note[len(prefix):].rpartition(" (")
            if how.startswith("자동"):
                reason = "말씀이 없어 자동으로 지었어요."
            else:
                reason = "말씀이 없어 기본값을 썼어요." + (f" {p.default_rationale}" if p.default_rationale else "")
        else:
            return None
        if limit := self._limit_hint(p):
            reason += f" {limit}"
        return f"{p.label} {shown}: {reason}"

    def _limit_hint(self, p) -> str:
        limits = self.kb.limits()
        hints = {
            "targetTemperature": f"장비 최대 온도는 {limits.max_temp_c:g} °C예요.",
            "rampRate": f"장비 최대 승온 속도는 {limits.max_ramp_c_per_min:g} °C/min이에요.",
            "targetPressure": f"펌프 최저 도달 압력은 {limits.min_pressure_kpa:g} kPa예요.",
            "maxHoldDuration": f"가열 구간은 {limits.max_process_min:g}분을 넘을 수 없어요.",
        }
        return hints.get(p.key, "")

    def _unchanged(self, draft: DraftState, interp: Interpretation, message: str = "") -> PlanResult:
        if START_PATTERN.search(message):
            reply = ("채팅으로는 공정을 시작하지 않아요. 검토 화면에서 조건을 확인하고 승인자 이름을 적은 뒤 "
                     "'승인하고 공정 시작' 버튼을 눌러 주세요. 초안은 그대로예요.")
        elif RESET_PATTERN.search(message):
            reply = "처음부터 하려면 '새 대화' 버튼을 누르세요. 초안이 지워져요. 지금 초안은 그대로 두었어요."
        else:
            reply = (
                "이번 말씀에서는 공정 조건을 찾지 못했어요. 초안은 그대로예요: " + self._describe(draft.values) + ".\n"
                "값을 바꾸려면 '온도 90도로'처럼, 이유가 궁금하면 '왜 …?'처럼 말씀해 주세요."
            )
        if interp.problems:
            reply = "확인이 필요한 내용: " + " / ".join(interp.problems) + "\n" + reply
        return PlanResult(reply=reply, draft=draft, recipe=None, notes=[], issues=[], values=interp.values, extracted=interp.extracted, model=interp.model)

    def _drop_duplicate_numbers(self, message: str, values: dict, extracted: dict, problems: list[str]) -> None:
        """문장에 한 번 나온 숫자를 두 항목에 넣었으면("판정 구간 1분" → 판정 구간과 최소 유지 시간) 하나만 남긴다.

        숫자 바로 앞에 이름(동의어)이 붙은 항목을 남기고, 없으면 먼저 정의된 항목을 남긴다.
        """
        groups: dict[str, list[str]] = {}
        for key in values:
            raw = extracted.get(key)
            if self.kb.param[key].kind == "number" and isinstance(raw, dict):
                groups.setdefault(f"{raw['value']:g}", []).append(key)
        message_numbers = numbers_in(message)
        for number, keys in groups.items():
            occurrences = sum(1 for n in message_numbers if n == float(number))
            if len(keys) < 2 or len(keys) <= occurrences:
                continue  # 진짜 중복(두 항목 이상)일 때만. 문장에 없는 숫자는 근거 검사가 따로 본다
            before = [m.start() for m in re.finditer(rf"(?<![\d.]){re.escape(number)}(?![\d.])", message)]
            def cued(key: str) -> bool:
                p = self.kb.param[key]
                return any(squash(n) in squash(message[max(0, pos - 14):pos]) for pos in before for n in (p.label, *p.synonyms))
            keep = [k for k in keys if cued(k)] or keys[:1]
            for key in keys:
                if key not in keep[:occurrences]:
                    problems.append(f"{self.kb.param[key].label} {number}: 같은 숫자를 다른 항목에도 넣어 뺐습니다.")
                    values.pop(key)

    def _lost_required_value(self, message: str, draft: DraftState, interp: Interpretation) -> bool:
        """필수 항목이 비었는데 문장에 숫자+단위가 있으면, 예시 때문에 값을 놓쳤을 가능성이 크다."""
        known = draft.values | interp.values
        missing = [p for p in self.kb.parameters if p.required and p.key not in known]
        return bool(missing) and len(NUMBER_WITH_UNIT.findall(message)) >= 2

    def _backup_extract(self, message: str, values: dict, notes: list[str]) -> None:
        """LLM이 숫자를 하나도 못 뽑았고, 문장이 항목 하나를 가리키며 숫자+단위가 하나뿐이면 코드가 직접 읽는다.

        작은 모델은 "압력은 0.5 kPa 낮춰" 같은 수정 지시에서 값을 비우는 일이 있다. 단위가 온톨로지에서
        그 항목과 같은 물리량일 때만 받는다.
        """
        targets = [p for p in self._question_targets(message) if p.kind == "number"]
        found = NUMBER_WITH_UNIT.findall(message)
        if len(targets) != 1 or len(found) != 1:
            return
        p, (number, unit) = targets[0], found[0]
        try:
            value, _ = self.kb.to_canonical(p, float(number), unit)
        except UnitError:
            return
        values[p.key] = value
        notes.append(f"{p.label}: {number} {unit} (문장에서 직접 읽음)")

    def _apply_relative(self, message: str, values: dict, notes: list[str], extracted: dict, draft: DraftState) -> None:
        """'온도 10도 올려줘'처럼 상대 변경이면 초안 값에 더하거나 뺀다. 숫자 뒤에 로/으로/까지가 있으면 절대값이다.

        '압력 절반으로', '두 배로'처럼 숫자 없는 배수 표현은 문장이 가리키는 항목 하나에 적용한다.
        """
        effective = self._effective(draft.values)
        up, down = RELATIVE_UP.search(message), RELATIVE_DOWN.search(message)
        numbers = re.findall(r"\d+(?:\.\d+)?", message)
        # 초안이 없으면 더할 기준이 없고, "분당 3도씩 올려서 110도"처럼 숫자가 여럿이거나 동사가 숫자 앞에 오면 계획 문장이다
        verb = up or down
        relative = draft.values and bool(up) != bool(down) and len(numbers) == 1 and verb.start() > message.find(numbers[0])
        if relative:
            for key, value in list(values.items()):
                p = self.kb.param[key]
                raw = extracted.get(key)
                direct = next((n for n in notes if n.startswith(f"{p.label}: ") and "직접 읽음" in n), None)
                if p.kind != "number" or key not in effective or not (isinstance(raw, dict) or direct):
                    continue
                number = f"{raw['value']:g}" if isinstance(raw, dict) else direct.split(": ", 1)[1].split()[0]
                m = re.search(rf"(?<![\d.]){re.escape(number)}(?![\d.])\s*(\S{{0,8}})", message)
                if m and re.search(r"(으로|로|까지)[^가-힣]*$", m.group(1)):
                    continue  # "100도로", "120도까지" → 절대값
                delta = value if up else -value
                new = round(effective[key] + delta, 4)
                values[key] = new
                notes.append(f"{p.label}: {effective[key]:g} → {new:g} {p.unit_symbol} ({delta:+g} {p.unit_symbol})")
        factor = next((f for pattern, f in MULTIPLIERS if re.search(pattern, message)), None)
        if factor is not None and not any(self.kb.param[k].kind == "number" for k in values):
            targets = [p for p in self._question_targets(message) if p.kind == "number" and p.key in effective]
            if len(targets) == 1:
                p = targets[0]
                new = round(effective[p.key] * factor, 4)
                values[p.key] = new
                notes.append(f"{p.label}: {effective[p.key]:g} → {new:g} {p.unit_symbol} (×{factor:g})")

    def _effective(self, values: dict) -> dict:
        """초안의 실제 값: 말한 값 + 말하지 않은 숫자 항목의 기본값."""
        effective = dict(values)
        for p in self.kb.parameters:
            if p.key not in effective and p.kind == "number" and (default := default_for(p.json_field)) is not None:
                effective[p.key] = default
        return effective

    def _changed(self, old: dict, new: dict) -> list:
        """새 값 중 초안의 실제 값(기본값 포함)과 다른 항목."""
        effective = self._effective(old)
        return [p for p in self.kb.parameters if p.key in new and p.key in effective and effective[p.key] != new[p.key]]

    def _changes(self, old: dict, new: dict) -> str:
        effective = self._effective(old)
        parts = []
        for p in self._changed(old, new):
            before, after = effective[p.key], new[p.key]
            if p.kind == "number":
                parts.append(f"{p.label} {before:g} → {after:g} {p.unit_symbol}")
            elif p.kind == "choice":
                names = dict(p.choices)
                parts.append(f"{p.label} {names.get(before, before)} → {names.get(after, after)}")
            else:
                parts.append(f"{p.label} {before} → {after}")
        return f"바꿨어요: {', '.join(parts)}.\n" if parts else ""

    def _build_recipe(self, values: dict) -> tuple[Recipe, list[str]]:
        notes = []
        values = dict(values)
        if not values.get("recipeName"):
            parts = [values.get("solventName"), f"{values['targetTemperature']:g}°C", "진공 건조"]
            values["recipeName"] = " ".join(p for p in parts if p)
            notes.append(f"실험 이름: '{values['recipeName']}' (자동 생성)")
        data: dict = {}
        for p in self.kb.parameters:
            if p.key in values:
                node = data
                *parents, leaf = p.json_field.split(".")
                for name in parents:
                    node = node.setdefault(name, {})
                node[leaf] = values[p.key]
            elif p.kind == "number" and (default := default_for(p.json_field)) is not None:
                notes.append(f"{p.label}: {default:g} {p.unit_symbol} (기본값)")
        return Recipe.model_validate(data), notes

    def _describe(self, values: dict) -> str:
        parts = []
        for p in self.kb.parameters:
            if p.key not in values or p.key == "notes":
                continue
            value = values[p.key]
            if p.kind == "number":
                parts.append(f"{p.label} {value:g} {p.unit_symbol}")
            elif p.kind == "choice":
                parts.append(f"{p.label} {dict(p.choices).get(value, value)}")
            else:
                parts.append(f"{p.label} {value}")
        return ", ".join(parts)

    def _ask_reply(self, values: dict, missing: list, problems: list[str]) -> str:
        lines = []
        if problems:
            lines.append("확인이 필요한 내용: " + " / ".join(problems))
        understood = self._describe(values)
        lines.append(f"지금까지 이해한 조건: {understood}." if understood else "계획에서 공정 조건을 찾지 못했습니다.")
        needed = ", ".join(f"{p.label}({p.unit_symbol})" for p in missing)
        lines.append(f"레시피를 만들려면 이 값이 필요해요: {needed}.")
        return "\n".join(lines)

    def _draft_reply(self, r: Recipe, issues: list[Issue], problems: list[str]) -> str:
        lines = []
        if problems:
            lines.append("확인이 필요한 내용: " + " / ".join(problems))
        lines.append(
            f"레시피 초안을 만들었습니다: 목표 압력 {r.vacuum.target_kpa:g} kPa, "
            f"목표 온도 {r.heating.target_c:g} °C (승온 {r.heating.ramp_c_per_min:g} °C/분), "
            f"포토다이오드로 건조 완료 판정 (최대 {r.endpoint.max_hold_min:g}분)."
        )
        errors = [i for i in issues if i.level == "error"]
        if errors:
            lines.append(f"실행할 수 없는 항목이 {len(errors)}개 있습니다: {errors[0].message} 값을 바꿔 말씀해 주세요.")
        else:
            lines.append("검토 화면에서 확인한 뒤 승인해 주세요.")
        return "\n".join(lines)
