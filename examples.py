"""작업자가 승인한 레시피를 학습 데이터로 쌓고, 비슷한 과거 문장을 프롬프트 예시로 되먹인다.

모델 가중치는 그대로다. 쌓이는 것은 (작업자가 말한 문장 → 작업자가 검토·수정해 승인한 값) 쌍이고,
이것을 두 곳에 쓴다.
- 프롬프트 예시: 새 문장이 오면 비슷한 과거 문장과 그 정답을 찾아 LLM에게 보여 준다 (인컨텍스트 러닝).
  가중치를 바꾸지 않으므로 즉시 효과가 나고, 파일을 지우면 그대로 되돌아간다.
- 평가 사례: `to_cases()`로 cases.jsonl 형식으로 내보내 한 번도 보지 않은 시험 세트를 만든다.
  데이터가 수백 개 쌓이면 LoRA 파인튜닝 자료로도 쓸 수 있다.

유사도는 외부 임베딩 없이 글자 2-gram Jaccard에 온톨로지 개체(용매·기판) 일치 보너스를 더한다.
8 GB 노트북에서 임베딩 모델까지 올리지 않기 위해서다.
"""
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from .knowledge import MATERIAL_TYPES, KnowledgeBase, _flatten
from .recipe import Recipe


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def bigrams(text: str) -> set[str]:
    t = squash(text)
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) > 1 else {t}


def recipe_values(kb: KnowledgeBase, recipe: Recipe) -> dict:
    """레시피를 파라미터 키 → 값(기준 단위) 사전으로 바꾼다. 빈 문자열은 뺀다."""
    flat = _flatten(recipe.model_dump())
    return {p.key: flat[p.json_field] for p in kb.parameters if flat.get(p.json_field) not in (None, "")}


class ExampleStore:
    def __init__(self, path: Path, kb: KnowledgeBase):
        self.path = Path(path)
        self.kb = kb

    # ------------------------------------------------------------------ 저장

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def record(self, turns: list[dict], ai_values: dict, final_values: dict, approved_by: str, run_id: str) -> dict:
        """승인 한 번을 기록한다. turns는 채팅 턴마다 {message, extracted(LLM 원출력), values(해석한 값)}."""
        keys = {p.key for p in self.kb.parameters}
        changed = {k: v for k, v in final_values.items() if k in keys and ai_values.get(k) != v}
        removed = [k for k in ai_values if k in keys and k not in final_values]
        example = {
            "id": uuid.uuid4().hex[:8],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "messages": [t["message"] for t in turns],
            "turns": turns,
            "ai_values": ai_values,
            "final_values": final_values,
            "changed": changed,
            "removed": removed,
            "corrected": bool(changed or removed),
            "approved_by": approved_by,
            "run_id": run_id,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
        return example

    def stats(self) -> dict:
        examples = self.load()
        return {
            "count": len(examples),
            "single_turn": sum(len(e["turns"]) == 1 for e in examples),
            "corrected": sum(e["corrected"] for e in examples),
        }

    # ------------------------------------------------------------------ 되먹이기

    def similar(self, message: str, k: int = 2, min_score: float = 0.2) -> list[dict]:
        """새 문장과 비슷한 과거 문장을 찾는다. 한 턴짜리 기록만 예시로 쓴다 (여러 턴은 문장 하나로 값이 안 맞는다)."""
        if k <= 0:
            return []
        query, query_entities = bigrams(message), {e.iri for e in self.kb.link_entities(message)}
        scored = []
        for example in self.load():
            if len(example["turns"]) != 1:
                continue
            sentence = example["messages"][0]
            grams = bigrams(sentence)
            jaccard = len(query & grams) / len(query | grams) if query | grams else 0.0
            shared = query_entities & {e.iri for e in self.kb.link_entities(sentence)}
            score = jaccard + 0.1 * len(shared)
            if score >= min_score:
                scored.append((score, example))
        scored.sort(key=lambda s: -s[0])
        return [example for _, example in scored[:k]]

    def example_json(self, example: dict) -> dict:
        """예시로 보여 줄 JSON: LLM이 그 문장에서 뽑은 원출력을 작업자가 고친 값으로 바로잡은 것.

        LLM이 뽑지 않은 항목(작업자가 양식에서 더한 값)은 넣지 않는다. 문장에 없는 값을 지어내라고
        가르치는 셈이 되기 때문이다. 작업자가 지운 항목은 뺀다.
        """
        turn = example["turns"][0]
        raw = {k: v for k, v in (turn.get("extracted") or {}).items() if v is not None and k != "intent"}
        for key in list(raw):
            p = self.kb.param.get(key)
            if p is None:
                raw.pop(key)
                continue
            final = example["final_values"].get(key)
            interpreted = (turn.get("values") or {}).get(key)
            if final is None:
                raw.pop(key)
            elif interpreted != final:
                raw[key] = {"value": final, "unit": p.unit_symbol} if p.kind == "number" else final
        raw["intent"] = "plan"
        return raw

    def prompt_block(self, examples: list[dict]) -> str:
        if not examples:
            return ""
        lines = ["참고 예시 (작업자가 확인한 과거 문장. 항목 고르는 방식만 참고하고, 값은 아래 문장에서 읽는다)"]
        for example in examples:
            lines.append(f"문장: {example['messages'][0]}")
            lines.append(f"JSON: {json.dumps(self.example_json(example), ensure_ascii=False)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ 평가 사례로 내보내기

    def to_cases(self) -> list[dict]:
        """한 턴짜리 기록을 cases.jsonl 형식으로. 정답은 LLM이 뽑았던 항목에 대한 작업자 확인값이다."""
        cases = []
        for example in self.load():
            if len(example["turns"]) != 1:
                continue
            keys = [k for k in self.example_json(example) if k != "intent"]
            expect = {k: example["final_values"][k] for k in keys if k in example["final_values"]}
            cases.append({"id": f"op-{example['id']}", "category": "operator", "message": example["messages"][0],
                          "expect": expect, "intent": "plan"})
        return cases


def main(argv: list[str]) -> int:
    """python3 -m process_ai.examples stats | export-cases [출력 파일]"""
    from . import config

    store = ExampleStore(config.EXAMPLES_PATH, KnowledgeBase())
    command = argv[0] if argv else "stats"
    if command == "stats":
        print(json.dumps(store.stats(), ensure_ascii=False))
    elif command == "export-cases":
        out = Path(argv[1]) if len(argv) > 1 else Path(__file__).parent / "evaluation" / "cases_from_operators.jsonl"
        cases = store.to_cases()
        out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
        print(f"{len(cases)}개 사례 → {out}")
    else:
        print(main.__doc__)
        return 2
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
