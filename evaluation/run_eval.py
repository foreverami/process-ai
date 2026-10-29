"""로컬 LLM 모델 비교 평가.

같은 실험 계획 문장 모음(cases.jsonl)을 여러 모델에 넣고, 온톨로지 정규화까지 거친 결과를
정답과 비교한다. 모델을 느낌이 아니라 숫자로 고르기 위한 도구다.

    python3 -m process_ai.evaluation.run_eval                          # 서버에 올라온 모델
    python3 -m process_ai.evaluation.run_eval --model qwen3-4b --model gemma-3-4b-it
    python3 start.py eval --url rules                                  # 규칙 기반 추출기 (LLM 없는 비교 기준)

지표
- 필드 정확도: 정답 항목 중 맞게 뽑은 비율 (단위 변환 후 1% 이내면 정답)
- 환각: 문장에 없는데 채운 항목 수 (안전과 직결되므로 0이어야 한다)
- 온톨로지 보완: LLM이 놓친 것을 온톨로지 사전이 채운 횟수
- 보완 제외 정확도: 같은 LLM 출력에서 온톨로지 사전 보완 단계만 빼고 다시 채점한 필드 정확도.
  필드 정확도와의 차이가 온톨로지 보완의 기여분이다 (LLM을 다시 부르지 않는다)
- 검사 제외 환각: 같은 LLM 출력에서 근거·종류 검사만 빼고 센 환각 수. 환각과의 차이가 검사가 막은 수다
- 유형별 정확도: cases.jsonl의 category(단위 변환, 용어 연결, 후속 답변 등)별 필드 정확도
- 의도: 사례에 "intent"(plan/revise/question/other)가 있으면 Planner.resolve_intent 결과를 항목 하나로 채점한다
"""
import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from ..config import LLM_BASE_URL, LLM_REASONING_EFFORT, LLM_TIMEOUT_S
from ..knowledge import KnowledgeBase
from ..llm import LLMError
from ..planner import DraftState, Interpretation, Planner
from ..rules import make_extractor

CASES_PATH = Path(__file__).with_name("cases.jsonl")
RESULTS_DIR = Path(__file__).with_name("results")
ONTOLOGY_MARKERS = ("온톨로지 사전", "(장비의")
CATEGORY_LABELS = {
    "basic": "기본", "unit": "단위 변환", "synonym": "용어 연결", "qualitative": "정성 표현·공정 순서",
    "multi": "여러 값 구분", "followup": "후속 답변", "english": "영어·혼용", "informal": "구어체·띄어쓰기",
    "unknown": "미등록 용어", "irrelevant": "조건 없음", "question": "질문·수정 의도", "revise": "값 수정",
    "ambiguous": "모호한 계획",
}


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def matches(expected, actual) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and abs(actual - expected) <= max(abs(expected) * 0.01, 1e-6)
    return str(actual).strip().lower() == str(expected).strip().lower()


def score_case(case: dict, values: dict) -> dict:
    expect = case.get("expect", {})
    allowed = set(expect) | set(case.get("allow", []))
    correct = [k for k, v in expect.items() if k in values and matches(v, values[k])]
    return {
        "correct": correct,
        "wrong": [k for k in expect if k not in correct],
        "hallucinated": [k for k in values if k not in allowed],
    }


def passed(score: dict) -> bool:
    return not score["wrong"] and not score["hallucinated"]


def accuracy(rows: list[dict], key: str = "correct") -> float | None:
    total = sum(r["expected"] for r in rows)
    return round(sum(len(r[key]) for r in rows) / total, 3) if total else None


def row_base(case: dict) -> dict:
    expected = len(case.get("expect", {})) + ("intent" in case)
    return {"id": case["id"], "category": case.get("category", "basic"), "expected": expected}


def error_row(case: dict, error: str, latency_s: float) -> dict:
    return row_base(case) | {
        "error": error, "correct": [], "wrong": list(case.get("expect", {})) + (["intent"] if "intent" in case else []),
        "hallucinated": [], "passed": False,
        "correct_no_fill": [], "passed_no_fill": False, "hallucinated_no_guard": [], "latency_s": latency_s,
    }


def score_row(planner: Planner, case: dict, interp: Interpretation) -> dict:
    draft = DraftState(**case.get("draft", {}))
    # 채팅과 같은 판단을 거친다: 질문·잡담으로 판단되면 뽑힌 값은 버려지므로 채점에도 넣지 않는다
    intent = planner.resolve_intent(case["message"], interp, draft)
    used = interp.values if intent in ("plan", "revise") else {}

    def score_with_intent(values: dict) -> dict:
        score = score_case(case, values)
        if "intent" in case:
            score["correct" if intent == case["intent"] else "wrong"].append("intent")
        return score

    score = score_with_intent(used) | {"intent": intent}
    # 같은 LLM 출력으로 온톨로지 사전 보완만, 근거·종류 검사만 각각 빼고 다시 채점한다
    extraction = planner.extraction_model.model_validate(interp.extracted)
    no_fill = score_with_intent(planner.normalize(case["message"], extraction, interp.model, ontology_fill=False, draft=draft).values if used else {})
    no_guard = score_case(case, planner.normalize(case["message"], extraction, interp.model, guard=False, draft=draft).values if used else {})
    return row_base(case) | {
        **score,
        "passed": passed(score),
        "correct_no_fill": no_fill["correct"],
        "passed_no_fill": passed(no_fill),
        "hallucinated_no_guard": no_guard["hallucinated"],
        "problems": interp.problems,
        "values": interp.values,
        "extracted": interp.extracted,
        "ontology_assists": sum(any(m in note for m in ONTOLOGY_MARKERS) for note in interp.notes),
        "latency_s": interp.latency_s,
        "model": interp.model,
    }


async def evaluate(planner: Planner, cases: list[dict], partial: Path | None = None) -> dict:
    """사례를 하나씩 돌린다. partial을 주면 사례마다 결과를 한 줄씩 저장하고, 이미 저장된 사례는 건너뛴다.

    8 GB 노트북에서는 평가 도중 메모리가 부족해 중단하는 일이 있어, 중단해도 이어서 돌릴 수 있게 한다.
    """
    done: dict[str, dict] = {}
    if partial and partial.exists():
        done = {row["id"]: row for row in map(json.loads, partial.read_text(encoding="utf-8").splitlines()) if row}
    rows = []
    for case in cases:
        if case["id"] in done:
            rows.append(done[case["id"]])
            continue
        started = time.perf_counter()
        try:
            interp = await planner.interpret(case["message"], DraftState(**case.get("draft", {})))
        except LLMError as e:
            row = error_row(case, str(e), round(time.perf_counter() - started, 2))
        else:
            row = score_row(planner, case, interp)
        rows.append(row)
        if partial:
            with partial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summarize(rows)


def rescore(planner: Planner, cases: list[dict], saved: dict) -> dict:
    """저장된 결과의 LLM 원출력을 지금 코드의 후처리(정규화·검사·보완)로 다시 채점한다. LLM은 부르지 않는다.

    프롬프트를 바꾸지 않고 후처리만 고쳤을 때 쓴다. 응답 시간은 저장된 값을 그대로 쓴다.
    """
    by_id = {c["id"]: c for c in cases}
    rows = []
    for old in saved["rows"]:
        case = by_id.get(old["id"])
        if case is None:
            continue
        if "error" in old:
            rows.append(error_row(case, old["error"], old["latency_s"]))
            continue
        extraction = planner.extraction_model.model_validate(old["extracted"])
        interp = planner.normalize(case["message"], extraction, old["model"], draft=DraftState(**case.get("draft", {})))
        interp.latency_s = old["latency_s"]
        rows.append(score_row(planner, case, interp))
    return summarize(rows) | {"rescored_from": saved.get("tag", "")}


def summarize(rows: list[dict]) -> dict:
    by_category = {}
    for category in dict.fromkeys(r["category"] for r in rows):
        group = [r for r in rows if r["category"] == category]
        by_category[category] = {
            "cases": len(group),
            "passed": sum(r["passed"] for r in group),
            "field_accuracy": accuracy(group),
            "field_accuracy_no_fill": accuracy(group, "correct_no_fill"),
        }
    return {
        "model": next((r["model"] for r in rows if "model" in r), None),
        "cases": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "field_accuracy": accuracy(rows),
        "passed_no_fill": sum(r["passed_no_fill"] for r in rows),
        "field_accuracy_no_fill": accuracy(rows, "correct_no_fill"),
        "hallucinations": sum(len(r["hallucinated"]) for r in rows),
        "hallucinations_no_guard": sum(len(r["hallucinated_no_guard"]) for r in rows),
        "errors": sum("error" in r for r in rows),
        "ontology_assists": sum(r.get("ontology_assists", 0) for r in rows),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2) if rows else None,
        "by_category": by_category,
        "rows": rows,
    }


def percent(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "-"


def format_report(reports: list[dict]) -> str:
    lines = [
        "| 모델 | 통과 사례 | 필드 정확도 | 환각 | 오류 | 온톨로지 보완 | 평균 응답(초) | 보완 제외 정확도 | 검사 제외 환각 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r['model']} | {r['passed']}/{r['cases']} | {percent(r['field_accuracy'])} | {r['hallucinations']} | "
            f"{r['errors']} | {r['ontology_assists']} | {r['avg_latency_s']} | {percent(r['field_accuracy_no_fill'])} | "
            f"{r['hallucinations_no_guard']} |"
        )
    categories = list(dict.fromkeys(c for r in reports for c in r["by_category"]))
    lines += [
        "\n유형별 필드 정확도 (괄호: 온톨로지 보완 제외)",
        "| 유형 | 사례 수 | " + " | ".join(r["model"] for r in reports) + " |",
        "|---|---|" + "---|" * len(reports),
    ]
    for category in categories:
        cells = []
        for r in reports:
            c = r["by_category"].get(category)
            cells.append(f"{percent(c['field_accuracy'])} ({percent(c['field_accuracy_no_fill'])})" if c else "-")
        count = next(r["by_category"][category]["cases"] for r in reports if category in r["by_category"])
        lines.append(f"| {CATEGORY_LABELS.get(category, category)} | {count} | " + " | ".join(cells) + " |")
    for r in reports:
        failed = [row for row in r["rows"] if not row["passed"]]
        if failed:
            lines.append(f"\n{r['model']} 실패 사례")
            for row in failed:
                detail = row.get("error") or (
                    f"틀림 {row['wrong']} 환각 {row['hallucinated']} → {json.dumps(row.get('values'), ensure_ascii=False)}"
                )
                lines.append(f"- {row['id']}: {detail}")
    return "\n".join(lines)


async def run(models: list[str], url: str, timeout: float, cases_path: Path, tag: str = "", resume: bool = False) -> list[dict]:
    kb = KnowledgeBase()
    cases = load_cases(cases_path)
    reports = []
    for model in models or [""]:
        partial = RESULTS_DIR / f"partial-{tag or 'run'}-{(model or 'default').replace('/', '_')}.jsonl" if resume else None
        llm = make_extractor(kb, url, model, timeout, LLM_REASONING_EFFORT)  # --url rules 이면 규칙 기반 추출기
        try:
            status = await llm.status()
            if not status["ok"]:
                raise SystemExit(f"로컬 LLM을 쓸 수 없습니다: {status['error']}")
            planner = Planner(kb, llm)
            # 첫 요청의 모델 적재 시간이 응답 시간에 섞이지 않도록 한 번 미리 부른다
            try:
                await planner.interpret("IPA, 5 kPa, 80도")
            except LLMError:
                pass
            if partial:
                RESULTS_DIR.mkdir(exist_ok=True)
            report = await evaluate(planner, cases, partial)
        finally:
            await llm.aclose()
        report["model"] = report["model"] or model or "(서버 기본 모델)"
        reports.append(report)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 LLM 모델 비교 평가")
    parser.add_argument("--model", action="append", default=[], help="평가할 모델 이름 (여러 번 지정 가능)")
    parser.add_argument("--url", default=LLM_BASE_URL, help="OpenAI 호환 API 주소. 'rules'면 LLM 없이 규칙 기반 추출기")
    parser.add_argument("--timeout", type=float, default=LLM_TIMEOUT_S)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--tag", default="", help="결과 파일 이름에 붙일 실험 이름 (예: v1-baseline)")
    parser.add_argument("--rescore", type=Path, help="저장된 결과 파일의 LLM 출력을 지금 코드로 다시 채점 (LLM 호출 없음)")
    parser.add_argument("--resume", action="store_true",
                        help="사례마다 results/partial-<tag>-<model>.jsonl에 저장하고, 중단 후 다시 실행하면 이어서 돌린다")
    args = parser.parse_args()

    if args.rescore:
        planner = Planner(KnowledgeBase(), llm=None)
        saved = json.loads(args.rescore.read_text(encoding="utf-8"))
        reports = [rescore(planner, load_cases(args.cases), report) for report in saved]
        args.tag = args.tag or f"{saved[0].get('tag') or args.rescore.stem}-rescored"
    else:
        reports = asyncio.run(run(args.model, args.url, args.timeout, args.cases, args.tag, args.resume))
    print(format_report(reports))
    RESULTS_DIR.mkdir(exist_ok=True)
    for report in reports:
        report["tag"] = args.tag
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}{'-' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n자세한 결과: {out}")


if __name__ == "__main__":
    main()
