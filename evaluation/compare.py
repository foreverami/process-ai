"""여러 평가 결과 파일을 실험 버전 × 모델 표 하나로 묶는다 (논문·보고서용).

    python3 -m process_ai.evaluation.compare results/*-v1-baseline.json results/*-v2-hints-guard.json
"""
import argparse
import json
from pathlib import Path

from .run_eval import percent


def load(paths: list[Path]) -> list[dict]:
    reports = []
    for path in paths:
        for report in json.loads(path.read_text(encoding="utf-8")):
            report["tag"] = report.get("tag") or path.stem
            reports.append(report)
    return reports


def format_table(reports: list[dict]) -> str:
    lines = [
        "| 실험 | 모델 | 통과 사례 | 필드 정확도 | 환각 | 검사 제외 환각 | 보완 제외 정확도 | 평균 응답(초) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r['tag']} | {r['model']} | {r['passed']}/{r['cases']} | {percent(r['field_accuracy'])} | "
            f"{r['hallucinations']} | {r.get('hallucinations_no_guard', '-')} | "
            f"{percent(r.get('field_accuracy_no_fill'))} | {r['avg_latency_s']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="평가 결과 비교표")
    parser.add_argument("results", type=Path, nargs="+", help="run_eval이 저장한 결과 JSON 파일들")
    args = parser.parse_args()
    print(format_table(load(args.results)))


if __name__ == "__main__":
    main()
