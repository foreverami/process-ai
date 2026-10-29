"""어느 폴더 이름에서든 서버를 띄우는 실행 스크립트.

코드는 process_ai 패키지로 import되지만 GitHub 저장소 폴더 이름은 process-ai다. 이 스크립트가 이 폴더를
process_ai라는 이름으로 등록한 뒤 서버를 띄우므로, 폴더 이름을 바꾸거나 상위 폴더로 옮길 필요가 없다.

    python3 start.py                      # http://localhost:8100 (로컬 LLM: PROCESS_LLM_URL, 기본 LM Studio)
    python3 start.py --port 8200
    PROCESS_LLM_URL=rules python3 start.py   # 로컬 LLM 없이 규칙 기반 추출기로
    python3 start.py eval --url rules        # 모델 비교 평가 (evaluation/run_eval.py 인자를 그대로 넘김)
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = "process_ai"


def ensure_package():
    """이 폴더를 process_ai 패키지로 등록한다 (이미 import돼 있으면 그대로)."""
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    spec = importlib.util.spec_from_file_location(PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["eval"]:
        ensure_package()
        from process_ai.evaluation.run_eval import main as eval_main

        sys.argv = [f"{sys.argv[0]} eval", *argv[1:]]
        eval_main()
        return
    parser = argparse.ArgumentParser(description="공정 자동화 AI 서버")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args(argv)
    ensure_package()
    import uvicorn

    llm = os.getenv("PROCESS_LLM_URL", "http://localhost:1234/v1")
    print(f"공정 자동화 AI: http://{args.host}:{args.port}  (추출기: {'규칙 기반, LLM 없음' if llm.strip().lower() in ('rules', 'rules://', 'builtin', 'none', 'off', '') else llm})")
    uvicorn.run("process_ai.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
