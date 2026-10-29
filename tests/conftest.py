"""폴더 이름이 process-ai든 process_ai든 테스트가 돌도록 start.py로 패키지를 등록한 뒤 fixture를 만든다."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("process_ai_start", Path(__file__).resolve().parents[1] / "start.py")
_start = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_start)
_start.ensure_package()

from process_ai.knowledge import KnowledgeBase  # noqa: E402


@pytest.fixture(scope="session")
def kb(tmp_path_factory):
    return KnowledgeBase(runs_dir=tmp_path_factory.mktemp("kb-runs"))
