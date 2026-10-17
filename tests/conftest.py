import pytest

from process_ai.knowledge import KnowledgeBase


@pytest.fixture(scope="session")
def kb(tmp_path_factory):
    return KnowledgeBase(runs_dir=tmp_path_factory.mktemp("kb-runs"))
