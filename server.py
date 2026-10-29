"""공정 자동화 서버: 운영 콘솔, 레시피 검증, 로컬 LLM 채팅, 공정 시작·정지, 실시간 상태.

실행:  python3 -m uvicorn process_ai.server:app --port 8100
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import config
from .controller import ProcessController, ProcessError
from .drivers import ArduinoDriver
from .examples import ExampleStore, recipe_values
from .knowledge import KnowledgeBase, KnowledgeError, Review
from .llm import LLMError
from .planner import DraftState, PlanResult, Planner
from .recipe import Recipe, active_minutes_upper_bound
from .rules import make_extractor
from .simulator import SimulatedChamber

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger("uvicorn.error")


async def warm_up_llm(planner: Planner, attempts: int = 3) -> None:
    """서버 시작 직후 짧은 문장을 한 번 보내 로컬 LLM이 시스템 프롬프트를 미리 처리하게 한다.

    프롬프트(약 1,500토큰) 처리는 메모리가 부족한 노트북에서 2분을 넘길 수 있어, 예열 없이는 작업자의
    첫 채팅이 시간 초과로 끝난다. LM Studio는 연결이 끊겨도 프롬프트 처리는 마치므로 재시도하면 캐시가 잡힌다.
    """
    for attempt in range(1, attempts + 1):
        try:
            interp = await planner.interpret("IPA 5 kPa 80도")
        except LLMError as e:
            log.warning("로컬 LLM 예열 %d/%d 실패: %s", attempt, attempts, e)
            continue
        log.info("로컬 LLM 예열 완료: %s (%.1f초)", interp.model, interp.latency_s)
        return


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    draft: DraftState = Field(default_factory=DraftState)


class Turn(BaseModel):
    message: str
    extracted: dict = Field(default_factory=dict)
    values: dict = Field(default_factory=dict)


class Feedback(BaseModel):
    """승인 때 화면이 보내는 학습 기록: 채팅 턴들과 AI가 만든 초안. 최종 레시피와 비교해 작업자의 수정을 남긴다."""

    turns: list[Turn] = Field(default_factory=list)
    ai_recipe: Recipe | None = None


class RunRequest(BaseModel):
    recipe: Recipe
    approved_by: str
    feedback: Feedback | None = None


class ReviewResult(Review):
    max_active_min: float
    recipe: Recipe  # 기본값이 채워진 레시피


def build_components() -> tuple[ProcessController, Planner, KnowledgeBase]:
    kb = KnowledgeBase(equipment=config.EQUIPMENT_ID, runs_dir=config.RUNS_DIR)
    if problems := kb.check_integrity():
        raise KnowledgeError("온톨로지 무결성 검사 실패:\n" + "\n".join(problems))
    if config.DRIVER == "arduino":
        driver = ArduinoDriver(config.SERIAL_PORT, config.SERIAL_BAUD)
    else:
        driver = SimulatedChamber(speed=config.SIM_SPEED)
    controller = ProcessController(driver, kb.limits(), kb.validate, runs_dir=config.RUNS_DIR)
    examples = ExampleStore(config.EXAMPLES_PATH, kb)
    # PROCESS_LLM_URL=rules 이면 로컬 LLM 대신 규칙 기반 추출기 (LLM을 돌릴 수 없는 PC용, rules.py)
    extractor = make_extractor(kb, config.LLM_BASE_URL, config.LLM_MODEL, config.LLM_TIMEOUT_S, config.LLM_REASONING_EFFORT)
    planner = Planner(kb, extractor, examples, config.FEWSHOT_K)
    return controller, planner, kb


def create_app(controller: ProcessController, planner: Planner, kb: KnowledgeBase, warm_up: bool = True) -> FastAPI:
    examples: ExampleStore | None = planner.examples
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(controller.run_forever())
        warm = asyncio.create_task(warm_up_llm(planner)) if warm_up else None
        yield
        if warm and not warm.done():
            warm.cancel()
        controller.stop()
        await task

    app = FastAPI(title="공정 자동화 AI", lifespan=lifespan)

    @app.get("/")
    async def console():
        # 브라우저가 예전 화면을 캐시에서 꺼내 쓰지 않게 한다 (코드를 고친 뒤 새로고침해도 옛 화면이 뜨는 문제)
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/status")
    async def status():
        return controller.status() | {"driver": type(controller.driver).__name__}

    @app.get("/api/llm/status")
    async def llm_status():
        return await planner.llm.status()

    @app.get("/api/knowledge/materials")
    async def materials():
        return kb.materials()

    @app.get("/api/knowledge/export")
    async def export_knowledge():
        return Response(
            await asyncio.to_thread(kb.export_turtle),
            media_type="text/turtle; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="process-knowledge.ttl"'},
        )

    @app.post("/api/recipe/validate", response_model=ReviewResult)
    async def validate(recipe: Recipe):
        review = await asyncio.to_thread(kb.review, recipe)
        return ReviewResult(
            **review.model_dump(),
            max_active_min=round(active_minutes_upper_bound(recipe), 1),
            recipe=recipe,
        )

    @app.post("/api/chat", response_model=PlanResult)
    async def chat(request: ChatRequest):
        try:
            return await planner.chat(request.message, request.draft)
        except LLMError as e:
            raise HTTPException(502, str(e)) from e

    @app.post("/api/runs")
    async def start_run(request: RunRequest):
        try:
            run_id = controller.start(request.recipe, request.approved_by)
        except ProcessError as e:
            raise HTTPException(409, str(e)) from e
        recorded = None
        if examples and request.feedback and request.feedback.turns:
            # 작업자가 검토 화면에서 고친 뒤 승인한 값이 정답이다. AI 초안과의 차이를 함께 남긴다
            ai_values = recipe_values(kb, request.feedback.ai_recipe) if request.feedback.ai_recipe else {}
            recorded = examples.record(
                [t.model_dump() for t in request.feedback.turns], ai_values, recipe_values(kb, request.recipe),
                request.approved_by, run_id,
            )
        return {"run_id": run_id, "example_id": recorded["id"] if recorded else None,
                "corrected": recorded["corrected"] if recorded else None}

    @app.get("/api/examples/stats")
    async def examples_stats():
        return examples.stats() if examples else {"count": 0, "single_turn": 0, "corrected": 0}

    @app.post("/api/runs/abort")
    async def abort_run():
        return {"aborted": controller.abort()}

    @app.post("/api/door/open")
    async def open_door():
        try:
            controller.request_open()
        except ProcessError as e:
            raise HTTPException(409, str(e)) from e
        return {"ok": True}

    @app.websocket("/ws/process")
    async def process_feed(websocket: WebSocket):
        await websocket.accept()
        queue = controller.subscribe()
        try:
            await websocket.send_json({"type": "status", **controller.status()})
            while True:
                await websocket.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            controller.unsubscribe(queue)

    return app


app = create_app(*build_components())
