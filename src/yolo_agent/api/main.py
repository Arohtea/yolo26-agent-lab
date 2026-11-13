"""训练平台 API、工作台与进程生命周期管理。"""

import json
import logging
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text

from yolo_agent import __version__
from yolo_agent.api.datasets import router as dataset_router
from yolo_agent.api.annotations import router as annotation_router
from yolo_agent.api.experiments import router as experiment_router
from yolo_agent.api.evaluations import router as evaluation_router
from yolo_agent.api.agents import router as agent_router
from yolo_agent.api.workspace import router as workspace_router
from yolo_agent.api.web import assets, router as web_router
from yolo_agent.services.datasets import DatasetError
from yolo_agent.models.database import create_database_engine
from yolo_agent.services.runtime import RuntimeReport, collect_runtime
from yolo_agent.settings import get_settings
from yolo_agent.utils.logging import configure_logging
from yolo_agent.workers.queue import RUNTIME_PREFIX, redis_connection

logger = logging.getLogger(__name__)


class DependencyStatus(BaseModel):
    """依赖状态中只暴露错误类型，连接凭据不进入 HTTP 响应。"""

    status: str
    error_type: str | None = None


class LiveResponse(BaseModel):
    """进程存活契约，不以外部依赖作为存活条件。"""

    status: str = "alive"
    version: str = __version__


class ReadyResponse(BaseModel):
    """三个必需依赖全部成功时 status 为 ready。"""

    status: str
    dependencies: dict[str, DependencyStatus]


class WorkerSnapshot(BaseModel):
    """尚未过期的 Worker 环境报告。"""

    worker_name: str
    runtime: RuntimeReport


class RuntimeResponse(BaseModel):
    """API 环境与最近 Worker 环境分开，避免混淆执行位置。"""

    api: RuntimeReport
    latest_worker: WorkerSnapshot | None = None
    workers: list[WorkerSnapshot]
    worker_report_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """创建进程独享的数据库连接池并确保退出时释放。

    :param app: 当前 FastAPI 实例。
    :yields: 可处理请求的生命周期窗口。
    """
    settings = get_settings()
    configure_logging(settings, "api")
    app.state.engine = create_database_engine(settings)
    logger.info("API 启动", extra={"requested_device": settings.device})
    try:
        yield
    finally:
        app.state.engine.dispose()


app = FastAPI(
    title="YOLO26 Agent Lab", version=__version__, lifespan=lifespan,
    description="数据、人审、YOLO26 异步训练、验证评估与有限预算自主优化。未审核数据不可训练，生产发布必须人工确认。",
)
app.include_router(dataset_router)
app.include_router(annotation_router)
app.include_router(experiment_router)
app.include_router(evaluation_router)
app.include_router(agent_router)
app.include_router(workspace_router)
app.include_router(web_router)
app.mount("/assets", assets, name="web-assets")


@app.exception_handler(DatasetError)
async def dataset_error_handler(request: Request, error: DatasetError) -> JSONResponse:
    """将可预期业务错误转为稳定的 HTTP 错误响应。

    :param request: 当前请求。
    :param error: 业务服务错误。
    :returns: 不暴露异常堆栈的说明与可选任务 ID。
    """
    return JSONResponse(status_code=error.status_code, content={"detail": str(error), "task_id": error.task_id})


@app.get("/api/v1/health/live", response_model=LiveResponse, tags=["健康检查"])
def live() -> LiveResponse:
    """查询 API 进程是否可以处理请求。

    :returns: 存活状态及应用版本。
    """
    return LiveResponse()


@app.get(
    "/api/v1/health/ready", response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse, "description": "至少一个必需依赖不可用"}},
    tags=["健康检查"],
)
def ready(request: Request, response: Response) -> ReadyResponse:
    """短时检查 SQLite、Redis 和 MLflow，不修改外部服务数据。

    :param request: 提供当前应用数据库引擎的请求。
    :param response: 根据依赖状态设置 HTTP 状态码。
    :returns: 每个必需依赖的状态；任意异常时 HTTP 503。
    """
    settings = get_settings()
    statuses: dict[str, DependencyStatus] = {}
    try:
        with request.app.state.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        statuses["sqlite"] = DependencyStatus(status="ok")
    except Exception as exc:
        statuses["sqlite"] = DependencyStatus(status="unavailable", error_type=type(exc).__name__)
    try:
        with redis_connection(settings) as connection:
            connection.ping()
        statuses["redis"] = DependencyStatus(status="ok")
    except Exception as exc:
        statuses["redis"] = DependencyStatus(status="unavailable", error_type=type(exc).__name__)
    try:
        with httpx.Client(timeout=settings.health_timeout, trust_env=False) as client:
            result = client.get(f"{settings.mlflow_tracking_uri.rstrip('/')}/health")
            result.raise_for_status()
        statuses["mlflow"] = DependencyStatus(status="ok")
    except Exception as exc:
        statuses["mlflow"] = DependencyStatus(status="unavailable", error_type=type(exc).__name__)
    healthy = all(status.status == "ok" for status in statuses.values())
    if not healthy:
        response.status_code = 503
        logger.warning("就绪检查发现依赖不可用", extra={"dependencies": {name: item.model_dump() for name, item in statuses.items()}})
    return ReadyResponse(status="ready" if healthy else "not_ready", dependencies=statuses)


@app.get("/api/v1/system/runtime", response_model=RuntimeResponse, tags=["运行环境"])
def runtime() -> RuntimeResponse:
    """采集 API 环境并读取带 TTL 的 Worker 独立环境快照。

    :returns: API 环境、在线 Worker 报告及报告读取错误。
    """
    settings = get_settings()
    workers: list[WorkerSnapshot] = []
    error = None
    try:
        with redis_connection(settings) as connection:
            for index, key in enumerate(connection.scan_iter(match=f"{RUNTIME_PREFIX}*", count=100)):
                if index >= 100:
                    break
                raw = connection.get(key)
                if raw:
                    try:
                        report = RuntimeReport.model_validate(json.loads(raw))
                        workers.append(WorkerSnapshot(worker_name=key.decode().removeprefix(RUNTIME_PREFIX), runtime=report))
                    except (ValueError, TypeError):
                        logger.warning("忽略无法解析的 Worker 环境报告")
    except Exception as exc:
        error = type(exc).__name__
    workers.sort(key=lambda item: item.runtime.collected_at, reverse=True)
    return RuntimeResponse(
        api=collect_runtime(settings.device), workers=workers,
        latest_worker=workers[0] if workers else None, worker_report_error=error,
    )
