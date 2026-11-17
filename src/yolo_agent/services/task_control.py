"""复用 RQ 的撤销与进程停止能力，统一操作者的阶段终止语义。"""

from rq.command import send_stop_job_command
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.serializers import JSONSerializer
from redis.exceptions import RedisError
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from yolo_agent.models.agent import AgentIteration, AgentRun
from yolo_agent.models.dataset import DatasetImage, DatasetTask, utc_now
from yolo_agent.models.evaluation import EvaluationTask
from yolo_agent.models.experiment import ModelArtifact, ModelExport
from yolo_agent.services.datasets import DatasetError
from yolo_agent.workers.queue import redis_connection

MODELS = {"dataset": DatasetTask, "evaluation": EvaluationTask, "export": ModelExport}
ACTIVE = {"queued", "running", "waiting", "needs_review", "paused", "canceling"}


def stop_owner(session, kind, task):
    """操作者终止阶段时同步停止所属流程，禁止 Agent 将人工停止当失败重试。"""
    experiment_id = getattr(task, "experiment_id", None)
    if kind == "export":
        model = session.get(ModelArtifact, task.model_id)
        experiment_id = model.experiment_id if model else None
    owners = set(session.scalars(select(AgentIteration.run_id).where(AgentIteration.experiment_id == experiment_id))) if experiment_id else set()
    for run in session.scalars(select(AgentRun).where(AgentRun.status.in_(ACTIVE))):
        waiting = run.state.get("waiting", {})
        owned = run.id in owners or waiting.get("task_id") == task.id
        if kind == "dataset" and task.parameters.get("run_id"):
            owned = owned or run.state.get("annotation_run_id") == task.parameters["run_id"]
        if owned:
            run.stop_requested, run.pause_requested, run.status = True, False, "canceling"
            run.state = {**run.state, "stop_reason": "操作者终止当前阶段，不再自动重试"}


def require_running(task):
    """在业务最终发布前检查停止信号，避免取消与结果提交竞争。"""
    if task is None or task.status != "running" or task.stop_requested:
        raise DatasetError(409, "任务已终止，不再发布后续结果")


def cancel_task(engine, settings, kind, identity, *, stop_flow=True):
    model = MODELS[kind]
    with Session(engine) as session, session.begin():
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        task = session.get(model, identity)
        if task is None:
            raise DatasetError(404, "任务不存在")
        if task.status not in {"queued", "running", "canceling"}:
            return {"id": identity, "status": task.status}
        task.stop_requested, task.status = True, "canceling"
        task.error = "操作者请求终止"
        if stop_flow:
            stop_owner(session, kind, task)
    terminal = False
    try:
        with redis_connection(settings) as connection:
            try:
                job = Job.fetch(identity, connection=connection, serializer=JSONSerializer)
                state = job.get_status(refresh=True)
                if state == "started":
                    send_stop_job_command(connection, identity)
                else:
                    if state not in {"finished", "failed", "stopped", "canceled"}:
                        job.cancel()
                    terminal = True
            except NoSuchJobError:
                terminal = True
    except RedisError as error:
        raise DatasetError(503, "停止请求已保存，队列暂不可用；任务不再发布结果，可稍后重试终止") from error
    if terminal:
        with Session(engine) as session, session.begin():
            task = session.get(model, identity)
            if task.status == "canceling":
                task.status, task.finished_at = "stopped", utc_now()
                if kind == "dataset":
                    session.execute(update(DatasetImage).where(DatasetImage.import_task_id == identity,
                        DatasetImage.status == "pending").values(status="failed", error="操作者终止检查，图片尚未处理"))
                if kind == "dataset" and task.kind == "auto_annotate":
                    from yolo_agent.models.annotation import AnnotationRun
                    run = session.get(AnnotationRun, task.parameters["run_id"])
                    if run:
                        run.status, run.error = "stopped", "操作者终止候选标注"
        if kind == "dataset":
            from yolo_agent.services.garbage import collect_task_garbage
            collect_task_garbage(engine, settings, identity)
        elif kind == "export":
            from yolo_agent.services.garbage import collect_export_garbage
            collect_export_garbage(engine, settings, identity)
        elif kind == "evaluation":
            from yolo_agent.services.garbage import collect_evaluation_garbage
            collect_evaluation_garbage(engine, settings, identity)
    with Session(engine) as session:
        task = session.get(model, identity)
        return {"id": identity, "status": task.status}
