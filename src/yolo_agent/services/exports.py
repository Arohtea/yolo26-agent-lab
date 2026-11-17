"""部署格式导出编排：操作者与 Agent 共用，实际转换由 Worker 执行。"""

from datetime import datetime, timezone

from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.serializers import JSONSerializer
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from yolo_agent.models.dataset import new_id, utc_now
from yolo_agent.models.experiment import ModelArtifact, ModelExport
from yolo_agent.schemas.experiment import ModelExportCreate
from yolo_agent.services.datasets import DatasetError
from yolo_agent.settings import Settings
from yolo_agent.workers.queue import enqueue_export_task, redis_connection


def fail_export(engine: Engine, export_id: str, reason: str) -> None:
    """只终结仍在进行的导出，已完成的文件记录不被覆盖。

    :param engine: 当前进程业务库。
    :param export_id: 导出任务。
    :param reason: 不含凭据的失败说明。
    :returns: 无。
    """
    with Session(engine) as session, session.begin():
        record = session.get(ModelExport, export_id)
        if record and record.status in {"queued", "running", "canceling"}:
            record.status, record.error, record.finished_at = ("stopped" if record.stop_requested else "failed"), ("操作者终止导出" if record.stop_requested else reason[:4000]), utc_now()


class ExportService:
    """导出不改变原始权重，也不等同于生产发布。"""

    def __init__(self, engine: Engine, settings: Settings):
        """绑定数据库和队列配置。

        :param engine: 业务数据库。
        :param settings: 队列与存储配置。
        :returns: 无。
        """
        self.engine, self.settings = engine, settings

    def create(self, model_id: str, data: ModelExportCreate, *, export_id: str | None = None, requested_by: str = "operator") -> ModelExport:
        """登记并提交一次格式转换，编排器重放同一身份时返回已有记录。

        :param model_id: 已注册权重。
        :param data: 目标格式与原因。
        :param export_id: 编排器生成的确定性身份，HTTP 入口不提供。
        :param requested_by: operator 或 agent:运行 ID，用于审计来源。
        :returns: 已入队或已存在的导出任务。
        """
        if export_id:
            with Session(self.engine) as session:
                existing = session.get(ModelExport, export_id)
                if existing:
                    if existing.model_id != model_id or existing.format != data.format:
                        raise DatasetError(409, "导出身份不能重放不同输入")
                    return self.get(export_id)
        try:
            with Session(self.engine, expire_on_commit=False) as session, session.begin():
                if session.get(ModelArtifact, model_id) is None:
                    raise DatasetError(404, "模型工件不存在")
                record = ModelExport(id=export_id or new_id(), model_id=model_id, format=data.format, requested_by=requested_by, reason=data.reason)
                session.add(record)
        except IntegrityError as error:
            raise DatasetError(409, "该权重已有同格式导出正在进行") from error
        try:
            enqueue_export_task(self.settings, record.id)
        except Exception as error:
            fail_export(self.engine, record.id, f"导出入队失败：{type(error).__name__}")
            raise DatasetError(503, "导出队列不可用，失败记录已保存") from error
        return record

    def get(self, export_id: str) -> ModelExport:
        """读取持久状态，并收敛 Worker 异常退出留下的活跃记录。

        :param export_id: 导出任务。
        :returns: 任务快照。
        """
        with Session(self.engine) as session:
            record = session.get(ModelExport, export_id)
            if record is None:
                raise DatasetError(404, "导出任务不存在")
            reason = None
            if record.status in {"queued", "running", "canceling"}:
                try:
                    with redis_connection(self.settings) as connection:
                        job = Job.fetch(export_id, connection=connection, serializer=JSONSerializer)
                        if job.get_status(refresh=True) in {"failed", "stopped", "canceled"}:
                            reason = "导出进程已终结"
                except NoSuchJobError:
                    if (datetime.now(timezone.utc) - datetime.fromisoformat(record.created_at)).total_seconds() > 60:
                        reason = "导出队列记录缺失"
                except RedisError:
                    pass
            if reason:
                session.close()
                fail_export(self.engine, export_id, reason)
                from yolo_agent.services.garbage import collect_export_garbage
                collect_export_garbage(self.engine, self.settings, export_id)
                return self.get(export_id)
            session.expunge(record)
            return record

    def list(self, *, model_ids: list[str]) -> list[ModelExport]:
        """按权重列出导出历史，最新在前。

        :param model_ids: 已按项目或实验筛选的权重。
        :returns: 导出记录。
        """
        if not model_ids:
            return []
        with Session(self.engine) as session:
            return list(session.scalars(select(ModelExport).where(ModelExport.model_id.in_(model_ids)).order_by(ModelExport.created_at.desc()).limit(200)))
