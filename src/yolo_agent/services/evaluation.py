"""评估编排只接受已注册模型与其绑定的已审数据。"""

from datetime import datetime, timezone
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.serializers import JSONSerializer
from redis.exceptions import RedisError
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from yolo_agent.models.dataset import new_id, utc_now
from yolo_agent.models.evaluation import EvaluationTask
from yolo_agent.models.experiment import Experiment, ModelArtifact
from yolo_agent.schemas.evaluation import EvaluationCreate
from yolo_agent.schemas.experiment import ExecutionLimits
from yolo_agent.services.datasets import DatasetError
from yolo_agent.settings import Settings
from yolo_agent.workers.queue import enqueue_evaluation_task, redis_connection


def fail_evaluation(engine: Engine, task_id: str, reason: str) -> None:
    """保留评估异常，评估失败不抹掉已经完成的训练。

    :param engine: 当前进程业务库。
    :param task_id: 评估任务。
    :param reason: 已去除凭据的错误说明。
    :returns: 无。
    """
    with Session(engine) as session, session.begin():
        task = session.get(EvaluationTask, task_id)
        if task and task.status in {"queued", "running"}:
            task.status, task.error, task.finished_at = "failed", reason[:4000], utc_now()


class EvaluationService:
    """API 和训练完成回调共用的异步评估入口。"""

    def __init__(self, engine: Engine, settings: Settings):
        """绑定短事务数据库和队列配置。

        :param engine: 数据库。
        :param settings: 执行配置。
        :returns: 无。
        """
        self.engine, self.settings = engine, settings

    def create(self, model_id: str, data: EvaluationCreate, *, task_id: str | None = None, execution_limits: ExecutionLimits | None = None) -> EvaluationTask:
        """通过模型推导数据绑定，不允许调用者替换数据或选择测试集。

        :param model_id: 已注册候选模型。
        :param data: 诊断阈值。
        :param task_id: 编排器重放时的确定性身份，不接受 HTTP 传入。
        :param execution_limits: 内部资源上限，只能收紧全局限制。
        :returns: 已提交的评估任务。
        """
        if task_id:
            with Session(self.engine) as session:
                existing=session.get(EvaluationTask,task_id)
                if existing:
                    if existing.model_id!=model_id or existing.parameters!=data.model_dump():
                        raise DatasetError(409,"评估身份不能重放不同输入")
                    return self.get(task_id)
        limits=execution_limits or ExecutionLimits(gpu_memory_ratio=self.settings.max_gpu_memory_ratio,timeout_seconds=min(172800,self.settings.evaluation_job_timeout))
        try:
            with Session(self.engine, expire_on_commit=False) as session, session.begin():
                model = session.get(ModelArtifact, model_id)
                if model is None:
                    raise DatasetError(404, "模型不存在")
                experiment = session.get(Experiment, model.experiment_id)
                if experiment.status != "completed":
                    raise DatasetError(409, "只评估已完成实验，停止任务的中间权重不能作为最终结果")
                task = EvaluationTask(id=task_id or new_id(), experiment_id=experiment.id, model_id=model_id, dataset_export_id=experiment.dataset_export_id, parameters=data.model_dump(),execution_limits=limits.model_dump())
                session.add(task)
        except IntegrityError as error:
            raise DatasetError(409, "模型已有活跃评估任务") from error
        try:
            enqueue_evaluation_task(self.settings, task.id,timeout_seconds=limits.timeout_seconds)
        except Exception as error:
            fail_evaluation(self.engine, task.id, f"评估入队失败：{type(error).__name__}")
            raise DatasetError(503, "评估入队失败，失败记录已保存") from error
        return task

    def get(self, task_id: str) -> EvaluationTask:
        """读取持久状态并处理工作进程异常退出。

        :param task_id: 评估标识。
        :returns: 任务快照。
        """
        with Session(self.engine) as session:
            task = session.get(EvaluationTask, task_id)
            if task is None:
                raise DatasetError(404, "评估任务不存在")
            reason = None
            if task.status in {"queued", "running"}:
                try:
                    with redis_connection(self.settings) as connection:
                        job = Job.fetch(task_id, connection=connection, serializer=JSONSerializer)
                        if job.get_status(refresh=True) in {"failed", "stopped", "canceled"}:
                            reason = "评估进程已终结"
                except NoSuchJobError:
                    if (datetime.now(timezone.utc)-datetime.fromisoformat(task.created_at)).total_seconds() > 60:
                        reason = "评估队列记录缺失"
                except RedisError:
                    pass
            if reason:
                session.close()
                fail_evaluation(self.engine, task_id, reason)
                return self.get(task_id)
            session.expunge(task)
            return task
