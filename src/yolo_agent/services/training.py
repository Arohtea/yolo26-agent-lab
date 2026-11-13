"""实验创建、受审数据验证及可恢复异步训练编排。"""

import hashlib
import json
from datetime import datetime, timezone

from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.serializers import JSONSerializer
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import AnnotationRun, DatasetExport
from yolo_agent.models.dataset import Dataset, DatasetVersion, new_id, utc_now
from yolo_agent.models.experiment import Experiment, TrainingTask
from yolo_agent.schemas.experiment import ExecutionLimits, ExperimentCreate, TrainingParameters
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.settings import Settings
from yolo_agent.workers.queue import enqueue_training_task, redis_connection


def validate_export(settings: Settings, export: DatasetExport, *, verify_files: bool) -> dict:
    """验证审核工件身份，执行端再逐文件验证数据，防止训练数据漂移。

    :param settings: 受控存储配置。
    :param export: 数据库中已封存的审核后工件。
    :param verify_files: 是否执行耗时逐文件校验。
    :returns: 已验证的工件清单。
    """
    if not export.summary.get("ready_for_training"):
        raise DatasetError(409, "数据未通过人工审核，禁止训练")
    root = owned_path(settings, export.directory)
    path = root / "manifest.json"
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != export.manifest_sha256:
        raise DatasetError(409, "数据工件清单缺失或指纹不符")
    manifest = json.loads(path.read_text())
    if manifest.get("export_id") != export.id or manifest.get("version_id") != export.version_id:
        raise DatasetError(409, "数据工件身份与实验绑定不一致")
    if verify_files:
        for relative, expected in manifest["sha256"].items():
            file = (root / relative).resolve()
            if not file.is_relative_to(root) or not file.is_file():
                raise DatasetError(409, "数据工件文件缺失或路径越界")
            with file.open("rb") as stream:
                if hashlib.file_digest(stream,"sha256").hexdigest() != expected:
                    raise DatasetError(409, f"数据工件文件指纹变化：{relative}")
    return manifest


def fail_training_task(engine: Engine, task_id: str, reason: str) -> None:
    """仅终结仍然活跃的任务，不覆盖已经完成或主动停止的状态。

    :param engine: 当前进程数据库引擎。
    :param task_id: 训练任务标识。
    :param reason: 不含服务凭据的失败说明。
    :returns: 无。
    """
    with Session(engine) as session, session.begin():
        task = session.get(TrainingTask, task_id)
        if task and task.status in {"queued","running","canceling"}:
            task.status, task.error, task.finished_at = "failed", reason[:4000], utc_now()
            if task.started_at:
                task.elapsed_seconds = (datetime.now(timezone.utc)-datetime.fromisoformat(task.started_at)).total_seconds()
            experiment = session.get(Experiment, task.experiment_id)
            experiment.status, experiment.error = "failed", reason[:4000]


class TrainingService:
    """API 只处理配置和状态，训练本身由独立 Worker 执行。"""

    def __init__(self, engine: Engine, settings: Settings):
        """绑定 API 进程资源。

        :param engine: 业务数据库。
        :param settings: 队列与模型配置。
        :returns: 无。
        """
        self.engine, self.settings = engine, settings

    def create_experiment(self, data: ExperimentCreate, *, experiment_id: str | None = None) -> Experiment:
        """从已审工件推导项目与版本，禁止调用者伪造数据绑定关系。

        :param data: 受限训练配置、可选父实验及优化原因。
        :param experiment_id: 仅内部编排器提供的确定性 ID，重放同一节点不重复创建实验。
        :returns: 已冻结配置的实验。
        """
        with Session(self.engine, expire_on_commit=False) as session, session.begin():
            existing = session.get(Experiment,experiment_id) if experiment_id else None
            if existing:
                if existing.dataset_export_id!=data.dataset_export_id or existing.parameters!=data.parameters.model_dump() or existing.parent_experiment_id!=data.parent_experiment_id or existing.name!=data.name or existing.optimization_reason!=data.optimization_reason:
                    raise DatasetError(409,"同一实验身份不能重放不同输入")
                return existing
            export = session.get(DatasetExport, data.dataset_export_id)
            if export is None:
                raise DatasetError(404, "已审核数据工件不存在，请先完成审核和导出")
            version = session.get(DatasetVersion, export.version_id)
            run = session.get(AnnotationRun, export.run_id)
            if not version.sealed or run.status != "exported":
                raise DatasetError(409, "数据版本或审核工件未封存")
            validate_export(self.settings, export, verify_files=False)
            dataset = session.get(Dataset, version.dataset_id)
            if dataset.deleted_at:
                raise DatasetError(409, "所属数据集已从工作区删除，不能创建新实验")
            parent = session.get(Experiment, data.parent_experiment_id) if data.parent_experiment_id else None
            if data.parent_experiment_id and (parent is None or parent.project_id != dataset.project_id):
                raise DatasetError(422, "父实验必须属于同一项目")
            parameters = data.parameters.model_dump()
            previous = TrainingParameters.model_validate(parent.parameters).model_dump() if parent else {}
            changed = {key:{"before":previous.get(key),"after":value} for key,value in parameters.items() if parent and previous.get(key) != value}
            if parent and parent.dataset_export_id != export.id:
                changed["dataset_export_id"] = {"before":parent.dataset_export_id,"after":export.id}
            experiment = Experiment(id=experiment_id or new_id(),project_id=dataset.project_id, dataset_version_id=version.id, dataset_export_id=export.id, parent_experiment_id=data.parent_experiment_id, name=data.name, parameters=parameters, optimization_reason=data.optimization_reason, changed_variables=changed)
            session.add(experiment)
        return experiment

    def start(self, experiment_id: str, *, resume: bool = False, execution_limits: ExecutionLimits | None = None) -> TrainingTask:
        """占用单机训练槽并异步提交，恢复只读取本实验已验证检查点。

        :param experiment_id: 实验标识。
        :param resume: 是否从该实验最后一次中断的检查点继续。
        :param execution_limits: 内部编排器提供的更严格资源限制。
        :returns: 已入队训练任务。
        """
        task_id = new_id()
        limits = execution_limits or ExecutionLimits(gpu_memory_ratio=self.settings.max_gpu_memory_ratio,timeout_seconds=min(172800,self.settings.training_job_timeout))
        checkpoint_identity = None
        if resume:
            with Session(self.engine) as session:
                experiment = session.get(Experiment, experiment_id)
                if experiment is None:
                    raise DatasetError(404, "实验不存在")
                version = session.get(DatasetVersion, experiment.dataset_version_id)
                if session.get(Dataset, version.dataset_id).deleted_at:
                    raise DatasetError(409, "所属数据集已删除，不能重新启动训练")
                if experiment.status in {"queued","running","canceling","completed"}:
                    raise DatasetError(409, "实验正在执行或已完成，不能重复启动")
                parent_task = session.scalar(select(TrainingTask).where(TrainingTask.experiment_id == experiment_id).order_by(TrainingTask.created_at.desc()).limit(1))
                if parent_task is None or parent_task.status not in {"failed","stopped"} or not parent_task.checkpoint_path:
                    raise DatasetError(409, "没有可恢复的中断检查点")
                checkpoint_identity = (parent_task.id, parent_task.checkpoint_path, parent_task.checkpoint_sha256)
            checkpoint = owned_path(self.settings, checkpoint_identity[1])
            if not checkpoint.is_file():
                raise DatasetError(409, "恢复检查点缺失")
            # 大检查点先在事务外核验，领取时重新比对身份，避免校验窗口内悄悄换源。
            with checkpoint.open("rb") as stream:
                if hashlib.file_digest(stream,"sha256").hexdigest() != checkpoint_identity[2]:
                    raise DatasetError(409, "恢复检查点指纹不符")
        try:
            with Session(self.engine, expire_on_commit=False) as session, session.begin():
                experiment = session.get(Experiment, experiment_id)
                if experiment is None:
                    raise DatasetError(404, "实验不存在")
                if experiment.status in {"queued","running","canceling","completed"}:
                    raise DatasetError(409, "实验正在执行或已完成，不能重复启动")
                parent_task = session.scalar(select(TrainingTask).where(TrainingTask.experiment_id == experiment_id).order_by(TrainingTask.created_at.desc()).limit(1))
                if resume:
                    if parent_task is None or parent_task.status not in {"failed","stopped"} or (parent_task.id, parent_task.checkpoint_path, parent_task.checkpoint_sha256) != checkpoint_identity:
                        raise DatasetError(409, "校验期间恢复来源已变化，请重新请求恢复")
                version = session.get(DatasetVersion, experiment.dataset_version_id)
                if session.get(Dataset, version.dataset_id).deleted_at:
                    raise DatasetError(409, "所属数据集已删除，不能重新启动训练")
                model = self.settings.training_model_dir / experiment.parameters["model"]
                if not resume and (not model.is_file() or not model.with_suffix(".provenance.json").is_file()):
                    raise DatasetError(503, "预训练模型未准备，请运行 scripts/prepare_training_models.py")
                task = TrainingTask(id=task_id, experiment_id=experiment_id, parent_task_id=parent_task.id if resume else None, total_epochs=experiment.parameters["epochs"], log_path=f"training/{task_id}/train.log",execution_limits=limits.model_dump())
                session.add(task)
                experiment.status, experiment.error = "queued", None
        except IntegrityError as error:
            raise DatasetError(409, "单机训练槽已有活跃任务，请等待或先停止当前训练") from error
        try:
            enqueue_training_task(self.settings, task_id,timeout_seconds=limits.timeout_seconds)
        except Exception as error:
            fail_training_task(self.engine, task_id, f"训练入队失败：{type(error).__name__}")
            raise DatasetError(503, "训练队列不可用，任务已保存为失败状态") from error
        return task

    def get_task(self, task_id: str) -> TrainingTask:
        """查询持久进度，并收敛 RQ 超时或工作进程退出状态。

        :param task_id: 任务标识。
        :returns: 训练任务快照。
        """
        with Session(self.engine) as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                raise DatasetError(404, "训练任务不存在")
            reason = None
            if task.status in {"queued","running","canceling"}:
                try:
                    with redis_connection(self.settings) as connection:
                        job = Job.fetch(task_id, connection=connection, serializer=JSONSerializer)
                        if job.get_status(refresh=True) in {"failed","stopped","canceled"}:
                            reason = "训练进程已终结，请查看日志及最后检查点"
                except NoSuchJobError:
                    if (datetime.now(timezone.utc)-datetime.fromisoformat(task.created_at)).total_seconds() > 60:
                        reason = "训练队列记录缺失，任务可能在入队时中断"
                except RedisError:
                    pass
            if reason:
                session.close()
                fail_training_task(self.engine, task_id, reason)
                return self.get_task(task_id)
            session.expunge(task)
            return task

    def cancel(self, task_id: str) -> TrainingTask:
        """向计算回调提交协作停止请求，保留有效优化器检查点。

        :param task_id: 任务标识。
        :returns: 已停止或正在停止的任务状态。
        """
        with Session(self.engine) as session, session.begin():
            task = session.get(TrainingTask, task_id)
            if task is None:
                raise DatasetError(404, "训练任务不存在")
            if task.status not in {"queued","running","canceling"}:
                raise DatasetError(409, "任务已经终结")
            task.stop_requested = True
            experiment = session.get(Experiment, task.experiment_id)
            if task.status == "queued":
                task.status, task.finished_at, experiment.status = "stopped", utc_now(), "stopped"
            else:
                task.status = experiment.status = "canceling"
        return self.get_task(task_id)
