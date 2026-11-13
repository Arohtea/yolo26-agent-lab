"""Ultralytics 真实训练执行，指标来自回调，失败与停止均保留检查点。"""

import hashlib
import json
import logging
import math
import os
import shutil
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import mlflow
import yaml
from rq import get_current_job
from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import DatasetExport
from yolo_agent.models.database import create_database_engine
from yolo_agent.models.dataset import new_id, utc_now
from yolo_agent.models.experiment import Experiment, ExperimentMetric, ModelArtifact, TrainingTask
from yolo_agent.schemas.experiment import TrainingParameters
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.services.runtime import collect_runtime, require_device
from yolo_agent.services.training import fail_training_task, validate_export
from yolo_agent.services.ultralytics_runtime import configure_ultralytics
from yolo_agent.settings import get_settings
from yolo_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def on_training_failure(job, connection, exc_type, exc_value, traceback) -> None:
    """进程异常、超时也必须释放训练槽并记录失败。

    :param job: RQ 任务。
    :param connection: RQ 连接。
    :param exc_type: 异常类型。
    :param exc_value: 异常内容。
    :param traceback: 异常堆栈。
    :returns: 无。
    """
    engine = create_database_engine(get_settings())
    try:
        fail_training_task(engine, job.id, f"{getattr(exc_type,'__name__',str(exc_type))}: {exc_value}")
    finally:
        engine.dispose()


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def _numbers(values: dict) -> dict[str, float]:
    result = {}
    for key,value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[key] = number
    return result


def run_training_task(task_id: str) -> dict:
    """在独立工作进程内训练，所有输入都从已经校验的业务记录读取。

    :param task_id: 已登记的训练任务。
    :returns: 真实状态、指标及权重注册信息。
    """
    settings = get_settings()
    configure_logging(settings,"training-task")
    job = get_current_job()
    if job is None or job.id != task_id:
        raise RuntimeError("训练只能由对应 RQ 任务执行")
    engine = create_database_engine(settings)
    started = time.monotonic()
    try:
        with Session(engine, expire_on_commit=False) as session, session.begin():
            task = session.get(TrainingTask, task_id)
            if task is not None and task.status == "stopped":
                return {"task_id":task_id,"status":"stopped","reason":"queued_task_canceled"}
            claimed = session.execute(update(TrainingTask).where(TrainingTask.id == task_id, TrainingTask.status == "queued").values(status="running", started_at=utc_now()))
            if claimed.rowcount != 1:
                raise RuntimeError("训练任务已被执行或状态不允许启动")
            task = session.get(TrainingTask, task_id)
            experiment = session.get(Experiment, task.experiment_id)
            export = session.get(DatasetExport, experiment.dataset_export_id)
            experiment.status = "running"
            experiment_id, parameters, export_id = experiment.id, TrainingParameters.model_validate(experiment.parameters).model_dump(), export.id
            parent_experiment_id, version_id = experiment.parent_experiment_id, experiment.dataset_version_id
            export_directory, log_relative = export.directory, task.log_path
            parent_task = session.get(TrainingTask, task.parent_task_id) if task.parent_task_id else None
            parent_checkpoint = parent_task.checkpoint_path if parent_task else None
            parent_checksum = parent_task.checkpoint_sha256 if parent_task else None
            execution_limits = task.execution_limits
        # 工件不可变，领取任务后再校验大文件，避免全量读取期间占用 SQLite 写锁。
        export_manifest = validate_export(settings, export, verify_files=True)
        report = collect_runtime(settings.device)
        device = require_device(report)
        output = owned_path(settings, f"training/{task_id}")
        output.mkdir(parents=True, exist_ok=True)
        manifest_copy = output / "dataset-manifest.json"
        manifest_copy.write_text(json.dumps(export_manifest,ensure_ascii=False,indent=2)+"\n")

        # 训练只暴露 train/val，测试集既不传入训练器也不参与参数选择。
        data = yaml.safe_load((owned_path(settings,export_directory)/"data.yaml").read_text())
        data.pop("test",None)
        data["path"] = str(owned_path(settings,export_directory))
        data_path = output / "training-data.yaml"
        data_path.write_text(yaml.safe_dump(data,allow_unicode=True))
        resolved = dict(parameters)
        if parameters["batch"] in ("auto", -1) or isinstance(parameters["batch"], float):
            if device.startswith("cuda"):
                resolved["batch"] = parameters["batch"] if isinstance(parameters["batch"], float) else -1
            else:
                resolved["batch"] = max(1,min(4,export_manifest["summary"]["split_counts"]["train"]))
        # 打开训练库自带绘图：Agent 每轮结束后要读取 results、混淆矩阵与 PR 等 val 图。
        resolved.update(device=device,workers=0,amp=False,plots=True,save=True,val=True,verbose=True,project=str(output),name="run",exist_ok=True)
        with Session(engine) as session, session.begin():
            session.get(TrainingTask, task_id).device = device
            session.get(Experiment, experiment_id).resolved_parameters = resolved

        state = {"epoch":None,"last_stop_check":0.0}
        resume_path = output / "resume.pt"
        log_path = owned_path(settings,log_relative)
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT","10")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES","2")
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("YOLO26-Agent-Lab")
        with mlflow.start_run(tags={"kind":"training","is_training":"true","experiment_id":experiment_id,"training_task_id":task_id,"dataset_version_id":version_id,"dataset_export_id":export_id,"parent_experiment_id":parent_experiment_id or ""}) as tracking:
            with Session(engine) as session, session.begin():
                session.get(TrainingTask,task_id).mlflow_run_id = tracking.info.run_id
                session.get(Experiment,experiment_id).mlflow_run_id = tracking.info.run_id
            mlflow.log_params({**parameters,"resolved_device":device,"resolved_batch":resolved["batch"],"dataset_manifest_sha256":hashlib.sha256(manifest_copy.read_bytes()).hexdigest()})
            mlflow.log_artifact(str(manifest_copy),artifact_path="dataset")

            def check_stop(trainer) -> None:
                """以有界频率读取停止请求，使用训练库的批次间协作退出。"""
                if time.monotonic()-state["last_stop_check"] < 0.5:
                    return
                state["last_stop_check"] = time.monotonic()
                with Session(engine) as session:
                    if session.get(TrainingTask,task_id).stop_requested:
                        trainer.stop = True

            def start_epoch(trainer) -> None:
                """区分真实训练 epoch 与训练库末尾的额外最佳权重验证事件。"""
                state["epoch"] = int(trainer.epoch)

            def record_optimizer(trainer) -> None:
                """保存自动优化器实际解析结果，避免把请求 lr0 冒充已采用值。

                :param trainer: 已建立优化器的官方训练器。
                :returns: 无；实际优化器及参数组写入实验解析配置。
                """
                actual={"class":type(trainer.optimizer).__name__,"groups":[_numbers({key:group[key] for key in ("lr","initial_lr","weight_decay") if key in group}) for group in trainer.optimizer.param_groups]}
                trainer_parameters = {key: getattr(trainer.args, key) for key in TrainingParameters.model_fields if key != "model" and hasattr(trainer.args, key)}
                augmentation = getattr(trainer.train_loader.dataset, "hyp", trainer.args)
                effective_augmentation = {key: getattr(augmentation, key) for key in ("mosaic", "mixup", "cutmix", "close_mosaic") if hasattr(augmentation, key)}
                with Session(engine) as session,session.begin():
                    experiment=session.get(Experiment,experiment_id)
                    experiment.resolved_parameters={**experiment.resolved_parameters,"optimizer_effective":actual,"trainer_parameters":trainer_parameters,"augmentation_effective_at_start":effective_augmentation,"gpu_memory_ratio_limit":min(settings.max_gpu_memory_ratio,execution_limits.get("gpu_memory_ratio",0.9))}

            def save_resume(trainer) -> None:
                """在最终去除优化器前保存可恢复副本，并原子替换上一检查点。"""
                if trainer.last.is_file():
                    temporary = output / "resume.pending.pt"
                    shutil.copy2(trainer.last,temporary)
                    temporary.replace(resume_path)
                    digest = _hash(resume_path)
                    with Session(engine) as session, session.begin():
                        current = session.get(TrainingTask,task_id)
                        current.checkpoint_path = resume_path.relative_to(settings.storage_dir.resolve()).as_posix()
                        current.checkpoint_sha256 = digest

            def record_epoch(trainer) -> None:
                """只记录训练库实际给出的有限指标，不生成占位 Loss 或 mAP。"""
                if int(trainer.epoch) != state["epoch"]:
                    return
                epoch = int(trainer.epoch)+1
                values = _numbers({**trainer.label_loss_items(trainer.tloss),**trainer.metrics,**trainer.lr})
                with Session(engine) as session, session.begin():
                    for name,value in values.items():
                        statement = insert(ExperimentMetric).values(experiment_id=experiment_id,task_id=task_id,epoch=epoch,name=name,value=value,created_at=utc_now())
                        session.execute(statement.on_conflict_do_update(index_elements=["task_id","epoch","name"],set_={"value":value}))
                    current = session.get(TrainingTask,task_id)
                    current.current_epoch, current.elapsed_seconds = epoch,time.monotonic()-started
                # MLflow 不接受 Ultralytics 指标中的括号；业务库保留原名便于追溯。
                mlflow.log_metrics({key.replace("(","").replace(")",""):value for key,value in values.items()},step=epoch)

            with log_path.open("a",buffering=1,encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                configure_ultralytics(settings)
                from ultralytics import YOLO
                import torch

                torch.set_num_threads(min(4,torch.get_num_threads()))
                if device.startswith("cuda"):
                    torch.cuda.set_per_process_memory_fraction(min(settings.max_gpu_memory_ratio,execution_limits.get("gpu_memory_ratio",0.9)), device=device)
                if parent_checkpoint:
                    checkpoint = owned_path(settings,parent_checkpoint)
                    if _hash(checkpoint) != parent_checksum:
                        raise DatasetError(409,"恢复检查点已变化")
                    model = YOLO(str(checkpoint))
                else:
                    pretrained = settings.training_model_dir / parameters["model"]
                    provenance = json.loads(pretrained.with_suffix(".provenance.json").read_text())
                    if provenance["repo_id"] != "Ultralytics/YOLO26" or provenance.get("revision") != "070ac3c51435984d992ba92eb3f234daf4e4500d" or provenance["model"] != parameters["model"] or _hash(pretrained) != provenance["sha256"]:
                        raise DatasetError(409,"预训练模型来源或指纹不符")
                    model = YOLO(str(pretrained))
                model.add_callback("on_train_epoch_start",start_epoch)
                model.add_callback("on_train_start",record_optimizer)
                model.add_callback("on_train_batch_end",check_stop)
                model.add_callback("on_model_save",save_resume)
                model.add_callback("on_fit_epoch_end",record_epoch)
                model.train(data=str(data_path),resume=bool(parent_checkpoint),**resolved)
                trainer = model.trainer
                final_metrics = _numbers(trainer.metrics)
                with Session(engine) as session:
                    stopped = session.get(TrainingTask,task_id).stop_requested
                records = []
                for kind,source in (("best",trainer.best),("last",trainer.last)):
                    if source.is_file():
                        artifact_id = new_id()
                        relative = f"registry/{artifact_id}/{kind}.pt"
                        destination = owned_path(settings,relative)
                        destination.parent.mkdir(parents=True)
                        shutil.copy2(source,destination)
                        digest = _hash(destination)
                        destination.chmod(0o444)
                        records.append(ModelArtifact(id=artifact_id,experiment_id=experiment_id,task_id=task_id,kind=kind,path=relative,sha256=digest,metrics=final_metrics))
                        mlflow.log_artifact(str(destination),artifact_path=f"weights/{kind}")
                if not stopped and not any(record.kind == "best" for record in records):
                    raise RuntimeError("训练结束但没有生成 best.pt")
                status = "stopped" if stopped else "completed"
                best_relative = trainer.best.relative_to(settings.storage_dir.resolve()).as_posix() if trainer.best.is_file() else None
                mlflow.set_tags({"training_status":status,"actual_device":device})
                if stopped:
                    mlflow.set_tag("stop_requested",True)
                result = {"task_id":task_id,"experiment_id":experiment_id,"status":status,"device":device,"metrics":final_metrics,"models":[{"id":record.id,"kind":record.kind,"sha256":record.sha256} for record in records]}
        with Session(engine) as session, session.begin():
            current = session.get(TrainingTask,task_id)
            current.status,current.finished_at,current.elapsed_seconds = status,utc_now(),time.monotonic()-started
            current.best_path = best_relative
            experiment = session.get(Experiment,experiment_id)
            experiment.status,experiment.final_metrics = status,final_metrics
            session.add_all(records)
        logger.info("真实训练结束",extra={"task_id":task_id,"result":result})
        if status == "completed":
            # 训练提交完成后另排独立评估，评估不可占用 API 或改变已完成训练的事实。
            from yolo_agent.services.evaluation import EvaluationService
            from yolo_agent.schemas.evaluation import EvaluationCreate

            try:
                best_model = next(item for item in result["models"] if item["kind"]=="best")
                from yolo_agent.schemas.experiment import ExecutionLimits
                evaluation = EvaluationService(engine,settings).create(best_model["id"],EvaluationCreate(),execution_limits=ExecutionLimits.model_validate(execution_limits))
                result["evaluation_id"] = evaluation.id
            except Exception:
                logger.exception("训练已完成，但自动评估提交失败，可通过模型评估接口重新提交",extra={"task_id":task_id})
        return result
    except Exception as error:
        fail_training_task(engine,task_id,f"{type(error).__name__}: {error}")
        logger.exception("训练失败，保留已写检查点，不自动重试",extra={"task_id":task_id})
        raise
    finally:
        engine.dispose()
