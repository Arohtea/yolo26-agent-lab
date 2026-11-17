"""独立计算进程中的真实验证评估与工件追踪。"""

import hashlib
import json
import logging
import math
import os
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout

import mlflow
import numpy as np
import yaml
from rq import get_current_job
from sqlalchemy import update
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import DatasetExport
from yolo_agent.models.database import create_database_engine
from yolo_agent.models.dataset import utc_now
from yolo_agent.models.evaluation import EvaluationTask
from yolo_agent.models.experiment import Experiment, ModelArtifact
from yolo_agent.services.datasets import owned_path
from yolo_agent.services.error_analysis import analyze_errors
from yolo_agent.services.evaluation import fail_evaluation
from yolo_agent.services.runtime import collect_runtime, require_device
from yolo_agent.services.training import validate_export
from yolo_agent.services.ultralytics_runtime import configure_ultralytics
from yolo_agent.settings import get_settings
from yolo_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def on_evaluation_stopped(job, connection) -> None:
    """RQ 停止回调立即收敛验证任务。"""
    on_evaluation_failure(job, connection, RuntimeError, "操作者终止评估", None)


def on_evaluation_failure(job, connection, exc_type, exc_value, traceback) -> None:
    """计算超时与进程异常必须写回业务库。

    :param job: RQ 任务。
    :param connection: 当前队列连接。
    :param exc_type: 异常类型。
    :param exc_value: 异常内容。
    :param traceback: 堆栈。
    :returns: 无。
    """
    engine = create_database_engine(get_settings())
    try:
        fail_evaluation(engine, job.id, f"{getattr(exc_type,'__name__',str(exc_type))}: {exc_value}")
        from yolo_agent.services.garbage import collect_evaluation_garbage
        collect_evaluation_garbage(engine, get_settings(), job.id)
    finally:
        engine.dispose()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key):_json_value(item) for key,item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_evaluation_task(task_id: str) -> dict:
    """计算 mAP、逐类 AP、PR、混淆矩阵及错误，不提供测试集入口。

    :param task_id: 已登记的评估任务。
    :returns: 结构化报告摘要。
    """
    settings = get_settings()
    configure_logging(settings, "evaluation-task")
    job = get_current_job()
    if job is None or job.id != task_id:
        raise RuntimeError("评估只能由对应 RQ 任务执行")
    engine = create_database_engine(settings)
    try:
        with Session(engine, expire_on_commit=False) as session, session.begin():
            claimed = session.execute(update(EvaluationTask).where(EvaluationTask.id==task_id,EvaluationTask.status=="queued").values(status="running",started_at=utc_now()))
            if claimed.rowcount != 1:
                raise RuntimeError("评估已经执行或状态不允许启动")
            task = session.get(EvaluationTask, task_id)
            experiment = session.get(Experiment, task.experiment_id)
            model = session.get(ModelArtifact, task.model_id)
            export = session.get(DatasetExport, task.dataset_export_id)
            parameters, experiment_id, model_id = task.parameters, experiment.id, model.id
            version_id, export_id, export_directory = experiment.dataset_version_id, export.id, export.directory
            model_path, model_sha = owned_path(settings, model.path), model.sha256
            training_parameters, training_run = experiment.parameters, experiment.mlflow_run_id
            execution_limits = task.execution_limits
        # 校验完整数据可能持续较久，任务领取事务必须先提交，保留日志和控制写入能力。
        def progress(stage, completed_batches=0, total_batches=0):
            from yolo_agent.services.task_control import require_running
            with Session(engine) as session, session.begin():
                current = session.get(EvaluationTask, task_id)
                require_running(current)
                current.summary = {"progress": {"stage": stage, "completed_batches": completed_batches, "total_batches": total_batches}}
        progress("verifying_inputs")
        manifest = validate_export(settings, export, verify_files=True)
        with model_path.open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest() != model_sha:
                raise RuntimeError("注册权重指纹不符")
        runtime = collect_runtime(settings.device)
        device = require_device(runtime)
        output = owned_path(settings, f"evaluations/{task_id}")
        output.mkdir(parents=True, exist_ok=True)
        root = owned_path(settings, export_directory)
        config = yaml.safe_load((root / "data.yaml").read_text())
        config.pop("test", None)
        config["path"] = str(root)
        data_path = output / "validation-data.yaml"
        data_path.write_text(yaml.safe_dump(config, allow_unicode=True))
        classes = {item["class_id"]:item["name"] for item in manifest["classes"]}
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("YOLO26-Agent-Lab")
        with (output / "evaluation.log").open("w", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            configure_ultralytics(settings)
            from ultralytics import YOLO
            import torch

            torch.set_num_threads(min(4, torch.get_num_threads()))
            if device.startswith("cuda"):
                torch.cuda.set_per_process_memory_fraction(min(settings.max_gpu_memory_ratio,execution_limits.get("gpu_memory_ratio",0.9)), device=device)
            detector = YOLO(str(model_path))
            if detector.names != classes:
                raise RuntimeError("权重类别与已审核数据版本不一致")
            category_mapping = {}

            def capture_categories(validator) -> None:
                """从官方验证器读取 COCO 类别映射，避免把一基编号当作业务类别。

                :param validator: 当前 Ultralytics 验证器。
                :returns: 无；保存原始 JSON 类别 ID 到业务类别 ID 的映射。
                """
                category_mapping.update({category:class_id for class_id,category in enumerate(validator.class_map)})

            detector.add_callback("on_val_end", capture_categories)
            batch_progress = {"done": 0}
            def validation_batch(validator):
                batch_progress["done"] += 1
                progress("validating", batch_progress["done"], len(validator.dataloader))
            detector.add_callback("on_val_batch_end", validation_batch)
            progress("validating")
            metrics = detector.val(data=str(data_path), split="val", imgsz=training_parameters["imgsz"], tile=training_parameters.get("tile", False), batch=2 if device=="cpu" else 1, device=device, workers=0, conf=0.001, iou=0.7, max_det=300, save_json=True, plots=True, project=str(output), name="validation", exist_ok=True, verbose=True)
            validation_dir = output / "validation"
            predictions_path = validation_dir / "predictions.json"
            predictions = json.loads(predictions_path.read_text()) if predictions_path.is_file() else []
            predictions = [{**item,"class_id":category_mapping[item["category_id"]]} for item in predictions]
            progress("analyzing_errors")
            errors = analyze_errors(settings, task_id, root, output, classes, predictions, parameters)

            by_class = {int(class_id):index for index,class_id in enumerate(metrics.box.ap_class_index)}
            train_counts = Counter()
            for label in (root / "labels" / "train").glob("*.txt"):
                for line in label.read_text().splitlines():
                    if line.strip():
                        train_counts[int(line.split()[0])] += 1
            per_class = []
            for class_id,name in classes.items():
                index = by_class.get(class_id)
                precision, recall, ap50, ap = metrics.box.class_result(index) if index is not None else (None,)*4
                per_class.append({"class_id":class_id,"name":name,"train_instances":train_counts[class_id],"validation_instances":int(metrics.nt_per_class[class_id]),"precision":precision,"recall":recall,"ap50":ap50,"ap50_95":ap,"unavailable_reason":None if index is not None else "验证集没有此类真值，不能报告此类 AP"})
            curves = [{"name":name,"x":curve[0],"y":curve[1],"x_label":curve[2],"y_label":curve[3],"class_ids":metrics.box.ap_class_index} for name,curve in zip(metrics.curves,metrics.curves_results)]
            issues, recommendations = [], []
            if any(item["validation_instances"]==0 for item in per_class):
                issues.append({"code":"insufficient_validation_support","evidence":"部分已审核类别在固定验证集中没有真值，无法评价这些类别"})
                recommendations.append("collect_independent_validation_samples")
            if errors["fn"]:
                issues.append({"code":"missed_objects","count":errors["fn"]})
                recommendations.extend(["inspect_missed_validation_images","review_annotations"])
            if errors["fp"]:
                issues.append({"code":"false_positives","count":errors["fp"]})
                recommendations.append("inspect_false_positive_images")
            if errors["small_objects"]["total"] and errors["small_objects"]["recall"] < 0.5:
                issues.append({"code":"small_object_miss","evidence":errors["small_objects"]})
                recommendations.extend(["collect_small_object_samples","consider_increase_imgsz"])
            if max(train_counts.values(),default=0) > 3*min((train_counts[c] for c in classes),default=0):
                issues.append({"code":"class_imbalance","train_counts":dict(train_counts)})
                recommendations.append("collect_underrepresented_training_classes")
            summary = {"map50":metrics.box.map50,"map50_95":metrics.box.map,"precision":metrics.box.mp,"recall":metrics.box.mr,"validation_images":manifest["summary"]["split_counts"]["val"],"validation_instances":int(sum(metrics.nt_per_class)),"errors":errors,"issues":issues,"recommendations":list(dict.fromkeys(recommendations))}
            summary["metric_unavailable_reason"] = None
            if not summary["validation_instances"]:
                for key in ("map50","map50_95","precision","recall"):
                    summary[key] = None
                summary["metric_unavailable_reason"] = "验证集没有正目标真值，不能评估 AP 或召回"
            report = _json_value({"evaluation_id":task_id,"experiment_id":experiment_id,"model_id":model_id,"model_sha256":model_sha,"dataset_export_id":export_id,"dataset_version_id":version_id,"collected_at":utc_now(),"split":"val","test_used":False,"runtime":runtime.model_dump(mode="json"),"parameters":parameters,"metric_protocol":{"backend":"Ultralytics DetectionValidator","tile":training_parameters.get("tile",False),"imgsz":training_parameters["imgsz"],"ap_confidence_floor":0.001,"ap_iou_range":[0.5,0.95],"confusion_confidence":0.001,"confusion_iou":0.5,"note":"AP 的最佳 F1 操作点与错误清单的固定阈值不同，不能直接混同比较"},"summary":summary,"per_class":per_class,"curves":curves,"confusion_matrix":{"labels":[*classes.values(),"background"],"rows":"predicted","columns":"true","matrix":metrics.confusion_matrix.matrix},"limitations":["小样本验收不代表泛化能力","遮挡缺少真值元数据，未推断遮挡错误","测试集不参与评估或优化"]})
            report["prediction_categories"] = [{"coco_category_id":category,"class_id":class_id,"name":classes[class_id]} for category,class_id in category_mapping.items()]
            artifact_hashes = {}
            for file in output.rglob("*"):
                if file.is_file() and file.name not in {"evaluation.log","report.json"}:
                    with file.open("rb") as stream:
                        artifact_hashes[file.relative_to(output).as_posix()] = hashlib.file_digest(stream,"sha256").hexdigest()
            report["artifacts"] = artifact_hashes
            report_path = output / "report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
            with mlflow.start_run(tags={"kind":"validation","experiment_id":experiment_id,"evaluation_id":task_id,"model_id":model_id,"dataset_export_id":export_id,"training_mlflow_run_id":training_run or "","split":"val","test_used":"false"}) as tracking:
                mlflow.log_params({**parameters,"model_sha256":model_sha,"split":"val"})
                mlflow.log_metrics({key:float(summary[key]) for key in ("map50","map50_95","precision","recall") if summary[key] is not None and math.isfinite(summary[key])})
                mlflow.log_artifacts(str(output), artifact_path="evaluation")
                tracking_id = tracking.info.run_id
        with Session(engine) as session, session.begin():
            task = session.get(EvaluationTask, task_id)
            from yolo_agent.services.task_control import require_running
            require_running(task)
            task.status, task.finished_at = "completed", utc_now()
            task.summary = _json_value(summary)
            task.report_path = report_path.relative_to(settings.storage_dir.resolve()).as_posix()
            task.report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
            task.mlflow_run_id = tracking_id
        logger.info("真实验证评估完成", extra={"task_id":task_id,"device":device})
        return {"evaluation_id":task_id,"status":"completed","summary":_json_value(summary)}
    except Exception as error:
        fail_evaluation(engine, task_id, f"{type(error).__name__}: {error}")
        from yolo_agent.services.garbage import collect_evaluation_garbage
        collect_evaluation_garbage(engine, settings, task_id)
        logger.exception("验证评估失败", extra={"task_id":task_id})
        raise
    finally:
        engine.dispose()
