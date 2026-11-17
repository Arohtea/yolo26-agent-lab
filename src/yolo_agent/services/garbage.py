"""按任务归属回收可再生产物，数据库中仍被引用的原图和冻结数据始终保留。"""

import shutil
import logging

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import DatasetExport
from yolo_agent.models.dataset import DatasetImage, DatasetTask, DatasetVersion
from yolo_agent.models.experiment import ModelExport
from yolo_agent.models.evaluation import EvaluationTask
from yolo_agent.services.datasets import owned_path

logger = logging.getLogger(__name__)


def collect_evaluation_garbage(engine, settings, identity):
    """未发布的验证结果不可交付，失败后仅保留诊断日志。"""
    with Session(engine) as session:
        task = session.get(EvaluationTask, identity)
        if task is None or task.status not in {"failed", "stopped", "canceled"} or task.report_path:
            return
        directory = owned_path(settings, f"evaluations/{identity}")
        if not directory.is_dir():
            return
        for path in directory.iterdir():
            if path.name == "evaluation.log":
                continue
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                logger.warning("未完成评估文件回收失败", exc_info=True)


def collect_export_garbage(engine, settings, identity):
    """失败或终止的转换仅保留日志，注册权重与成功导出不在回收范围。"""
    with Session(engine) as session:
        record = session.get(ModelExport, identity)
        if record is None or record.status not in {"failed", "stopped", "canceled"} or record.path:
            return
        directory = owned_path(settings, f"model_exports/{identity}")
        if not directory.is_dir():
            return
        for path in directory.iterdir():
            if path.name == "export.log":
                continue
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                logger.warning("模型导出中间文件回收失败", exc_info=True)


def collect_task_garbage(engine, settings, identity):
    with Session(engine) as session, session.begin():
        # 与新任务登记共用 SQLite 写锁，避免清理刚好撞上下一次使用同一缓存。
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        task = session.get(DatasetTask, identity)
        if task is None or task.status not in {"completed", "failed", "stopped", "canceled"}:
            return []
        candidates = []
        run_ids = {task.parameters.get("run_id"), (task.result or {}).get("annotation_run_id")}
        for run_id in run_ids - {None}:
            active = session.scalar(select(DatasetTask.id).where(DatasetTask.parameters["run_id"].as_string() == run_id,
                DatasetTask.status.in_(["queued", "running", "canceling"])).limit(1))
            if not active:
                candidates.append(f"annotations/{run_id}/frames")
        if task.kind == "labeled_import":
            candidates += ([f"imports/{identity}/upload", f"imports/{identity}/extracted"] if task.status == "completed" else [f"imports/{identity}"])
            if task.status != "completed":
                candidates += [f"versions/{identity}", f"previews/imports/{identity}"]
        if task.kind == "dataset_export":
            candidates += [f"annotations/download-{identity}", f"dataset-downloads/{identity}"]
        if task.kind == "dataset_copy" and task.status != "completed":
            candidates += [f"images/{identity}", f"previews/copies/{identity}", f"versions/{identity}"]
        if task.status in {"failed", "stopped", "canceled"}:
            candidates += [f"dataset-downloads/{identity}", f"dataset-downloads/{identity}.zip"]
            export_id = ((task.result or {}).get("progress") or {}).get("export_id")
            if export_id and session.get(DatasetExport, export_id) is None:
                candidates.append(f"exports/{export_id}")
            version_id = ((task.result or {}).get("progress") or {}).get("version_id")
            if version_id and session.get(DatasetVersion, version_id) is None:
                candidates.append(f"versions/{version_id}")
        removed = []
        for relative in candidates:
            path = owned_path(settings, relative)
            # 查询源数据路径，而不以状态推断某目录可删。正常复制/导出不会把这里作为源数据。
            referenced = session.scalar(select(DatasetImage.id).where(or_(
                DatasetImage.storage_path.startswith(relative + "/", autoescape=True),
                DatasetImage.preview_path.startswith(relative + "/", autoescape=True))).limit(1))
            referenced = referenced or session.scalar(select(DatasetVersion.id).where(
                DatasetVersion.manifest_path.startswith(relative + "/", autoescape=True)).limit(1))
            referenced = referenced or session.scalar(select(DatasetExport.id).where(DatasetExport.archive_path == relative).limit(1))
            if not referenced and path.exists():
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    removed.append(relative)
                except OSError:
                    logger.warning("中间缓存清理失败，保留文件供后续回收", exc_info=True)
        return removed
