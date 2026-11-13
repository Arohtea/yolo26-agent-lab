"""在 RQ 独立进程内执行真实图像检查、分析与版本封存。"""

import hashlib
import json
import logging
from pathlib import Path

from rq import get_current_job
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from yolo_agent.models.database import create_database_engine
from yolo_agent.models.dataset import DatasetImage, DatasetTask, DatasetVersion, VersionImage, new_id, utc_now
from yolo_agent.schemas.dataset import ImageRead
from yolo_agent.services.datasets import fail_dataset_task, owned_path
from yolo_agent.services.inspector import analyze_images, inspect_image, split_images, encoded_image_suffix
from yolo_agent.settings import get_settings
from yolo_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _write_json(path: Path, content: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _snapshots(session: Session, ids: list[str]) -> list[dict]:
    images = []
    # 控制 IN 子句大小，同时保持任务提交时冻结的稳定顺序。
    by_id = {}
    for start in range(0, len(ids), 500):
        for image in session.scalars(select(DatasetImage).where(DatasetImage.id.in_(ids[start:start + 500]))):
            by_id[image.id] = image
    for image_id in ids:
        image = by_id.get(image_id)
        if image is None:
            raise ValueError("任务输入图片缺失，拒绝生成不完整分析或版本")
        images.append({**ImageRead.model_validate(image).model_dump(), "storage_path": image.storage_path})
    return images


def on_dataset_failure(job, connection, exc_type, exc_value, traceback) -> None:
    """在超时或计算进程异常退出时记录失败并释放数据集任务占用。

    :param job: RQ 任务对象。
    :param connection: RQ 提供的 Redis 连接。
    :param exc_type: 异常类型。
    :param exc_value: 异常内容。
    :param traceback: RQ 异常堆栈。
    :returns: 无。
    """
    engine = create_database_engine(get_settings())
    try:
        fail_dataset_task(engine, job.id, f"{getattr(exc_type, '__name__', str(exc_type))}: {exc_value}")
    finally:
        engine.dispose()


def run_dataset_task(task_id: str) -> dict:
    """仅执行持久任务表中的白名单操作，耗时处理不占用 API 线程。

    :param task_id: 已持久化的任务标识。
    :returns: 导入、分析或版本创建的可追踪结果摘要。
    :raises RuntimeError: 任务状态不允许重复执行。
    """
    settings = get_settings()
    configure_logging(settings, "dataset-task")
    job = get_current_job()
    if job is None or job.id != task_id:
        raise RuntimeError("数据集处理必须通过对应的 RQ 任务执行")
    engine = create_database_engine(settings)
    try:
        with Session(engine) as session, session.begin():
            claimed = session.execute(update(DatasetTask).where(DatasetTask.id == task_id, DatasetTask.status == "queued").values(status="running", started_at=utc_now()))
            if claimed.rowcount != 1:
                raise RuntimeError("任务不存在、已执行或已终结，拒绝重复执行")
            task = session.get(DatasetTask, task_id)
            kind, dataset_id, parameters = task.kind, task.dataset_id, task.parameters
        logger.info("开始处理数据集", extra={"task_id": task_id, "kind": kind, "dataset_id": dataset_id})
        if kind == "labeled_import":
            from yolo_agent.workers.dataset_transfer import import_labeled
            return import_labeled(engine, settings, task_id, dataset_id, parameters)
        if kind == "dataset_export":
            from yolo_agent.workers.dataset_transfer import export_dataset
            return export_dataset(engine, settings, task_id, parameters)
        if kind == "dataset_copy":
            from yolo_agent.workers.dataset_copy import copy_dataset
            return copy_dataset(engine, settings, task_id, dataset_id, parameters)
        version_record = None
        version_members = []

        if kind in {"import", "image_recheck"}:
            adaptations = []
            with Session(engine) as session:
                ids = parameters["image_ids"] if kind == "image_recheck" else list(session.scalars(select(DatasetImage.id).where(DatasetImage.import_task_id == task_id).order_by(DatasetImage.created_at, DatasetImage.id)))
            for index, image_id in enumerate(ids, start=1):
                with Session(engine) as session:
                    image = session.get(DatasetImage, image_id)
                    source = owned_path(settings, image.storage_path)
                preview_relative = f"previews/{image_id}.jpg"
                information = inspect_image(source, owned_path(settings, preview_relative), settings.max_image_pixels)
                information.setdefault("error", None)
                information["preview_path"] = preview_relative if information["status"] == "valid" else None
                if information["status"] == "valid":
                    suffix = encoded_image_suffix(source, information["format"])
                    if suffix != source.suffix.lower():
                        normalized = source.with_suffix(suffix)
                        source.rename(normalized)
                        information["storage_path"] = normalized.relative_to(settings.storage_dir.resolve()).as_posix()
                        adaptations.append({"image_id": image_id, "original_name": image.original_name, "encoding": information["format"], "storage_extension": suffix})
                with Session(engine) as session, session.begin():
                    session.execute(update(DatasetImage).where(DatasetImage.id == image_id).values(**information))
                    session.execute(update(DatasetTask).where(DatasetTask.id == task_id).values(processed=index))
            with Session(engine) as session:
                images = _snapshots(session, ids)
            report, _ = analyze_images(images, 4, settings.max_similarity_pairs)
            report["format_adaptations"] = adaptations
            result = {"image_count": len(ids), "valid_images": report["valid_images"], "status_counts": report["status_counts"], "corrected_extensions": len(adaptations)}
        elif kind in {"analysis", "version"}:
            with Session(engine) as session:
                images = _snapshots(session, parameters["image_ids"])
            policy = parameters["policy"]
            report, graph = analyze_images(images, policy["phash_threshold"], settings.max_similarity_pairs)
            result = {"image_count": len(images), "valid_images": report["valid_images"], "exact_duplicate_excess": report["exact_duplicate_excess"], "near_duplicate_groups": len(report["near_duplicate_groups"])}
            if kind == "version":
                snapshots, summary = split_images(images, graph, policy)
                # 封存前校验实际原图，阻止外部文件变动悄悄改变数据集版本。
                for image in snapshots:
                    with owned_path(settings, image["storage_path"]).open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != image["sha256"]:
                            raise ValueError(f"原图已发生变化，拒绝封存：{image['id']}")
                version_id = new_id()
                with Session(engine) as session:
                    number = (session.scalar(select(func.max(DatasetVersion.number)).where(DatasetVersion.dataset_id == dataset_id)) or 0) + 1
                created_at = utc_now()
                manifest_relative = f"versions/{version_id}/manifest.json"
                manifest = {
                    "schema_version": 1, "version_id": version_id, "dataset_id": dataset_id,
                    "number": number, "created_at": created_at, "policy": policy,
                    "summary": summary, "quality_report": report, "images": snapshots,
                    "annotation_status": "unlabeled", "ready_for_training": False,
                }
                manifest_path = owned_path(settings, manifest_relative)
                digest = _write_json(manifest_path, manifest)
                manifest_path.chmod(0o444)
                version_record = DatasetVersion(
                        id=version_id, dataset_id=dataset_id, task_id=task_id, number=number,
                        name=policy["name"], policy=policy, summary=summary,
                        manifest_path=manifest_relative, manifest_sha256=digest, created_at=created_at,
                    )
                version_members = [VersionImage(version_id=version_id, image_id=image["id"], split=image["split"], group_id=image["group_id"], snapshot=image) for image in snapshots]
                result.update(version_id=version_id, number=number, manifest_sha256=digest, summary=summary)
        else:
            raise ValueError("任务类型未在数据集处理白名单中")

        report.update(task_id=task_id, dataset_id=dataset_id, created_at=utc_now(), scope_image_ids=[image["id"] for image in images])
        report_relative = f"reports/{task_id}.json"
        _write_json(owned_path(settings, report_relative), report)
        with Session(engine) as session, session.begin():
            if version_record is not None:
                session.add(version_record)
                session.flush()
                session.add_all(version_members)
                session.flush()
                version_record.sealed = True
            session.execute(update(DatasetTask).where(DatasetTask.id == task_id).values(status="completed", result=result, processed=len(images), report_path=report_relative, finished_at=utc_now()))
        logger.info("数据集处理完成", extra={"task_id": task_id, "kind": kind, "result": result})
        return result
    except Exception as exc:
        fail_dataset_task(engine, task_id, f"{type(exc).__name__}: {exc}")
        logger.exception("数据集处理失败，不自动重试", extra={"task_id": task_id})
        raise
    finally:
        engine.dispose()
