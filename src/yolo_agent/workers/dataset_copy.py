"""跨项目复制独立原图、冻结划分和标注；不通过有损的导出再导入传递审核。"""

import hashlib
import shutil
from collections import Counter, defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import Annotation, AnnotationRun, ClassSchema, ImageReview, ReviewEvent
from yolo_agent.models.dataset import Dataset, DatasetImage, DatasetTask, DatasetVersion, VersionImage, new_id, utc_now
from yolo_agent.schemas.dataset import ImageRead
from yolo_agent.services.annotations import annotation_digest
from yolo_agent.services.datasets import DatasetService, owned_path
from yolo_agent.services.inspector import analyze_images
from yolo_agent.workers.dataset_tasks import _snapshots, _write_json


def copy_dataset(engine, settings, task_id, dataset_id, parameters):
    """失败时清除本任务私有文件，目标保留失败记录以便查看原因。"""
    try:
        return _copy_dataset(engine, settings, task_id, dataset_id, parameters)
    except Exception:
        for directory in (f"images/{task_id}", f"previews/copies/{task_id}", f"versions/{task_id}"):
            shutil.rmtree(owned_path(settings, directory), ignore_errors=True)
        raise


def _copy_dataset(engine, settings, task_id, dataset_id, parameters):
    """文件采用 shutil 独立复制，全部完成后一次发布业务记录。"""
    source_version_id = parameters.get("version_id")
    if source_version_id:
        DatasetService(engine, settings).manifest_path(source_version_id)
    with Session(engine) as session:
        # SQLite 默认延迟 BEGIN；显式读事务保证框、审核凭据来自同一时刻。
        session.connection().exec_driver_sql("BEGIN")
        project_id = session.get(Dataset, dataset_id).project_id
        version = session.get(DatasetVersion, source_version_id) if source_version_id else None
        rows = list(session.scalars(select(VersionImage).where(VersionImage.version_id == source_version_id).order_by(VersionImage.image_id))) if version else []
        snapshots = [{**row.snapshot, "split": row.split, "group_id": row.group_id} for row in rows] if version else _snapshots(session, parameters["image_ids"])
        if {row["id"] for row in snapshots} != set(parameters["image_ids"]):
            raise ValueError("复制范围与提交时不一致")
        previews = {}
        for start in range(0, len(snapshots), 500):
            ids = [row["id"] for row in snapshots[start:start + 500]]
            previews.update(session.execute(select(DatasetImage.id, DatasetImage.preview_path).where(DatasetImage.id.in_(ids))).all())
        snapshots = [{**row, "preview_path": previews.get(row["id"])} for row in snapshots]
        run = session.get(AnnotationRun, parameters["run_id"]) if parameters.get("run_id") else None
        if run and (run.version_id != source_version_id or run.status not in {"pending_review", "reviewed", "exported"}):
            raise ValueError("来源标注仍在变化，请完成来源处理后重新复制")
        schema = session.get(ClassSchema, run.class_schema_id) if run else None
        originals = list(session.scalars(select(Annotation).where(Annotation.run_id == run.id))) if run else []
        original_reviews = {review.image_id: review for review in session.scalars(select(ImageReview).where(ImageReview.run_id == run.id))} if run else {}
        for obj in [version, run, schema, *originals, *original_reviews.values()]:
            if obj is not None:
                session.expunge(obj)
    required = sum(row["size_bytes"] for row in snapshots)
    if shutil.disk_usage(settings.storage_dir).free < required + 256 * 1024**2:
        raise ValueError("数据盘空间不足以创建独立副本，请释放空间后重新复制")
    provenance = {key: parameters.get(key) for key in ("source_dataset_id", "source_project_id", "source_dataset_name", "version_id", "run_id")}
    provenance.update(copied_at=utc_now(), new_manual_review_performed=False)
    images, members, id_map = [], [], {}
    for index, source in enumerate(snapshots, start=1):
        identity = new_id()
        id_map[source["id"]] = identity
        path = owned_path(settings, source["storage_path"])
        relative = f"images/{task_id}/{identity}{path.suffix}"
        destination = owned_path(settings, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        with destination.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if source.get("sha256") and digest != source["sha256"]:
            raise ValueError("来源图片指纹变化，拒绝发布副本")
        preview = None
        if source.get("preview_path") and owned_path(settings, source["preview_path"]).is_file():
            preview = f"previews/copies/{task_id}/{identity}.jpg"
            owned_path(settings, preview).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(owned_path(settings, source["preview_path"]), owned_path(settings, preview))
        values = {key: source.get(key) for key in ("original_name", "status", "size_bytes", "sha256", "phash", "width", "height", "format", "exif_orientation", "error", "video_id", "scene_id", "batch_id")}
        image = DatasetImage(id=identity, dataset_id=dataset_id, import_task_id=task_id,
            storage_path=relative, preview_path=preview, created_at=utc_now(), **values)
        images.append(image)
        if version:
            members.append({**ImageRead.model_validate(image).model_dump(), "storage_path": relative,
                "split": source["split"], "group_id": source["group_id"]})
        with Session(engine) as session, session.begin():
            session.get(DatasetTask, task_id).processed = index
    copied_run_id, schema_id, version_id = (new_id(), new_id(), task_id) if run else (None, None, task_id if version else None)
    boxes, reviews, events = [], [], []
    source_boxes, copied_boxes = defaultdict(list), defaultdict(list)
    for original in originals:
        if original.image_id not in id_map:
            raise ValueError("来源标注引用了版本外图片")
        source_boxes[original.image_id].append(original)
        values = {key: getattr(original, key) for key in ("class_id", "class_name", "bbox", "confidence", "annotation_source", "annotation_model_version", "review_status", "flags", "reviewer")}
        box = Annotation(id=new_id(), run_id=copied_run_id, image_id=id_map[original.image_id], **values)
        boxes.append(box)
        copied_boxes[original.image_id].append(box)
    for old_id, identity in id_map.items():
        if not run:
            break
        old = original_reviews.get(old_id)
        inherited = bool(old and old.status in {"approved", "corrected"} and old.missed_objects_checked
            and old.reviewer and old.annotation_digest == annotation_digest(source_boxes[old_id])
            and all(box.review_status != "pending" for box in source_boxes[old_id]))
        reviews.append(ImageReview(run_id=copied_run_id, image_id=identity,
            status=old.status if inherited else "pending", missed_objects_checked=inherited,
            reviewer=old.reviewer if inherited else None, reviewed_at=old.reviewed_at if inherited else None,
            annotation_digest=annotation_digest(copied_boxes[old_id]) if inherited else None))
        events.append(ReviewEvent(run_id=copied_run_id, image_id=identity, reviewer=old.reviewer if inherited else "system:dataset_copy",
            source="dataset_copy", before=[], after=[{"source": provenance, "source_image_id": old_id,
                "source_annotation_digest": old.annotation_digest if old else None, "inherited_review": inherited}]))
    ready = bool(run and len(reviews) == len(images) and all(review.status in {"approved", "corrected"} for review in reviews))
    summary = {"image_count": len(images), "annotation_count": len(boxes),
        "class_distribution": dict(Counter(box.class_name for box in boxes if box.review_status != "rejected")),
        "split_counts": dict(Counter(member["split"] for member in members)),
        "reviewed_images": sum(review.status != "pending" for review in reviews),
        "unreviewed_images": sum(review.status == "pending" for review in reviews) if run else len(images),
        "ready_for_training": ready, "source_split_preserved": bool(version), "source": provenance}
    manifest_path, checksum = None, None
    if version:
        manifest_path = f"versions/{version_id}/manifest.json"
        quality, _ = analyze_images(members, 4, settings.max_similarity_pairs)
        quality.update(annotation_status="reviewed" if ready else "pending_review" if run else "unlabeled",
            ready_for_training=ready, class_distribution=summary["class_distribution"])
        checksum = _write_json(owned_path(settings, manifest_path), {"version_id": version_id, "dataset_id": dataset_id,
            "number": 1, "images": members, "summary": summary, "quality_report": quality, "source": provenance})
    result = {"dataset_id": dataset_id, "version_id": version_id, "annotation_run_id": copied_run_id, "summary": summary}
    report = f"reports/dataset-copy-{task_id}.json"
    _write_json(owned_path(settings, report), {**result, "source_image_ids": id_map})
    with Session(engine) as session, session.begin():
        target = session.get(Dataset, dataset_id)
        if target is None or target.deleted_at:
            raise ValueError("目标数据集已删除，拒绝发布副本")
        session.add_all(images)
        if version:
            copied_version = DatasetVersion(id=version_id, dataset_id=dataset_id, task_id=task_id, number=1,
                name=f"{version.name} · 跨项目副本", policy={"entry": "dataset_copy", "preserve_source_splits": True, "source": provenance},
                summary=summary, manifest_path=manifest_path, manifest_sha256=checksum)
            session.add(copied_version)
            session.flush()
            session.add_all([VersionImage(version_id=version_id, image_id=member["id"], split=member["split"], group_id=member["group_id"], snapshot=member) for member in members])
            session.flush()
            copied_version.sealed = True
        if run:
            session.add(ClassSchema(id=schema_id, project_id=project_id, name=f"{schema.name} · 副本", classes=schema.classes))
            session.flush()
            session.add(AnnotationRun(id=copied_run_id, version_id=version_id, class_schema_id=schema_id,
                status="reviewed" if ready else "pending_review", parameters={**run.parameters, "copy_source": provenance},
                model_version=run.model_version, processed=len(images), total=len(images)))
            session.flush()
            session.add_all(boxes + reviews + events)
        task = session.get(DatasetTask, task_id)
        task.status, task.result, task.report_path, task.finished_at = "completed", result, report, utc_now()
    return result
