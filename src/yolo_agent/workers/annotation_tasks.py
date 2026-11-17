"""真实检测、CVAT 人工审核同步及已审核 YOLO 工件导出。"""

import hashlib
import json
import logging
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from rq import get_current_job
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import Annotation, AnnotationRun, ClassSchema, DatasetExport, ImageReview, ReviewEvent
from yolo_agent.models.database import create_database_engine
from yolo_agent.models.dataset import DatasetTask, DatasetVersion, VersionImage, new_id, utc_now
from yolo_agent.schemas.annotation import AnnotationRead
from yolo_agent.services.annotations import annotation_digest, assert_reviewed
from yolo_agent.services.datasets import DatasetError, fail_dataset_task, owned_path
from yolo_agent.services.task_control import require_running
from yolo_agent.settings import get_settings
from yolo_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def on_annotation_failure(job, connection, exc_type, exc_value, traceback) -> None:
    """使超时和进程退出同时反映在任务及推理批次状态中。

    :param job: 失败的 RQ 任务。
    :param connection: RQ 连接。
    :param exc_type: 异常类型。
    :param exc_value: 异常内容。
    :param traceback: 异常堆栈。
    :returns: 无。
    """
    engine = create_database_engine(get_settings())
    try:
        fail_dataset_task(engine, job.id, f"{getattr(exc_type, '__name__', str(exc_type))}: {exc_value}")
        from yolo_agent.services.garbage import collect_task_garbage
        collect_task_garbage(engine, get_settings(), job.id)
    finally:
        engine.dispose()


def _verify_image(settings, snapshot: dict) -> Path:
    path = owned_path(settings, snapshot["storage_path"])
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != snapshot["sha256"]:
            raise DatasetError(409, "原图指纹变化，拒绝继续标注或导出")
    return path


def _prepare_frames(settings, run_id: str, members: list[dict], progress=None) -> list[Path]:
    paths = []
    for member in members:
        source = _verify_image(settings, member)
        if member.get("exif_orientation", 1) in {None, 1}:
            # CVAT 直接接收正向 JPEG/MPO、PNG 和 WEBP；仅重命名独立副本以关联图片编号。
            target = owned_path(settings, f"annotations/{run_id}/frames/{member['id']}{source.suffix.lower()}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            paths.append(target)
            if progress:
                progress("preparing_images", len(paths))
            continue
        target = owned_path(settings, f"annotations/{run_id}/frames/{member['id']}.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            if oriented.size != (member["width"], member["height"]):
                raise DatasetError(409, "方向转换后的图片尺寸与封存版本不一致")
            if target.is_file():
                try:
                    with Image.open(target) as cached:
                        if cached.format == "PNG" and cached.mode == "RGB" and cached.getexif().get(274, 1) == 1 and cached.size == oriented.size and cached.tobytes() == oriented.tobytes():
                            paths.append(target)
                            if progress:
                                progress("preparing_images", len(paths))
                            continue
                except (OSError, ValueError, SyntaxError):
                    pass  # 不复用损坏缓存，仍从已验证指纹的原图重新生成。
            oriented.save(target, format="PNG")
        paths.append(target)
        if progress:
            progress("preparing_images", len(paths))
    return paths


def run_annotation_task(task_id: str) -> dict:
    """按业务任务白名单分派，任何自动模型结果均不自动批准。

    :param task_id: 已持久化的任务 ID。
    :returns: 真实推理、CVAT 操作或已审核导出的结果。
    """
    settings = get_settings()
    configure_logging(settings, "annotation-task")
    job = get_current_job()
    if job is None or job.id != task_id:
        raise RuntimeError("标注处理必须由对应的 RQ 任务执行")
    engine = create_database_engine(settings)
    try:
        with Session(engine) as session, session.begin():
            changed = session.execute(update(DatasetTask).where(DatasetTask.id == task_id, DatasetTask.status == "queued").values(status="running", started_at=utc_now()))
            if changed.rowcount != 1:
                raise RuntimeError("任务已执行或状态不允许重新执行")
            task = session.get(DatasetTask, task_id)
            kind, parameters = task.kind, task.parameters
            run = session.get(AnnotationRun, parameters["run_id"])
            schema_id = parameters.get("reviewed_class_schema_id") if kind == "cvat_pull" else None
            schema = session.get(ClassSchema, schema_id or run.class_schema_id)
            reviewed_schema_id = schema.id
            member_rows = list(session.execute(select(VersionImage.snapshot, VersionImage.split).where(VersionImage.version_id == run.version_id).order_by(VersionImage.image_id)))
            members = [{**snapshot, "split": split} for snapshot, split in member_rows]
            classes, policy, run_id, project_id = schema.classes, run.parameters, run.id, schema.project_id
            cvat_task_id = run.cvat_task_id
            if kind == "auto_annotate":
                run.status = "running"
                session.add_all([ImageReview(run_id=run_id, image_id=member["id"]) for member in members])
        logger.info("开始标注任务", extra={"task_id":task_id,"operation":kind,"run_id":run_id})

        def report_progress(stage, completed_images=0, **details):
            with Session(engine) as session, session.begin():
                current = session.get(DatasetTask, task_id)
                require_running(current)
                current.processed = completed_images
                current.result = {"progress": {"stage": stage, "completed_images": completed_images,
                                  "total_images": len(members), "estimated": False, **details}}

        export_record = None
        if kind == "auto_annotate":
            # 使用标注专用设备，不继承训练 Worker 的 CUDA 选择。
            settings = settings.model_copy(update={"device": settings.annotation_device})
            import torch
            torch.set_num_threads(settings.annotation_cpu_threads)
            from yolo_agent.services.runtime import collect_runtime, require_device
            device=require_device(collect_runtime(settings.device))
            if device.startswith("cuda"):
                torch.cuda.set_per_process_memory_fraction(min(settings.max_gpu_memory_ratio, settings.annotation_gpu_memory_ratio, parameters.get("execution_limits",{}).get("gpu_memory_ratio",0.9)),device=device)
            report_progress("loading_annotation_models", 0, device=device, worker_queue=job.origin)
            from yolo_agent.services.annotators import AnnotationDispatcher, cvat_eligible, validate_source_plan

            with Session(engine) as session:
                manifest = validate_source_plan(session, settings, project_id, classes, policy.get("class_sources", {}), coco_model=policy.get("coco_model", "yolo26n.pt"), has_test=any(member["split"] == "test" for member in members))
            if policy.get("_annotators") is not None and manifest != policy["_annotators"]:
                raise DatasetError(409, "标注器来源与创建时的冻结清单不一致")
            detector = AnnotationDispatcher(settings, classes, policy, manifest)
            for index, member in enumerate(members, start=1):
                predictions = detector.detect(_verify_image(settings, member), member["split"])
                with Session(engine) as session, session.begin():
                    current = session.get(AnnotationRun, run_id)
                    for prediction in predictions:
                        session.add(Annotation(run_id=run_id, image_id=member["id"], **prediction))
                    current.processed = index
                    session.get(DatasetTask, task_id).processed = index
                    session.get(DatasetTask, task_id).result = {"progress": {"stage": "annotating_images", "completed_images": index,
                        "total_images": len(members), "device": device, "worker_queue": job.origin, "estimated": False}}
            with Session(engine) as session, session.begin():
                require_running(session.get(DatasetTask, task_id))
                session.get(AnnotationRun, run_id).status = "pending_review"
                boxes = list(session.scalars(select(Annotation).where(Annotation.run_id == run_id)))
                result = {"run_id":run_id,"status":"pending_review","image_count":len(members),"box_count":len(boxes),"proposed_class_distribution":dict(Counter(box.class_name for box in boxes)),"source_counts":dict(Counter(box.annotation_source for box in boxes)),"cvat_eligible_boxes":sum(cvat_eligible(AnnotationRead.model_validate(box).model_dump()) for box in boxes),"flag_counts":dict(Counter(flag for box in boxes for flag in box.flags)),"routing":[{"split":split,"source":source,"class_id":class_id,"images":count} for (split,source,class_id),count in sorted(detector.routing_counts.items())],"device":device,"is_ground_truth":False}
        elif kind == "cvat_push":
            from yolo_agent.services.cvat import upload_cvat

            report_progress("preparing_images", 0)
            frames = _prepare_frames(settings, run_id, members, progress=report_progress)
            with Session(engine) as session:
                boxes = [AnnotationRead.model_validate(box).model_dump() for box in session.scalars(select(Annotation).where(Annotation.run_id == run_id))]
            def remember_remote(remote_id):
                with Session(engine) as session, session.begin():
                    # 即使收到停止请求，也记录刚刚创建的远程任务，供查看和清理。
                    session.get(AnnotationRun, run_id).cvat_task_id = remote_id
            remote_id, digest = upload_cvat(settings, run_id, classes, frames, boxes,
                                            progress=lambda stage: report_progress(stage, len(frames)), on_created=remember_remote)
            with Session(engine) as session, session.begin():
                require_running(session.get(DatasetTask, task_id))
                current = session.get(AnnotationRun, run_id)
                current.cvat_task_id, current.cvat_fingerprint = remote_id, digest
            result = {"run_id":run_id,"cvat_task_id":remote_id,"review_url":f"{(settings.cvat_public_url or settings.cvat_url).rstrip('/')}/tasks/{remote_id}","status":"pending_review"}
        elif kind == "cvat_pull":
            from yolo_agent.services.cvat import download_cvat

            report_progress("reading_review")
            remote = download_cvat(settings, cvat_task_id, require_accepted=True)
            report_progress("validating_review")
            by_id = {member["id"]:member for member in members}
            if set(remote["frames"]) != set(by_id) or len(remote["frames"]) != len(members):
                raise DatasetError(409, "CVAT 帧集合已变化，不能导入审核结果")
            label_map = {category["name"]:category for category in classes}
            grouped = defaultdict(list)
            for shape in remote["shapes"]:
                name = remote["labels"].get(shape["label_id"])
                if name not in label_map or not 0 <= shape["frame"] < len(remote["frames"]):
                    raise DatasetError(422, "CVAT 存在未知类别或无效帧")
                image_id = remote["frames"][shape["frame"]]
                box = shape["points"]
                member = by_id[image_id]
                if len(box) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box) or not (0 <= box[0] < box[2] <= member["width"] and 0 <= box[1] < box[3] <= member["height"]):
                    raise DatasetError(422, "CVAT 框坐标不合法")
                attribute_id = remote["platform_attributes"].get(shape["label_id"])
                platform_id = next((attribute["value"] for attribute in shape.get("attributes", []) if attribute["spec_id"] == attribute_id), None)
                grouped[image_id].append((label_map[name], box, platform_id))
            with Session(engine) as session, session.begin():
                require_running(session.get(DatasetTask, task_id))
                current = session.get(AnnotationRun, run_id)
                current.class_schema_id = reviewed_schema_id
                if session.scalar(select(DatasetExport.id).where(DatasetExport.run_id == run_id)):
                    raise DatasetError(409, "已导出批次不可覆盖")
                original = list(session.scalars(select(Annotation).where(Annotation.run_id == run_id)))
                before_by_image = defaultdict(list)
                originals_by_id = {box.id:box for box in original}
                for box in original:
                    before_by_image[box.image_id].append(AnnotationRead.model_validate(box).model_dump())
                for annotation in original:
                    annotation.review_status, annotation.reviewer = "rejected", parameters["reviewer"]
                for image_id in by_id:
                    additions, reused = [], set()
                    for category, box, platform_id in grouped[image_id]:
                        matched = originals_by_id.get(platform_id)
                        if platform_id and (matched is None or matched.image_id != image_id):
                            raise DatasetError(422, "CVAT 框引用了其他图片或未知平台标注")
                        if matched is not None and matched.id not in reused:
                            reused.add(matched.id)
                            changed = matched.class_id != category["class_id"] or any(abs(left-right) > 0.0001 for left,right in zip(matched.bbox,box,strict=True))
                            matched.review_status = "corrected" if changed else "approved"
                            matched.bbox, matched.class_id, matched.class_name = box, category["class_id"], category["name"]
                            if changed:
                                matched.annotation_source, matched.confidence, matched.flags = "cvat_human_review", None, []
                        else:
                            additions.append(Annotation(id=new_id(), run_id=run_id, image_id=image_id, class_id=category["class_id"], class_name=category["name"], bbox=box, confidence=None, annotation_source="cvat_human_review", annotation_model_version=current.model_version, review_status="corrected", reviewer=parameters["reviewer"], flags=[]))
                    session.add_all(additions)
                    session.flush()
                    image_boxes = [box for box in original if box.image_id == image_id] + additions
                    review = session.get(ImageReview, (run_id, image_id))
                    review.status = "corrected" if any(box.review_status == "corrected" for box in image_boxes) else "approved"
                    review.reviewer, review.missed_objects_checked = parameters["reviewer"], True
                    review.annotation_digest, review.reviewed_at = annotation_digest(image_boxes), utc_now()
                    session.add(ReviewEvent(run_id=run_id, image_id=image_id, reviewer=parameters["reviewer"], source="cvat_acceptance", before=before_by_image[image_id], after=[AnnotationRead.model_validate(box).model_dump() for box in image_boxes]))
                current.cvat_fingerprint, current.status = remote["fingerprint"], "reviewed"
                result = {"run_id":run_id,"status":"reviewed",**assert_reviewed(session,current)}
        elif kind == "yolo_export":
            result, export_record = _export_dataset(engine, settings, run_id, members, classes, progress=report_progress)
        else:
            raise ValueError("标注任务类型未注册")
        relative = f"reports/{task_id}.json"
        owned_path(settings, relative).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        with Session(engine) as session, session.begin():
            if export_record is not None:
                current = session.get(AnnotationRun, run_id)
                assert_reviewed(session, current)
                session.add(export_record)
                current.status = "exported"
            task = session.get(DatasetTask, task_id)
            require_running(task)
            task.status, task.result, task.report_path = "completed", result, relative
            task.finished_at, task.processed = utc_now(), len(members)
        logger.info("标注任务完成", extra={"task_id":task_id,"result":result})
        from yolo_agent.services.garbage import collect_task_garbage
        collect_task_garbage(engine, settings, task_id)
        return result
    except Exception as error:
        on_annotation_failure(job, None, type(error), error, None)
        logger.exception("标注任务失败，不自动无限重试", extra={"task_id":task_id})
        raise
    finally:
        engine.dispose()


def _export_dataset(engine, settings, run_id: str, members: list[dict], classes: list[dict], *, progress=None) -> tuple[dict, DatasetExport]:
    import datumaro as dm

    def report(stage, completed_images=0, **details):
        if progress:
            progress(stage, completed_images, **details)

    report("export_verifying_review")
    with Session(engine) as session:
        run = session.get(AnnotationRun, run_id)
        summary = assert_reviewed(session, run)
        boxes = list(session.scalars(select(Annotation).where(Annotation.run_id == run_id, Annotation.review_status.in_(["approved", "corrected"]))))
        version_id, cvat_id, expected_cvat = run.version_id, run.cvat_task_id, run.cvat_fingerprint
    if cvat_id:
        from yolo_agent.services.cvat import download_cvat

        actual = download_cvat(settings, cvat_id, require_accepted=True)
        if actual["fingerprint"] != expected_cvat:
            raise DatasetError(409, "CVAT 标注在审核同步后发生变化，请重新审核并同步")
    export_id = new_id()
    relative = f"exports/{export_id}"
    target = owned_path(settings, relative)
    report("export_preparing_images")
    # 复用普通数据导出的策略：正向原图直接交给 Datumaro，旋转图片才生成 PNG。
    # 不重编码大型 JPEG/MPO，不再为封存重复解码和复制一整套无损审核缓存。
    by_id = {}
    for index, member in enumerate(members, start=1):
        by_id[member["id"]] = (_verify_image(settings, member) if member.get("exif_orientation", 1) in {None, 1}
                               else _prepare_frames(settings, run_id, [member])[0])
        report("export_preparing_images", index)
    grouped = defaultdict(list)
    for box in boxes:
        x1,y1,x2,y2 = box.bbox
        grouped[box.image_id].append(dm.Bbox(x1,y1,x2-x1,y2-y1,label=box.class_id))
    dataset = dm.Dataset.from_iterable([dm.DatasetItem(id=member["id"], subset=member["split"], media=dm.Image.from_file(str(by_id[member["id"]])), annotations=grouped[member["id"]]) for member in members], categories=[category["name"] for category in classes])
    report("export_writing_dataset", len(members), export_id=export_id)
    dataset.export(str(target), "yolo_ultralytics", save_media=True)
    file_hashes = {}
    files = sorted(path for path in target.rglob("*") if path.is_file())
    report("export_hashing", len(members), completed_files=0, total_files=len(files), export_id=export_id)
    for index, path in enumerate(files, start=1):
        with path.open("rb") as stream:
            file_hashes[path.relative_to(target).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
        if index % 10 == 0 or index == len(files):
            report("export_hashing", len(members), completed_files=index, total_files=len(files), export_id=export_id)
    summary.update(image_count=len(members), annotation_count=len(boxes), split_counts=dict(Counter(member["split"] for member in members)), ready_for_training=True)
    manifest = {"export_id":export_id,"version_id":version_id,"run_id":run_id,"classes":classes,"summary":summary,"sha256":file_hashes,"reviewed_at":utc_now()}
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)+"\n")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    # 训练只需要目录和冻结清单；ZIP 复用已有数据集导出任务，在用户下载时单独生成。
    report("export_ready", len(members), export_id=export_id)
    record = DatasetExport(id=export_id, version_id=version_id, run_id=run_id, directory=relative, manifest_sha256=manifest_hash, archive_path="", archive_sha256="", summary=summary)
    return {"run_id":run_id,"export_id":export_id,"summary":summary,"archive_sha256":"","archive_on_demand":True}, record
