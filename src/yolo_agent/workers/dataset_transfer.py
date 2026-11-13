"""复用 Datumaro 完成 YOLO 检测数据导入和可追溯快照导出。"""

import hashlib
import math
import os
import shutil
import stat
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import datumaro as dm
import numpy as np
import yaml
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import Annotation, AnnotationRun, ClassSchema, ImageReview, ReviewEvent
from yolo_agent.models.dataset import Dataset, DatasetImage, DatasetTask, DatasetVersion, VersionImage, new_id, utc_now
from yolo_agent.schemas.dataset import ImageRead
from yolo_agent.services.annotations import annotation_digest
from yolo_agent.services.dataset_transfer import ignored_file, upload_name
from yolo_agent.services.datasets import DatasetService, owned_path
from yolo_agent.services.inspector import FORMATS, analyze_images, inspect_image
from yolo_agent.workers.dataset_tasks import _write_json


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_yolo_source(settings, task_id: str, parameters: dict) -> tuple[Path, dict, list]:
    """只接受包内的标准检测布局；忽略来源机器 data.yaml 的绝对 path。"""
    source = owned_path(settings, parameters["source"])
    if parameters["archive"]:
        extracted = source.parent / "extracted"
        extracted.mkdir()
        seen = set()
        with ZipFile(source / "dataset.zip") as archive:
            if len(archive.infolist()) > settings.max_dataset_images * 3 + 500:
                raise ValueError("ZIP 内文件数超过限制")
            for entry in archive.infolist():
                path = upload_name(entry.filename.rstrip("/") if entry.is_dir() else entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                    raise ValueError("ZIP 不接受符号链接或加密文件")
                if entry.is_dir() or ignored_file(path):
                    continue
                if path in seen:
                    raise ValueError("ZIP 包含重复文件名")
                seen.add(path)
                destination = extracted / path.as_posix()
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as incoming, destination.open("xb") as output:
                    shutil.copyfileobj(incoming, output, length=1024 * 1024)
        source = extracted
    configs = [path for path in source.rglob("*") if path.name in {"data.yaml", "data.yml"} and not ignored_file(path.relative_to(source))]
    if len(configs) != 1:
        raise ValueError("数据包必须包含唯一的 data.yaml（可带一层外部目录）")
    config = configs[0]
    content = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
    if not isinstance(content, dict):
        raise ValueError("data.yaml 必须是 YOLO 配置对象")
    names = content.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if (not isinstance(names, dict) or not 1 <= len(names) <= 1000
            or any(type(key) is not int for key in names)
            or set(names) != set(range(len(names)))
            or any(not isinstance(v, str) or not v.strip() or len(v) > 100 for v in names.values())
            or len(set(names.values())) != len(names)):
        raise ValueError("names 必须是从 0 连续编号的非空、唯一类别名称")
    names = {i: names[i] for i in range(len(names))}
    root, members, used_labels, used_images = config.parent, [], set(), set()
    view = source.parent / "normalized"
    view.mkdir()
    for split in ("train", "val", "test"):
        location = content.get(split)
        if location is None:
            continue
        if not isinstance(location, str):
            raise ValueError(f"{split} 目前仅支持包内图片目录，不支持列表或清单文件")
        folder = root / upload_name(location.removeprefix("./")).as_posix()
        if not folder.is_dir():
            raise ValueError(f"{split} 的图片目录不存在：{location}")
        parts = list(folder.relative_to(root).parts)
        if parts.count("images") != 1:
            raise ValueError("支持 train/images、valid/images 或 images/train 等标准 YOLO 目录")
        parts[parts.index("images")] = "labels"
        labels = root.joinpath(*parts)
        images = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in FORMATS and not ignored_file(p.relative_to(root)))
        for image in images:
            if image in used_images:
                raise ValueError("同一图片被重复配置到不同子集")
            used_images.add(image)
            label = labels / image.relative_to(folder).with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"图片缺少标签；负样本也须提供空 TXT：{image.relative_to(root)}")
            if label in used_labels:
                raise ValueError("同名不同扩展名图片共用标签，请先改为唯一文件名")
            used_labels.add(label)
            for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"仅支持矩形检测标签（每行 5 列）：{label.name}:{line_no}")
                category, x, y, w, h = map(float, fields)
                if (not all(math.isfinite(v) for v in (category, x, y, w, h))
                        or not category.is_integer() or int(category) not in names
                        or not 0 <= x <= 1 or not 0 <= y <= 1 or not 0 < w <= 1 or not 0 < h <= 1):
                    raise ValueError(f"类别或归一化坐标无效：{label.name}:{line_no}")
            stem = f"{len(members):08d}"
            with Image.open(image, formats=list(FORMATS.values())) as media:
                encoding = media.format
                if media.width * media.height > settings.max_image_pixels:
                    raise ValueError(f"图片超过解码像素限制：{image.name}")
            suffix = image.suffix.lower()
            if FORMATS[suffix] != encoding and not (encoding == "MPO" and suffix in {".jpg", ".jpeg"}):
                suffix = next(ext for ext, fmt in FORMATS.items() if fmt == encoding)
            image_target = view / "images" / split / (stem + suffix)
            label_target = view / "labels" / split / (stem + ".txt")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            os.link(image, image_target)
            label_target.write_text(label.read_text(encoding="utf-8-sig"), encoding="utf-8")
            members.append({"stem": stem, "name": image.relative_to(root).as_posix(), "split": split,
                "extension_corrected": suffix != image.suffix.lower(), "mpo_primary_frame": encoding == "MPO"})
    if not members or len(members) > settings.max_dataset_images:
        raise ValueError("数据包为空或图片数超过数据集限制")
    orphan_labels = {p for p in root.rglob("*.txt") if "labels" in p.relative_to(root).parts and not ignored_file(p.relative_to(root))} - used_labels
    all_images = {p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in FORMATS and not ignored_file(p.relative_to(root))}
    if orphan_labels or all_images != used_images:
        raise ValueError("存在未被划分引用的图片或无法匹配图片的标签，请修正后重试")
    splits = {m["split"] for m in members}
    (view / "data.yaml").write_text(yaml.safe_dump({**{s: f"images/{s}" for s in splits}, "names": names}, sort_keys=False), encoding="utf-8")
    return view, names, members


def import_labeled(engine, settings, task_id: str, dataset_id: str, parameters: dict) -> dict:
    view, names, source_members = prepare_yolo_source(settings, task_id, parameters)
    imported = dm.Dataset.import_from(str(view), format="yolo")
    categories = imported.categories()[dm.AnnotationType.label].items
    if [c.name for c in categories] != list(names.values()):
        raise ValueError("Datumaro 解析类别与 data.yaml 不一致")
    items = list(imported)
    if len(items) != len(source_members):
        raise ValueError("Datumaro 解析图片数量与来源不一致")
    sources = {m["stem"]: m for m in source_members}
    reviewed = parameters.get("accepted_reviewed_source", False)
    attribution = {"source": "external_labeled_yolo", "reviewer": parameters.get("reviewer", ""),
        "confirmation": parameters.get("confirmation", ""), "accepted_reviewed_source": reviewed,
        "accepted_at": utc_now(), "new_manual_review_performed": False}
    schema_id, version_id, run_id = new_id(), new_id(), new_id()
    images, members, boxes, reviews, events, clipping = [], [], [], [], [], []
    counts, splits = Counter(), Counter()
    with Session(engine) as session, session.begin():
        task = session.get(DatasetTask, task_id)
        task.total = len(items)
        existing_count = session.scalar(select(func.count()).select_from(DatasetImage).where(DatasetImage.dataset_id == dataset_id))
        if existing_count + len(items) > settings.max_dataset_images:
            raise ValueError("追加导入后数据集图片数超过上限")
        project_id = session.get(Dataset, dataset_id).project_id
        number = (session.scalar(select(func.max(DatasetVersion.number)).where(DatasetVersion.dataset_id == dataset_id)) or 0) + 1
    for index, item in enumerate(items, 1):
        source = sources[Path(item.id).name]
        split = {"valid": "val", "validation": "val"}.get(item.subset, item.subset)
        if split != source["split"]:
            raise ValueError("解析划分与来源不一致")
        path = Path(item.media.path).resolve()
        if not path.is_relative_to(view.resolve()):
            raise ValueError("解析媒体路径越界")
        identity = new_id()
        preview = f"previews/{identity}.jpg"
        info = inspect_image(path, owned_path(settings, preview), settings.max_image_pixels, allow_mpo_primary_frame=True)
        if info["status"] != "valid":
            raise ValueError(f"图片未通过检查：{source['name']} {info.get('error')}")
        if info["exif_orientation"] != 1:
            raise ValueError(f"图片包含 EXIF 旋转，请先校正图片及标签：{source['name']}")
        image = DatasetImage(id=identity, dataset_id=dataset_id, import_task_id=task_id,
            original_name=source["name"], storage_path=str(path.relative_to(settings.storage_dir.resolve())),
            size_bytes=path.stat().st_size, preview_path=preview, created_at=utc_now(), **info)
        images.append(image)
        snapshot = {**ImageRead.model_validate(image).model_dump(), "storage_path": image.storage_path, "split": split}
        members.append(snapshot)
        image_boxes, before = [], []
        for annotation in item.annotations:
            if annotation.type != dm.AnnotationType.bbox:
                raise ValueError("仅支持矩形检测标注")
            original = np.asarray(annotation.points, dtype=np.float64)
            bounded = np.clip(original, [0, 0, 0, 0], [image.width, image.height, image.width, image.height])
            if not np.all(np.isfinite(bounded)) or bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
                raise ValueError(f"标注框无效：{source['name']}")
            clipped = not np.allclose(original, bounded, atol=1e-6, rtol=0)
            if clipped:
                clipping.append({"image": source["name"], "before": original.tolist(), "after": bounded.tolist()})
            box = Annotation(id=new_id(), run_id=run_id, image_id=identity, class_id=annotation.label,
                class_name=names[annotation.label], bbox=bounded.tolist(), confidence=None,
                annotation_source="external_labeled_yolo", annotation_model_version="operator-upload",
                review_status="approved" if reviewed and not clipped else "pending",
                flags=["boundary_clipped_on_import"] if clipped else [], reviewer=attribution["reviewer"] if reviewed else None)
            image_boxes.append(box)
            before.append({"class_id": annotation.label, "bbox": original.tolist()})
            counts[box.class_name] += 1
        accepted = reviewed and all(b.review_status == "approved" for b in image_boxes)
        reviews.append(ImageReview(run_id=run_id, image_id=identity, status="approved" if accepted else "pending",
            missed_objects_checked=accepted, reviewer=attribution["reviewer"] if accepted else None,
            annotation_digest=annotation_digest(image_boxes) if accepted else None, reviewed_at=attribution["accepted_at"] if accepted else None))
        boxes.extend(image_boxes)
        if reviewed:
            events.append(ReviewEvent(id=new_id(), run_id=run_id, image_id=identity, reviewer=attribution["reviewer"],
                source="operator_authorized_labeled_import", before=before,
                after=[{"class_id": b.class_id, "bbox": b.bbox, "status": b.review_status, "authorization": attribution} for b in image_boxes]))
        splits[split] += 1
        with Session(engine) as session, session.begin():
            session.get(DatasetTask, task_id).processed = index
    pending = sum(r.status == "pending" for r in reviews)
    ready = pending == 0
    quality, _ = analyze_images(members, 4, settings.max_similarity_pairs)
    summary = {"image_count": len(members), "annotation_count": len(boxes), "split_counts": {s: splits[s] for s in ("train", "val", "test")},
        "class_distribution": dict(counts), "reviewed_images": len(members) - pending, "unreviewed_images": pending,
        "ready_for_training": ready, "source_split_preserved": True, "clipped_boxes": len(clipping),
        "corrected_extensions": sum(m["extension_corrected"] for m in source_members),
        "mpo_primary_frames": sum(m["mpo_primary_frame"] for m in source_members),
        "warnings": ["保留来源划分，未重新去重；请关注相邻帧跨集造成的数据泄漏"] + (["越界框已裁剪，受影响图片须重新审核"] if clipping else [])}
    quality.update(annotation_status="external_reviewed" if ready else "pending_review", class_distribution=dict(counts),
        class_distribution_reason="来自上传的 YOLO 标签", ready_for_training=ready)
    relative = f"versions/{version_id}/manifest.json"
    checksum = _write_json(owned_path(settings, relative), {"version_id": version_id, "dataset_id": dataset_id, "number": number,
        "images": members, "summary": summary, "quality_report": quality, "review_attribution": attribution, "clipped_boxes": clipping})
    classes = [{"class_id": i, "name": name, "prompt": name} for i, name in names.items()]
    result = {"dataset_id": dataset_id, "version_id": version_id, "annotation_run_id": run_id, "summary": summary}
    report = f"reports/labeled-import-{task_id}.json"
    _write_json(owned_path(settings, report), {**result, "clipped_boxes": clipping, "review_attribution": attribution})
    # 图片、版本、框和审核凭据一次提交；坏标签不会留下半个可用版本。
    with Session(engine) as session, session.begin():
        session.add_all(images)
        session.add(ClassSchema(id=schema_id, project_id=project_id, name=f"上传 YOLO 类别 · {version_id[:8]}", classes=classes))
        session.flush()
        version = DatasetVersion(id=version_id, dataset_id=dataset_id, task_id=task_id, number=number,
            name="上传已标注数据 · 保留原划分", policy={"entry": "external_labeled_yolo", "preserve_source_splits": True},
            summary=summary, manifest_path=relative, manifest_sha256=checksum, sealed=False)
        session.add(version)
        session.flush()
        session.add_all([VersionImage(version_id=version_id, image_id=m["id"], split=m["split"], group_id="external:"+m["id"], snapshot=m) for m in members])
        session.flush()
        version.sealed = True
        session.add(AnnotationRun(id=run_id, version_id=version_id, class_schema_id=schema_id,
            status="reviewed" if ready else "pending_review", parameters={"review_attribution": attribution},
            model_version="external-labeled-yolo", processed=len(members), total=len(members)))
        session.flush()
        session.add_all(boxes + reviews + events)
        task = session.get(DatasetTask, task_id)
        task.result, task.report_path, task.status, task.finished_at = result, report, "completed", utc_now()
    return result


def export_dataset(engine, settings, task_id: str, parameters: dict) -> dict:
    """普通导出保留待审核标注，与已审核训练工件严格分开。"""
    from yolo_agent.workers.annotation_tasks import _prepare_frames, _verify_image

    version_id, run_id = parameters["version_id"], parameters.get("run_id")
    DatasetService(engine, settings).manifest_path(version_id)
    with Session(engine) as session:
        members = list(session.scalars(select(VersionImage.snapshot).where(VersionImage.version_id == version_id).order_by(VersionImage.image_id)))
        classes, boxes, reviews, provenance = [], [], [], None
        if run_id:
            run = session.get(AnnotationRun, run_id)
            if not run or run.version_id != version_id or run.status not in {"pending_review", "reviewed", "exported"}:
                raise ValueError("绑定的标注批次不可导出")
            classes = session.get(ClassSchema, run.class_schema_id).classes
            provenance = run.parameters.get("review_attribution")
            boxes = list(session.scalars(select(Annotation).where(Annotation.run_id == run_id, Annotation.review_status != "rejected").order_by(Annotation.id)))
            reviews = list(session.scalars(select(ImageReview).where(ImageReview.run_id == run_id)))
    if not members:
        raise ValueError("版本没有图片")
    with Session(engine) as session, session.begin():
        session.get(DatasetTask, task_id).total = len(members)
    target = owned_path(settings, f"dataset-downloads/{task_id}")
    # 无旋转的原图直接交给 Datumaro 复制，避免把大型航拍 JPEG 全部膨胀成 PNG。
    frames_by_id = {m["id"]: _verify_image(settings, m) for m in members if m.get("exif_orientation", 1) in {None, 1}}
    rotated = [m for m in members if m.get("exif_orientation", 1) not in {None, 1}]
    frames_by_id.update({p.stem: p for p in _prepare_frames(settings, f"download-{task_id}", rotated)})
    grouped = defaultdict(list)
    for box in boxes:
        x1, y1, x2, y2 = box.bbox
        grouped[box.image_id].append(dm.Bbox(x1, y1, x2-x1, y2-y1, label=box.class_id))
    if run_id:
        dm.Dataset.from_iterable([dm.DatasetItem(id=m["id"], subset=m["split"],
            media=dm.Image.from_file(str(frames_by_id[m["id"]])), annotations=grouped[m["id"]]) for m in members],
            categories=[c["name"] for c in classes]).export(str(target), "yolo_ultralytics", save_media=True)
        # Datumaro 的 path 指向导出目录，下载后需改为可移植的相对路径。
        config = target / "data.yaml"
        content = yaml.safe_load(config.read_text())
        content["path"] = "."
        for split in {m["split"] for m in members}:
            content[split] = f"images/{split}"
            listing = target / f"{split}.txt"
            if listing.is_file():
                listing.unlink()
        config.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        for member in members:
            destination = target / "images" / member["split"] / (member["id"] + frames_by_id[member["id"]].suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(frames_by_id[member["id"]], destination)
    summary = {"image_count": len(members), "annotation_count": len(boxes), "split_counts": dict(Counter(m["split"] for m in members)),
        "annotation_status": "labeled" if run_id else "unlabeled", "training_export": False,
        "warning": "此包用于数据迁移；可训练工件须另经审核后封存。"}
    _write_json(target / "manifest.json", {"version_id": version_id, "run_id": run_id, "classes": classes, "summary": summary,
        "images": [{"id": m["id"], "original_name": m["original_name"], "split": m["split"]} for m in members],
        "annotations": [{"image_id": b.image_id, "class_id": b.class_id, "bbox": b.bbox, "review_status": b.review_status, "flags": b.flags, "reviewer": b.reviewer, "source": b.annotation_source} for b in boxes],
        "reviews": [{"image_id": r.image_id, "status": r.status, "missed_objects_checked": r.missed_objects_checked, "reviewer": r.reviewer, "reviewed_at": r.reviewed_at} for r in reviews],
        "review_attribution": provenance,
        "sha256": {p.relative_to(target).as_posix(): file_digest(p) for p in sorted(target.rglob("*")) if p.is_file()}})
    archive = Path(shutil.make_archive(str(target), "zip", root_dir=target))
    result = {"version_id": version_id, "run_id": run_id, "summary": summary,
        "archive_path": archive.relative_to(settings.storage_dir.resolve()).as_posix(), "archive_sha256": file_digest(archive)}
    report = f"reports/dataset-export-{task_id}.json"
    _write_json(owned_path(settings, report), result)
    with Session(engine) as session, session.begin():
        task = session.get(DatasetTask, task_id)
        task.result, task.report_path, task.status, task.finished_at, task.processed = result, report, "completed", utc_now(), len(members)
    shutil.rmtree(owned_path(settings, f"annotations/download-{task_id}"), ignore_errors=True)
    return result
