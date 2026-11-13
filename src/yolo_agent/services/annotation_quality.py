"""从不可变人工审核事件追溯候选来源，避免修正后来源字段覆盖造成统计失真。"""

from sqlalchemy import select

from yolo_agent.models.annotation import Annotation, ClassSchema, ReviewEvent
from yolo_agent.models.dataset import VersionImage
from yolo_agent.schemas.annotation import AnnotationRead
from yolo_agent.services.annotators import cvat_eligible

AUTOMATIC_SOURCES = {"grounding_dino", "yolo26_coco", "project_model"}


def source_quality(session, run, boxes: list[Annotation]) -> dict:
    """修改/删除归属于原始来源及原类别，补画归于该图该类实际使用的标注器。"""
    events = list(session.scalars(select(ReviewEvent).where(ReviewEvent.run_id == run.id).order_by(ReviewEvent.created_at, ReviewEvent.id)))
    originals = {}
    for event in events:
        for box in event.before:
            if box.get("annotation_source") in AUTOMATIC_SOURCES:
                originals.setdefault(box["id"], box)
    for box in boxes:
        if box.annotation_source in AUTOMATIC_SOURCES:
            originals.setdefault(box.id, AnnotationRead.model_validate(box).model_dump())
    current = {box.id: box for box in boxes}
    groups = {}

    def row(source, name):
        return groups.setdefault((source, name), {"source": source, "class_name": name, "proposed": 0, "retained": 0, "corrected": 0, "deleted": 0, "added": 0, "pending": 0, "filtered": 0})

    revision = run.parameters.get("_annotation_revision", 1)
    for identity, original in originals.items():
        counts = row(original["annotation_source"], original["class_name"])
        counts["proposed"] += 1
        latest = current.get(identity)
        if revision >= 2 and not cvat_eligible({**original, "review_status": "pending"}):
            counts["filtered"] += 1
        elif latest is None or latest.review_status == "rejected":
            counts["deleted"] += 1
        else:
            counts[{"approved": "retained", "corrected": "corrected", "pending": "pending"}.get(latest.review_status, "pending")] += 1

    initial_schema = session.get(ClassSchema, run.parameters.get("class_schema_id", run.class_schema_id))
    original_ids = {category["class_id"] for category in initial_schema.classes} if initial_schema else set()
    splits = dict(session.execute(select(VersionImage.image_id, VersionImage.split).where(VersionImage.version_id == run.version_id)).all())
    choices = {int(key): value for key, value in run.parameters.get("class_sources", {}).items()}
    for box in boxes:
        if box.id in originals or box.review_status not in {"approved", "corrected"}:
            continue
        if box.annotation_source == "external_reviewed_yolo":
            row(box.annotation_source, box.class_name)["retained"] += 1
            continue
        source = choices.get(box.class_id, {}).get("source", "grounding_dino") if box.class_id in original_ids else "human_added_class"
        if source == "project_model" and splits.get(box.image_id) == "test":
            source = "grounding_dino"
        row(source, box.class_name)["added"] += 1
    return {"source_class_stats": [groups[key] for key in sorted(groups)], "source_quality_has_review_events": bool(events), "source_quality_reason": "修改和删除按原来源及原类别计数；补画按该图该类实际标注器归属。过滤框不算人工删除；删除后重画会分别计数，不能当作独立准确率。"}
