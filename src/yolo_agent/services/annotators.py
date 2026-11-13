"""按操作者确认的类别来源调用成熟检测器，所有自动结果仍需人工审核。"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import AnnotationRun, ClassSchema, DatasetExport
from yolo_agent.models.dataset import DatasetVersion
from yolo_agent.models.experiment import Experiment, ModelArtifact, TrainingTask
from yolo_agent.schemas.annotation import AnnotationSource, COCO_CLASSES, validate_coco_mapping
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.services.training import validate_export

MODEL_REPO = "Ultralytics/YOLO26"
MODEL_REVISION = "070ac3c51435984d992ba92eb3f234daf4e4500d"
DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
CVAT_EXCLUDED_FLAGS = frozenset({"duplicate_box", "abnormal_area"})


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def coco_provenance(settings, name: str) -> dict:
    if name not in {"yolo26n.pt", "yolo26s.pt"}:
        raise DatasetError(422, "COCO 标注只支持 YOLO26n 和备选 YOLO26s")
    path = settings.training_model_dir / name
    sidecar = path.with_suffix(".provenance.json")
    if not path.is_file() or not sidecar.is_file():
        raise DatasetError(503, f"本地 {name} 尚未准备，请使用固定模型准备脚本")
    entry = json.loads(sidecar.read_text())
    if entry.get("repo_id") != MODEL_REPO or entry.get("revision") != MODEL_REVISION or entry.get("model") != name or file_sha256(path) != entry.get("sha256"):
        raise DatasetError(409, f"{name} 来源或权重指纹不符")
    return entry


def project_model_entry(session: Session, settings, project_id: str, model_id: str) -> dict:
    model = session.get(ModelArtifact, model_id)
    experiment = session.get(Experiment, model.experiment_id) if model else None
    if not model or not experiment or experiment.project_id != project_id:
        raise DatasetError(422, "项目标注模型必须属于当前项目")
    training = session.get(TrainingTask, model.task_id)
    if model.kind != "best" or experiment.status != "completed" or not training or training.status != "completed" or model.status not in {"candidate", "published", "retired"}:
        raise DatasetError(422, "只能选择已完成实验的有效 best 权重")
    export = session.get(DatasetExport, experiment.dataset_export_id)
    version = session.get(DatasetVersion, export.version_id) if export else None
    annotation = session.get(AnnotationRun, export.run_id) if export else None
    schema = session.get(ClassSchema, annotation.class_schema_id) if annotation else None
    if not export or not version or not version.sealed or not schema or annotation.status != "exported":
        raise DatasetError(409, "项目模型缺少封存的已审训练数据来源")
    manifest = validate_export(settings, export, verify_files=False)
    root = owned_path(settings, export.directory)
    counts = Counter()
    for label in (root / "labels" / "train").glob("*.txt"):
        expected = manifest["sha256"].get(label.relative_to(root).as_posix())
        if not expected or file_sha256(label) != expected:
            raise DatasetError(409, "项目模型训练标签指纹不符")
        counts.update(int(line.split()[0]) for line in label.read_text().splitlines() if line.strip())
    path = owned_path(settings, model.path)
    if not path.is_file() or file_sha256(path) != model.sha256:
        raise DatasetError(409, "项目模型权重缺失或指纹不符")
    return {"model_id": model.id, "experiment_id": experiment.id, "base_model": experiment.parameters["model"], "path": model.path, "sha256": model.sha256, "classes": schema.classes, "covered_classes": [category["name"] for category in schema.classes if counts[category["class_id"]] > 0]}


def validate_source_plan(session, settings, project_id: str, classes: list[dict], sources: dict, *, coco_model="yolo26n.pt", has_test=True) -> dict:
    choices = {int(key): AnnotationSource.model_validate(value) for key, value in sources.items()}
    validate_coco_mapping(choices)
    if set(choices) - {category["class_id"] for category in classes}:
        raise DatasetError(422, "候选来源引用了类别版本之外的 class_id")
    manifest = {}
    for category in classes:
        choice = choices.get(category["class_id"], AnnotationSource())
        if choice.source == "yolo26_coco":
            if "yolo26_coco" not in manifest:
                manifest["yolo26_coco"] = coco_provenance(settings, coco_model)
        elif choice.source == "project_model":
            key = f"project_model:{choice.model_id}"
            if key not in manifest:
                manifest[key] = project_model_entry(session, settings, project_id, choice.model_id)
            if category["name"] not in manifest[key]["covered_classes"]:
                raise DatasetError(422, f"项目模型没有该类别的已审训练正样本：{category['name']}")
        if choice.source == "grounding_dino" or (choice.source == "project_model" and has_test):
            path = settings.annotation_model_path / "provenance.json"
            if not path.is_file():
                raise DatasetError(503, "DINO 兜底模型尚未准备")
            entry = json.loads(path.read_text())
            if entry.get("model_id") != DINO_MODEL or entry.get("revision") != DINO_REVISION:
                raise DatasetError(409, "DINO 来源不在固定版本白名单内")
            manifest["grounding_dino"] = {"model_id": DINO_MODEL, "revision": DINO_REVISION}
    return manifest


def cvat_eligible(box: dict) -> bool:
    return box.get("review_status") != "rejected" and not CVAT_EXCLUDED_FLAGS.intersection(box.get("flags", []))


def finalize_candidates(records: list[dict], policy: dict) -> list[dict]:
    """复用 torchvision 跨类别 NMS；被抑制框保留审计标记，但不会发送 CVAT。"""
    if len(records) > policy["max_boxes_per_image"]:
        raise ValueError("候选框超过上限，请提高阈值或减少类别后新建批次")
    if records:
        import torch
        from torchvision.ops import nms

        # 合并多个标注器后必须重新判断，不能沿用单个标注器阶段的抑制结果。
        for box in records:
            box["flags"] = [flag for flag in box["flags"] if flag != "duplicate_box"]
        keep = set(nms(torch.tensor([box["bbox"] for box in records], dtype=torch.float32), torch.tensor([box["confidence"] for box in records], dtype=torch.float32), policy["duplicate_iou_threshold"]).tolist())
        for index, box in enumerate(records):
            if index not in keep and "duplicate_box" not in box["flags"]:
                box["flags"].append("duplicate_box")
    return records


def prediction_record(box, score: float, category: dict, width: int, height: int, policy: dict) -> dict | None:
    """坐标裁剪复用 torchvision，保留可审核的风险信息。"""
    import torch
    from torchvision.ops import clip_boxes_to_image

    original = torch.as_tensor(box, dtype=torch.float32)
    clipped = clip_boxes_to_image(original, (height, width))
    if not torch.isfinite(clipped).all() or not 0 <= score <= 1:
        return None
    x1, y1, x2, y2 = clipped.tolist()
    if x2 <= x1 or y2 <= y1:
        return None
    flags = []
    if not torch.equal(original, clipped):
        flags.append("clipped_to_image")
    if score < policy["low_confidence_threshold"]:
        flags.append("low_confidence")
    if not 0.00001 <= (x2-x1)*(y2-y1)/(width*height) <= 0.95:
        flags.append("abnormal_area")
    return {"class_id": category["class_id"], "class_name": category["name"], "bbox": [x1,y1,x2,y2], "confidence": score, "flags": flags}


class YoloAnnotator:
    """固定 COCO 权重推理，类别映射由操作者确认。"""

    def __init__(self, settings, model_name="yolo26n.pt"):
        from yolo_agent.services.ultralytics_runtime import configure_ultralytics
        from yolo_agent.services.runtime import collect_runtime, require_device

        self.provenance = coco_provenance(settings, model_name)
        self.version = f"{model_name}@{self.provenance['sha256']}"
        self.source = "yolo26_coco"
        self.device = require_device(collect_runtime(settings.device))
        configure_ultralytics(settings)
        from ultralytics import YOLO

        self.model = YOLO(str(settings.training_model_dir / model_name))
        if [self.model.names[index] for index in range(len(self.model.names))] != list(COCO_CLASSES):
            raise DatasetError(409, "COCO 权重类别顺序不匹配，拒绝映射")

    def detect(self, path: Path, classes: list[dict], policy: dict, choices: dict) -> list[dict]:
        mapping = {COCO_CLASSES.index(choices[category["class_id"]].coco_class): category for category in classes}
        return self._predict(path, mapping, policy)

    def _predict(self, path, mapping, policy):
        from PIL import Image, ImageOps

        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        result = self.model.predict(image, classes=list(mapping), conf=policy["box_threshold"], iou=policy["duplicate_iou_threshold"], max_det=policy["max_boxes_per_image"], imgsz=640, device=self.device, verbose=False, save=False)[0]
        records = []
        for bbox, score, label in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist(), strict=True):
            category = mapping.get(int(label))
            record = prediction_record(bbox, score, category, image.width, image.height, policy) if category else None
            if record:
                records.append(record)
        return records


class ProjectModelAnnotator(YoloAnnotator):
    """只加载服务端校验过的本项目 best 权重，再次核对实际模型名称。"""

    def __init__(self, settings, entry: dict):
        from yolo_agent.services.ultralytics_runtime import configure_ultralytics
        from yolo_agent.services.runtime import collect_runtime, require_device

        path = owned_path(settings, entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise DatasetError(409, "项目标注权重指纹变化")
        self.version, self.source = f"{entry['model_id']}@{entry['sha256']}", "project_model"
        self.device = require_device(collect_runtime(settings.device))
        configure_ultralytics(settings)
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        expected = {category["class_id"]: category["name"] for category in entry["classes"]}
        if self.model.names != expected:
            raise DatasetError(409, "项目权重实际类别与注册来源不一致")

    def detect(self, path, classes, policy, choices):
        labels = {name: index for index, name in self.model.names.items()}
        return self._predict(path, {labels[category["name"]]: category for category in classes}, policy)


class AnnotationDispatcher:
    """同一批次按类别分组、按模型复用；test 图片从不传给项目模型。"""

    def __init__(self, settings, classes, policy, manifest):
        self.settings, self.classes, self.policy, self.manifest = settings, classes, policy, manifest
        self.choices = {int(key): AnnotationSource.model_validate(value) for key, value in policy.get("class_sources", {}).items()}
        self.detectors = {}
        self.routing_counts = Counter()

    def detect(self, path, split):
        groups = defaultdict(list)
        for category in self.classes:
            choice = self.choices.get(category["class_id"], AnnotationSource())
            source = "grounding_dino" if split == "test" and choice.source == "project_model" else choice.source
            self.routing_counts[(split, source, category["class_id"])] += 1
            key = f"project_model:{choice.model_id}" if source == "project_model" else source
            groups[key].append(category)
        records = []
        for key, categories in groups.items():
            if key not in self.detectors:
                if key == "grounding_dino":
                    from yolo_agent.services.grounding import GroundingDetector
                    self.detectors[key] = GroundingDetector(self.settings)
                elif key == "yolo26_coco":
                    self.detectors[key] = YoloAnnotator(self.settings, self.policy.get("coco_model", "yolo26n.pt"))
                else:
                    self.detectors[key] = ProjectModelAnnotator(self.settings, self.manifest[key])
            detector = self.detectors[key]
            detected = detector.detect(path, categories, self.policy) if key == "grounding_dino" else detector.detect(path, categories, self.policy, self.choices)
            for box in detected:
                box.update(annotation_source="grounding_dino" if key == "grounding_dino" else detector.source, annotation_model_version=f"{DINO_MODEL}@{DINO_REVISION}:combined-v2" if key == "grounding_dino" else detector.version)
            records.extend(detected)
        return finalize_candidates(records, self.policy)
