"""离线比较标注候选与现有人审真值；不创建正式批次、不访问 test 图片。"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path


def main():
    if not os.getenv("CONDA_PREFIX") or os.getenv("CONDA_DEFAULT_ENV") == "base":
        raise RuntimeError("请使用项目独立 Conda 环境")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/acceptance/annotation-sources"))
    parser.add_argument("--coco-model", choices=("yolo26n.pt", "yolo26s.pt"), default="yolo26n.pt")
    args = parser.parse_args()
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from PIL import Image, ImageOps
    import torch
    from torchvision.ops import batched_nms
    from yolo_agent.models.annotation import Annotation, AnnotationRun, ClassSchema, ImageReview
    from yolo_agent.models.dataset import VersionImage, utc_now
    from yolo_agent.models.database import create_database_engine
    from yolo_agent.schemas.annotation import AutoAnnotateCreate, COCO_CLASSES
    from yolo_agent.services.annotations import annotation_digest
    from yolo_agent.services.annotators import AnnotationDispatcher, YoloAnnotator, cvat_eligible, prediction_record, file_sha256
    from yolo_agent.services.datasets import owned_path
    from yolo_agent.services.grounding import GroundingDetector
    from yolo_agent.settings import get_settings

    settings = get_settings()
    engine = create_database_engine(settings)
    with Session(engine) as session:
        run = session.get(AnnotationRun, args.run_id)
        if not run or run.status not in {"reviewed", "exported"}:
            raise RuntimeError("请提供已人工审核批次")
        classes = session.get(ClassSchema, run.class_schema_id).classes
        members = list(session.execute(select(VersionImage.snapshot, VersionImage.split).where(VersionImage.version_id == run.version_id, VersionImage.split.in_(["train", "val"])).order_by(VersionImage.image_id)))
        samples = []
        for member, split in members:
            boxes = list(session.scalars(select(Annotation).where(Annotation.run_id == run.id, Annotation.image_id == member["id"])))
            review = session.get(ImageReview, (run.id, member["id"]))
            if not review or not review.missed_objects_checked or review.status not in {"approved", "corrected"} or review.annotation_digest != annotation_digest(boxes):
                raise RuntimeError("人审真值缺失或摘要过期")
            path = owned_path(settings, member["storage_path"])
            if file_sha256(path) != member["sha256"]:
                raise RuntimeError("源图指纹变化")
            samples.append({"image_id": member["id"], "path": str(path), "width": member["width"], "height": member["height"], "split": split, "truth": [{"class_id": box.class_id, "class_name": box.class_name, "bbox": box.bbox} for box in boxes if box.review_status in {"approved", "corrected"}]})
        version_id = run.version_id
    args.output.mkdir(parents=True, exist_ok=True)
    policy = AutoAnnotateCreate(version_id=version_id, class_schema_id="evaluation-only", coco_model=args.coco_model).model_dump()
    choices = {category["class_id"]: {"source": "yolo26_coco", "coco_class": category["prompt"].lower()} for category in classes if category["prompt"].lower() in COCO_CLASSES}
    mixed = AnnotationDispatcher(settings, classes, {**policy, "class_sources": choices}, {})
    detector, yolo = GroundingDetector(settings), YoloAnnotator(settings, args.coco_model)
    mixed.detectors.update(grounding_dino=detector, yolo26_coco=yolo)
    from yolo_agent.schemas.annotation import AnnotationSource
    coco_choices = {key: AnnotationSource.model_validate(value) for key, value in choices.items()}
    coco_classes = [category for category in classes if category["class_id"] in choices]
    all_predictions = {}
    durations = Counter()
    for sample in samples:
        path = Path(sample["path"])
        predictions = {}
        started = time.monotonic()
        # 精确保留旧逐类提示、0.25 阈值及仅类内 NMS；仅作为本地离线基线。
        legacy_policy = {**policy, "box_threshold": 0.25}
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        records = []
        for category in classes:
            inputs = detector.processor(images=image, text=category["prompt"].lower()+".", return_tensors="pt").to(detector.device)
            with torch.inference_mode():
                output = detector.model(**inputs)
            result = detector.processor.post_process_grounded_object_detection(output, inputs.input_ids, threshold=0.25, text_threshold=policy["text_threshold"], target_sizes=[(image.height,image.width)])[0]
            for box, score in zip(result["boxes"], result["scores"], strict=True):
                record = prediction_record(box, float(score.item()), category, image.width, image.height, legacy_policy)
                if record:
                    records.append(record)
        if records:
            kept = set(batched_nms(torch.tensor([box["bbox"] for box in records]), torch.tensor([box["confidence"] for box in records]), torch.tensor([box["class_id"] for box in records]), policy["duplicate_iou_threshold"]).tolist())
            for index, box in enumerate(records):
                if index not in kept:
                    box["flags"].append("duplicate_box")
        predictions["old_dino"] = records
        durations["old_dino"] += time.monotonic()-started
        for name, invoke in (
            ("new_dino", lambda: detector.detect(path, classes, policy)),
            ("yolo_coco", lambda: yolo.detect(path, coco_classes, policy, coco_choices)),
            ("mixed", lambda: mixed.detect(path, sample["split"])),
        ):
            started = time.monotonic()
            predictions[name] = invoke()
            durations[name] += time.monotonic()-started
        all_predictions[sample["image_id"]] = predictions
        print(json.dumps({"image_id": sample["image_id"], "raw_boxes": {name: len(rows) for name,rows in predictions.items()}}, ensure_ascii=False), flush=True)
        (args.output / "candidates.json").write_text(json.dumps(all_predictions, ensure_ascii=False, indent=2))

    os.environ["FIFTYONE_DATABASE_DIR"] = str((settings.storage_dir/"fiftyone/db").resolve())
    os.environ["FIFTYONE_CONFIG_PATH"] = str((settings.storage_dir/"fiftyone/config.json").resolve())
    os.environ["FIFTYONE_DO_NOT_TRACK"] = "true"
    import fiftyone as fo
    fo.config.show_progress_bars = False
    results = {}
    names = {category["name"] for category in classes}
    for condition in ("old_dino", "new_dino", "yolo_coco", "mixed"):
        dataset = fo.Dataset()
        raw_count = sent_count = 0
        try:
            for sample in samples:
                predictions = all_predictions[sample["image_id"]][condition]
                eligible = predictions if condition == "old_dino" else [box for box in predictions if cvat_eligible(box)]
                raw_count += len(predictions)
                sent_count += len(eligible)
                def detection(box, predicted=False):
                    x1,y1,x2,y2 = box["bbox"]
                    kwargs = {"label": box["class_name"], "bounding_box": [x1/sample["width"],y1/sample["height"],(x2-x1)/sample["width"],(y2-y1)/sample["height"]]}
                    if predicted:
                        kwargs["confidence"] = box["confidence"]
                    return fo.Detection(**kwargs)
                dataset.add_sample(fo.Sample(filepath=sample["path"], image_id=sample["image_id"], ground_truth=fo.Detections(detections=[detection(box) for box in sample["truth"]]), predictions=fo.Detections(detections=[detection(box,True) for box in eligible])))
            typed = dataset.evaluate_detections("predictions", gt_field="ground_truth", method="coco", eval_key="typed", iou=0.5, classwise=True, compute_mAP=False, progress=False)
            counts = {name: Counter() for name in names}
            for truth, predicted in zip(typed.ytrue, typed.ypred, strict=True):
                if truth in names:
                    counts[truth]["tp" if truth == predicted else "fn"] += 1
                if predicted in names and truth != predicted:
                    counts[predicted]["fp"] += 1
            located = dataset.evaluate_detections("predictions", gt_field="ground_truth", method="coco", eval_key="located", iou=0.5, classwise=False, compute_mAP=False, progress=False)
            pairs = [(truth,predicted) for truth,predicted in zip(located.ytrue,located.ypred,strict=True) if truth in names and predicted in names]
            per_class = {}
            for name,row in counts.items():
                tp,fp,fn = (row[key] for key in ("tp","fp","fn"))
                per_class[name] = {"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None}
            totals = {key: sum(row[key] for row in per_class.values()) for key in ("tp","fp","fn")}
            results[condition] = {"per_class":per_class,"totals":totals,"raw_boxes":raw_count,"cvat_boxes":sent_count,"matched_locations":len(pairs),"wrong_class_locations":sum(truth!=predicted for truth,predicted in pairs),"conditional_class_error_rate":sum(truth!=predicted for truth,predicted in pairs)/len(pairs) if pairs else None,"inference_seconds":durations[condition]}
        finally:
            dataset.delete()
    result = {"collected_at":utc_now(),"annotation_run_id":args.run_id,"version_id":version_id,"image_count":len(samples),"splits":["train","val"],"test_used":False,"ground_truth":samples,"coco_model":args.coco_model,"coco_provenance":yolo.provenance,"policy":policy,"mixed_sources":choices,"matching":"FiftyOne COCO IoU=0.5；类别错误另用 classwise=False 匹配位置","results":results,"limitations":["仅公开照片已审小样本，不能证明业务原图效果","没有读取 test 图片或真值，专业类别若无 train/val 真值则未验收","旧基线包含旧推送策略保留的全部候选，新方案统计按实际 CVAT 过滤策略","未用项目模型优化真值或重复人工审核"]}
    (args.output/"comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({name:{"totals":row["totals"],"class_error_rate":row["conditional_class_error_rate"]} for name,row in results.items()},ensure_ascii=False),flush=True)
    engine.dispose()


if __name__ == "__main__":
    main()
