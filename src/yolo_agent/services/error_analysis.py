"""复用 FiftyOne COCO 匹配定位错误，不另写目标匹配算法。"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from yolo_agent.settings import Settings


def analyze_errors(settings: Settings, task_id: str, root: Path, output: Path, classes: dict[int, str], predictions: list[dict], parameters: dict) -> dict:
    """在固定验证集上匹配预测与人审标签，保存逐图错误及小目标统计。

    :param settings: FiftyOne 私有存储位置。
    :param task_id: 评估任务，用于隔离临时数据集。
    :param root: 指纹已验证的 YOLO 工件目录。
    :param output: 当前评估工件目录。
    :param classes: 已审核类别 ID 到名称。
    :param predictions: Ultralytics 验证器导出的真实像素预测。
    :param parameters: 操作置信度、匹配 IoU 与原图小目标面积阈值。
    :returns: 错误计数、预览及有样本支持的小目标召回统计。
    """
    os.environ["FIFTYONE_DATABASE_DIR"] = str((settings.storage_dir / "fiftyone" / "db").resolve())
    os.environ["FIFTYONE_CONFIG_PATH"] = str((settings.storage_dir / "fiftyone" / "config.json").resolve())
    os.environ["FIFTYONE_DO_NOT_TRACK"] = "true"
    from ultralytics.utils.ops import xywh2ltwh
    import fiftyone as fo

    # 后台任务使用结构化日志，关闭交互进度条对日志流的重定向。
    fo.config.show_progress_bars = False

    by_image = defaultdict(list)
    for prediction in predictions:
        if prediction["score"] >= parameters["error_confidence"]:
            by_image[prediction["file_name"]].append(prediction)
    dataset = fo.Dataset(name=f"yolo26-eval-{task_id}")
    totals, small = Counter(), Counter()
    error_preview = []
    try:
        # 只从 val.txt 枚举，不加载 test 子集或根据测试结果选择阈值。
        for line in (root / "val.txt").read_text().splitlines():
            path = (root / line).resolve()
            if not path.is_relative_to(root / "images" / "val"):
                raise ValueError("验证图片路径越界")
            with Image.open(path) as image:
                width, height = image.size
            label_path = root / "labels" / "val" / f"{path.stem}.txt"
            labels = np.loadtxt(label_path, ndmin=2) if label_path.read_text().strip() else np.empty((0, 5))
            truth = []
            for row in labels:
                box = xywh2ltwh(row[1:].copy()).tolist()
                truth.append(fo.Detection(label=classes[int(row[0])], bounding_box=box, area_pixels=float(box[2]*box[3]*width*height)))
            detected = [fo.Detection(label=classes[item["class_id"]], bounding_box=[item["bbox"][0]/width,item["bbox"][1]/height,item["bbox"][2]/width,item["bbox"][3]/height], confidence=item["score"]) for item in by_image[path.name]]
            dataset.add_sample(fo.Sample(filepath=str(path), image_id=path.stem, ground_truth=fo.Detections(detections=truth), predictions=fo.Detections(detections=detected)))
        dataset.evaluate_detections("predictions", gt_field="ground_truth", method="coco", eval_key="diagnostic", iou=parameters["error_iou"], classwise=True, compute_mAP=False, progress=False)
        with (output / "errors.jsonl").open("w", encoding="utf-8") as stream:
            for sample in dataset:
                gt, pred = [], []
                for detection in sample.ground_truth.detections:
                    state = detection["diagnostic"]
                    totals[state] += 1
                    if detection["area_pixels"] < parameters["small_area_pixels"]:
                        small["total"] += 1
                        small[state] += 1
                    gt.append({"label":detection.label,"bbox":detection.bounding_box,"status":state,"iou":detection.to_dict().get("diagnostic_iou"),"area_pixels":detection["area_pixels"]})
                for detection in sample.predictions.detections:
                    state = detection["diagnostic"]
                    if state == "fp":
                        totals["fp"] += 1
                    pred.append({"label":detection.label,"bbox":detection.bounding_box,"confidence":detection.confidence,"status":state,"iou":detection.to_dict().get("diagnostic_iou")})
                counts = {key:sum(item["status"] == key for item in (gt if key == "fn" else pred)) for key in ("tp","fp","fn")}
                record = {"image_id":sample.image_id,"counts":counts,"ground_truth":gt,"predictions":pred}
                if counts["fp"] or counts["fn"]:
                    totals["error_images"] += 1
                    stream.write(json.dumps(record, ensure_ascii=False)+"\n")
                    if len(error_preview) < 100:
                        error_preview.append({"image_id":sample.image_id,"counts":counts})
        dataset.export(export_dir=str(output / "fiftyone"), dataset_type=fo.types.FiftyOneDataset, export_media=False, progress=False)
        return {"backend":"FiftyOne COCO classwise matching","confidence":parameters["error_confidence"],"iou":parameters["error_iou"],"tp":totals["tp"],"fp":totals["fp"],"fn":totals["fn"],"error_image_count":totals["error_images"],"error_preview":error_preview,"small_objects":{"definition":"原图面积小于阈值，不等同于 COCO 归一化评估面积", "area_pixels_threshold":parameters["small_area_pixels"],"total":small["total"],"tp":small["tp"],"fn":small["fn"],"recall":small["tp"]/small["total"] if small["total"] else None,"unavailable_reason":None if small["total"] else "验证集没有符合定义的小目标，无法评估召回"}}
    finally:
        # 大型图像仍留在封存工件，临时数据库只用于成熟评估库的匹配与导出。
        dataset.delete()
