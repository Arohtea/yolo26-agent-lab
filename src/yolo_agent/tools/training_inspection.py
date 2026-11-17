"""复用封存数据、评估报告和现有校验器，为技能提供可落地的证据查询。"""

import hashlib
import heapq
import json
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from yolo_agent.agents.experiment_design import comparison_design
from yolo_agent.models.annotation import DatasetExport
from yolo_agent.schemas.experiment import TrainingParameters
from yolo_agent.services.dataset_evidence import split_image_paths, verified_export_file, verified_split_image
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.services.training import validate_export

METRICS = ("map50_95", "map50", "precision", "recall")


def _page_classes(rows, names, offset, limit):
    available = {row["name"] for row in rows}
    if set(names) - available:
        raise ValueError(f"未知类别：{sorted(set(names) - available)}；请使用数据集实际类名")
    selected = [row for row in rows if not names or row["name"] in names]
    return {"total": len(selected), "offset": offset, "has_more": offset + limit < len(selected), "items": selected[offset:offset + limit]}


def _summary(report):
    source = report["summary"]
    errors = source.get("errors") or {}
    return {**{name: source.get(name) for name in METRICS},
        "validation_images": source.get("validation_images"), "validation_instances": source.get("validation_instances"),
        "errors": {name: errors.get(name) for name in ("confidence", "iou", "tp", "fp", "fn", "error_image_count", "small_objects")},
        "issues": source.get("issues", []), "recommendations": source.get("recommendations", []),
        "metric_unavailable_reason": source.get("metric_unavailable_reason")}


class TrainingInspection:
    def __init__(self, tools):
        self.tools = tools
        self.settings = tools.settings

    def _evaluation(self, experiment_id):
        item, _ = self.tools._own(experiment_id)
        completed = [task for task in self.tools._evaluations(experiment_id) if task.status == "completed"]
        task = next((task for task in completed if task.id == item.evaluation_id), completed[-1] if completed else None)
        if task is None:
            raise DatasetError(409, "尚无完成且未失效的固定 val 评估，请等待评估或先查看训练曲线")
        return task, self.tools.report(task.id)

    def _export(self, export_id):
        allowed = {row["dataset_export_id"] for row in self.tools.allowed_exports()}
        if export_id not in allowed:
            raise PermissionError("只能读取本运行允许的同类别已审工件")
        with Session(self.tools.engine) as session:
            export = session.get(DatasetExport, export_id)
            manifest = validate_export(self.settings, export, verify_files=False)
            session.expunge(export)
        return export, manifest

    def _images(self, result, paths, requested):
        supported = not self.tools.service.get(self.tools.run_id).state.get("image_inputs_rejected", False)
        result["images_attached"] = bool(paths and requested and supported)
        result["image_note"] = "图片仅在下一次决策可见；框坐标为归一化左上角宽高，数值以结构化记录为准。" if supported else "当前模型不接受图片，已提供结构化标注和错误事实。"
        if result["images_attached"]:
            result["_images"] = [path.relative_to(self.settings.storage_dir.resolve()).as_posix() for path in paths]
        return result

    def parameters(self, args):
        run = self.tools.service.get(self.tools.run_id)
        pairs = self.tools.iterations()
        experiment = self.tools._own(args.experiment_id)[1] if args.experiment_id else (pairs[-1][1] if pairs else None)
        baseline = experiment.parameters if experiment else run.initial_parameters
        base = TrainingParameters.model_json_schema()["properties"]
        constrained = self.tools._parameter_model().model_json_schema()["properties"]
        pinned = set(run.state.get("operator_fields", [])) if not pairs else set()
        rows = {}
        for name in args.names:
            if name not in base:
                rows[name] = {"supported": False, "reason": "该参数未向本平台 Agent 开放；不能因技能提及而直接传入"}
            else:
                rows[name] = {"supported": True, "adjustable": name in constrained, "current_value": baseline.get(name),
                    "value_source": "experiment" if experiment else "operator_first_round" if name in pinned else "platform_default",
                    "required_first_round": name in pinned, "schema": constrained.get(name, base[name])}
        installed = None
        for package in ("ultralytics-opencv-headless", "ultralytics"):
            try:
                installed = {"distribution": package, "version": version(package)}
                break
            except PackageNotFoundError:
                pass
        return {"installed_training_library": installed, "parameters": rows, "supported_names": list(constrained),
            "allowed_models": self.tools.prepared_models(), "model_locked": len(run.policy["allowed_models"]) == 1,
            "note": "Schema 是本运行实际接受的范围；平台默认值不是用户指定，实际优化器、批量和显存须以训练器记录为准。"}

    def experiment(self, args):
        item, experiment = self.tools._own(args.experiment_id)
        result = {"experiment_id": experiment.id, "section": args.section, "status": experiment.status}
        if args.section == "overview":
            return {**result, **self.tools.experiment_brief(item, experiment, full=False),
                "parent_experiment_id": experiment.parent_experiment_id,
                "conclusion": {key: (item.conclusion or {}).get(key) for key in ("map50_95", "delta_previous", "delta_parent", "causal_limit")},
                "charts": self.tools.available_charts(experiment.id), "sections": ["parameters", "curve", "evaluation", "training"]}
        if args.section == "parameters":
            names = args.parameter_names or ["model", "epochs", "imgsz", "batch", "optimizer", "lr0", "mosaic", "tile"]
            unknown = set(names) - set(TrainingParameters.model_fields)
            if unknown:
                raise ValueError(f"未知参数：{sorted(unknown)}")
            resolved = experiment.resolved_parameters or {}
            actual = resolved.get("trainer_parameters") or {}
            return {**result, "requested": {name: experiment.parameters.get(name) for name in names},
                "actual_trainer_parameters": {name: actual.get(name) for name in names},
                "optimizer_effective": resolved.get("optimizer_effective"),
                "augmentation_effective_at_start": resolved.get("augmentation_effective_at_start"),
                "available_names": list(TrainingParameters.model_fields)}
        if args.section == "curve":
            table = self.tools.epoch_table(experiment.id, limit=1000)
            available = table["columns"][1:]
            names = args.metric_names or [name for name in ("metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)", "train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss") if name in available]
            if set(names) - set(available):
                raise ValueError(f"指标尚无记录：{sorted(set(names) - set(available))}；可用指标：{available}")
            indices = [0, *[table["columns"].index(name) for name in names]] if table["columns"] else []
            rows = table["rows"][args.offset:args.offset + args.limit]
            return {**result, "columns": ["epoch", *names], "rows": [[row[index] for index in indices] for row in rows],
                "total_epochs": table["epochs"], "offset": args.offset, "has_more": args.offset + args.limit < table["epochs"],
                "signals": table.get("signals", {}), "downsampled": False}
        if args.section == "training":
            tasks = self.tools._tasks(experiment.id)
            return {**result, "total": len(tasks), "offset": args.offset, "has_more": args.offset + args.limit < len(tasks),
                "attempts": [{"task_id": task.id, "status": task.status, "epoch": f"{task.current_epoch}/{task.total_epochs}",
                    "elapsed_seconds": task.elapsed_seconds, "device": task.device, "error": (task.error or "")[:500]} for task in tasks[args.offset:args.offset + args.limit]]}
        task, report = self._evaluation(experiment.id)
        return {**result, "evaluation_id": task.id, "summary": _summary(report), "metric_protocol": report["metric_protocol"],
            "per_class": _page_classes(report["per_class"], args.class_names, args.offset, args.limit),
            "report_sha256": task.report_sha256, "note": "混淆矩阵请用 read_confusion_matrix；定向错误图片请用 read_validation_errors。"}

    def dataset_examples(self, args):
        from ultralytics.utils.ops import xywh2ltwh

        export_id = args.dataset_export_id or self.tools.service.get(self.tools.run_id).dataset_export_id
        export, manifest = self._export(export_id)
        root = owned_path(self.settings, export.directory)
        classes = {item["class_id"]: item["name"] for item in manifest["classes"]}
        if args.class_name and args.class_name not in classes.values():
            raise ValueError("请使用封存数据集中的实际类别名称")
        paths = split_image_paths(self.settings, export, manifest, args.split)
        if set(args.image_ids) - {path.stem for path in paths}:
            raise PermissionError("指定图片不属于该封存 train/val 划分")
        selected, matching, images = [], 0, []
        for path in paths:
            if args.image_ids and path.stem not in args.image_ids:
                continue
            label = verified_export_file(self.settings, export, manifest, f"labels/{args.split}/{path.stem}.txt")
            labels = np.loadtxt(label, ndmin=2) if label.read_text().strip() else np.empty((0, 5))
            rows = [row for row in labels if not args.class_name or classes[int(row[0])] == args.class_name]
            if args.class_name and not rows:
                continue
            index = matching
            matching += 1
            if not args.offset <= index < args.offset + args.limit:
                continue
            image = verified_export_file(self.settings, export, manifest, path.relative_to(root).as_posix())
            with Image.open(image) as opened:
                dimensions = list(opened.size)
            boxes = [{"class_id": int(row[0]), "label": classes[int(row[0])], "bbox": [round(float(value), 6) for value in xywh2ltwh(row[1:].copy())]} for row in rows[args.box_offset:args.box_offset + 12]]
            selected.append({"image_id": path.stem, "dimensions": dimensions, "annotation_count": len(rows),
                "box_offset": args.box_offset, "annotations": boxes, "has_more_annotations": args.box_offset + 12 < len(rows)})
            images.append(image)
        return self._images({"dataset_export_id": export.id, "split": args.split, "total_matching_images": matching,
            "offset": args.offset, "has_more": args.offset + args.limit < matching, "items": selected,
            "note": "这些是已审核样例，不是增强后的训练批次，也不能用样例推断整体精度。"}, images, args.include_images)

    def validation_errors(self, args):
        task, report = self._evaluation(args.experiment_id)
        if args.class_name and args.class_name not in {row["name"] for row in report["per_class"]}:
            raise ValueError("请使用该验证报告中的实际类别名称")
        if args.small_only and args.error_kind == "fp":
            raise ValueError("small_only 只用于有真值面积记录的小目标漏检，不能用于误检")
        export, manifest = self._export(task.dataset_export_id)
        path = owned_path(self.settings, f"evaluations/{task.id}/errors.jsonl")
        if not path.is_file():
            raise DatasetError(409, "错误清单缺失")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != report["artifacts"].get("errors.jsonl"):
                raise DatasetError(409, "错误清单指纹变化")
        threshold = report["parameters"]["small_area_pixels"]
        total = 0
        def records():
            nonlocal total
            with path.open() as stream:
                for index, line in enumerate(stream):
                    row = json.loads(line)
                    if args.image_id and row["image_id"] != args.image_id:
                        continue
                    truth = [item for item in row["ground_truth"] if not args.class_name or item["label"] == args.class_name]
                    predictions = [item for item in row["predictions"] if not args.class_name or item["label"] == args.class_name]
                    missed = [item for item in truth if item["status"] == "fn" and (not args.small_only or (item.get("area_pixels") is not None and item["area_pixels"] < threshold))]
                    false = [item for item in predictions if item["status"] == "fp"]
                    score = len(missed) if args.error_kind == "fn" or args.small_only else len(false) if args.error_kind == "fp" else len(missed) + len(false)
                    if not score:
                        continue
                    total += 1
                    match_truth = args.error_kind != "fp"
                    match_predictions = args.error_kind != "fn" and not args.small_only
                    yield {"image_id": row["image_id"], "source_index": index, "matching_error_count": score,
                        "matching_counts": {"fn": len(missed) if match_truth else 0, "fp": len(false) if match_predictions else 0},
                        "image_counts_all_classes": row["counts"], "ground_truth_total": len(truth), "predictions_total": len(predictions),
                        "ground_truth": [{**box, "matches_filter": match_truth and box in missed} for box in truth[args.box_offset:args.box_offset + 6]],
                        "predictions": [{**box, "matches_filter": match_predictions and box in false} for box in predictions[args.box_offset:args.box_offset + 6]],
                        "annotations_truncated": args.box_offset + 6 < max(len(truth), len(predictions)), "box_offset": args.box_offset}
        ranked = heapq.nsmallest(args.offset + args.limit, records(), key=lambda row: (-row["matching_error_count"], row["source_index"]))
        selected = ranked[args.offset:args.offset + args.limit]
        images = [verified_split_image(self.settings, export, manifest, "val", row["image_id"]) for row in selected] if args.include_images else []
        return self._images({"evaluation_id": task.id, "class_name": args.class_name, "error_kind": args.error_kind,
            "small_only": args.small_only, "small_area_pixels_threshold": threshold, "total_matching_images": total,
            "offset": args.offset, "has_more": args.offset + args.limit < total, "items": selected,
            "protocol": report["parameters"], "report_sha256": task.report_sha256,
            "annotation_scope": "返回框已按 class_name 过滤；matches_filter 标记符合错误类型及面积条件的框。image_counts_all_classes 是整图计数，不能当作本次筛选数量。",
            "note": "按 matching_counts 中符合条件的错误数降序。错误清单使用固定置信度/IoU，不能代替 AP 或推断遮挡原因。"}, images, args.include_images)

    def compare(self, args):
        _, baseline = self.tools._own(args.baseline_experiment_id)
        _, candidate = self.tools._own(args.candidate_experiment_id)
        left_task, left = self._evaluation(baseline.id)
        right_task, right = self._evaluation(candidate.id)
        keys = ("backend", "ap_confidence_floor", "ap_iou_range", "confusion_confidence", "confusion_iou")
        reasons = []
        if baseline.dataset_export_id != candidate.dataset_export_id:
            reasons.append("数据工件不同，不能将验证指标差异直接当作调参收益")
        if any(key not in left["metric_protocol"] or key not in right["metric_protocol"] or left["metric_protocol"][key] != right["metric_protocol"][key] for key in keys):
            reasons.append("评估协议缺失或不一致，不能直接比较")
        comparable = not reasons
        def delta(a, b):
            return b - a if comparable and a is not None and b is not None else None
        left_classes = {row["name"]: row for row in left["per_class"]}
        rows = []
        for row in right["per_class"]:
            before = left_classes.get(row["name"], {})
            rows.append({"name": row["name"], "baseline_instances": before.get("validation_instances"), "candidate_instances": row.get("validation_instances"),
                "baseline_ap50_95": before.get("ap50_95"), "candidate_ap50_95": row.get("ap50_95"),
                "delta_ap50_95": delta(before.get("ap50_95"), row.get("ap50_95")),
                "delta_recall": delta(before.get("recall"), row.get("recall")), "unavailable_reason": row.get("unavailable_reason") or before.get("unavailable_reason")})
        design = comparison_design(baseline, candidate.parameters, candidate.dataset_export_id, resolved=candidate.resolved_parameters)
        # 参数变化与类别结果分别分页，避免实际训练器配置差异撑满单次阅读。
        for section in ("changed_variables", "effective_changes"):
            changes = list(design[section].items())
            design[section] = dict(changes[args.change_offset:args.change_offset + args.change_limit])
            design[f"{section}_page"] = {"total": len(changes), "offset": args.change_offset,
                "has_more": args.change_offset + args.change_limit < len(changes)}
        return {"baseline_experiment_id": baseline.id, "candidate_experiment_id": candidate.id,
            "comparable": comparable, "incomparability_reasons": reasons,
            "metrics": {name: {"baseline": left["summary"].get(name), "candidate": right["summary"].get(name), "delta": delta(left["summary"].get(name), right["summary"].get(name))} for name in METRICS},
            "per_class": _page_classes(rows, args.class_names, args.offset, args.limit),
            "comparison": design,
            "evaluation_ids": [left_task.id, right_task.id], "report_sha256": [left_task.report_sha256, right_task.report_sha256],
            "note": "比较不创建实验；多变量收益属于整组配置，单次比较不能证明因果。"}
