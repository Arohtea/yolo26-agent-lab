"""使用官方 CVAT SDK 上传与读取真实审核任务。"""

import hashlib
import json
import logging
from pathlib import Path

from cvat_sdk import make_client, models
from cvat_sdk.core.helpers import get_paginated_collection
from cvat_sdk.core.proxies.tasks import ResourceType

from yolo_agent.services.datasets import DatasetError
from yolo_agent.settings import Settings
from yolo_agent.services.annotators import cvat_eligible


def cvat_client(settings: Settings):
    """凭据仅取自部署配置，不接受任务参数中的远程地址。

    :param settings: 固定 CVAT 服务及密钥配置。
    :returns: 可作为上下文管理器使用的官方客户端。
    """
    return make_client(settings.cvat_url, credentials=(settings.cvat_username, settings.cvat_password.get_secret_value()))


def fingerprint(shapes: list[dict]) -> str:
    """保留 CVAT 框、属性与删除变化的稳定摘要。

    :param shapes: CVAT 原始框记录。
    :returns: 不依赖服务返回顺序的 SHA-256。
    """
    return hashlib.sha256(json.dumps(sorted(shapes, key=lambda shape:shape.get("id", 0)), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def upload_cvat(settings: Settings, run_id: str, classes: list[dict], frame_paths: list[Path], boxes: list[dict], *, progress=None, on_created=None) -> tuple[int, str]:
    """创建本地图片任务并按实际帧名关联自动标注。

    :param settings: CVAT 服务配置。
    :param run_id: 平台标注批次。
    :param classes: 已冻结类别体系。
    :param frame_paths: 方向已统一、保留原编码的图片。
    :param boxes: 待人工审核的真实检测框。
    :returns: CVAT 任务 ID 与上传后标注摘要。
    """
    if progress:
        progress("connecting_cvat")
    with cvat_client(settings) as client:
        task_name = f"YOLO Agent 审核 {run_id}"
        matches = get_paginated_collection(client.api_client.tasks_api.list_endpoint, search=run_id)
        existing = [client.tasks.retrieve(task.id) for task in matches if task.name == task_name]
        if len(existing) > 1:
            raise DatasetError(409, "发现多个同批次 CVAT 任务，请人工确定要保留的任务")
        if existing:
            task = existing[0]
        else:
            task = client.tasks.create(spec={"name":task_name, "labels":[{"name":category["name"],"type":"rectangle","attributes":[{"name":name,"input_type":"text","mutable":False,"values":[],"default_value":""} for name in ("platform_id", "annotation_source", "model_version", "confidence", "flags")]} for category in classes]})
        # 创建与传输分开；进程在大文件传输中被终止时，平台仍能定位远程任务。
        if on_created:
            on_created(task.id)
        if not existing or not task.size:
            if progress:
                progress("uploading_images")
            task.upload_data(
                resource_type=ResourceType.LOCAL, resources=[str(path) for path in frame_paths],
                params={"image_quality":100,"sorting_method":"lexicographical"},
            )
            task.fetch()
        if progress:
            progress("matching_frames")
        frame_lookup = {Path(frame.name).stem:index for index,frame in enumerate(task.get_frames_info())}
        if set(frame_lookup) != {path.stem for path in frame_paths}:
            raise DatasetError(409, "CVAT 实际帧与平台图片不一致，拒绝关联标注")
        if existing:
            current=task.get_annotations()
            # 空标注也可能是人工删除全部误检；服务版本字段不足以可靠区分，重放只关联。
            logging.getLogger(__name__).info("复用 CVAT 已有任务，保留全部当前人工内容",extra={"run_id":run_id,"cvat_task_id":task.id})
            return task.id,fingerprint([shape.to_dict() for shape in current.shapes])
        label_lookup = {label.name:label for label in task.get_labels()}
        if set(label_lookup) != {category["name"] for category in classes}:
            raise DatasetError(409, "已有 CVAT 类别与当前体系不一致，禁止覆盖")
        shapes = []
        for box in boxes:
            if not cvat_eligible(box):
                continue
            label = label_lookup[box["class_name"]]
            attributes = {attribute.name: attribute.id for attribute in label.attributes}
            values = {"platform_id":box["id"],"annotation_source":box.get("annotation_source", ""),"model_version":box.get("annotation_model_version", ""),"confidence":"" if box.get("confidence") is None else str(box["confidence"]),"flags":json.dumps(box.get("flags", []), ensure_ascii=False)}
            shapes.append(models.LabeledShapeRequest(
                type="rectangle", frame=frame_lookup[box["image_id"]], label_id=label.id,
                points=box["bbox"], source="auto", occluded=False, z_order=0,
                attributes=[models.AttributeValRequest(spec_id=attributes[name], value=value) for name,value in values.items()],
            ))
        if progress:
            progress("uploading_annotations")
        task.set_annotations(models.LabeledDataRequest(shapes=shapes, tags=[], tracks=[]))
        current = task.get_annotations()
        return task.id, fingerprint([shape.to_dict() for shape in current.shapes])


def download_cvat(settings: Settings, task_id: int, *, require_accepted: bool) -> dict:
    """读取实际框和人工验收状态，拒绝轨迹、多边形或混合标签静默丢失。

    :param settings: 固定 CVAT 服务配置。
    :param task_id: 已关联的任务 ID。
    :param require_accepted: 是否必须所有作业处于人工 acceptance/completed。
    :returns: 真实帧、类别和形状及内容摘要。
    """
    with cvat_client(settings) as client:
        task = client.tasks.retrieve(task_id)
        if task.get_meta().deleted_frames:
            raise DatasetError(409, "CVAT 存在删除的帧，不能将跳过的图片静默导出为负样本")
        jobs = task.get_jobs()
        if require_accepted:
            if not jobs:
                raise DatasetError(409, "CVAT 审核作业尚未准备好。请等待任务创建完成并刷新审核工作区，再进入作业逐图检查。")
            incomplete = [job for job in jobs if job.stage != "acceptance" or job.state != "completed"]
            if incomplete:
                stages = {"annotation": "标注", "validation": "检查", "acceptance": "验收"}
                states = {"new": "未开始", "in progress": "进行中", "completed": "已完成", "rejected": "已驳回"}
                examples = "；".join(f"作业 #{job.id}：{stages.get(job.stage, job.stage)} / {states.get(job.state, job.state)}" for job in incomplete[:5])
                raise DatasetError(409, f"还有 {len(incomplete)} 个 CVAT 作业未完成验收（{examples}）。请先保存修改，将每个作业的阶段设为“验收（acceptance）”、状态设为“已完成（completed）”，再回到工作台确认并回传。")
        annotations = task.get_annotations()
        if annotations.tracks or annotations.tags:
            raise DatasetError(422, "当前仅支持静态矩形检测标注，不能忽略轨迹或分类标签")
        shapes = [shape.to_dict() for shape in annotations.shapes]
        if any(shape["type"] != "rectangle" or shape.get("rotation", 0) != 0 for shape in shapes):
            raise DatasetError(422, "YOLO Detection 仅接收非旋转矩形，请先在 CVAT 修正")
        return {
            "frames":[Path(frame.name).stem for frame in task.get_frames_info()],
            "labels":{label.id:label.name for label in task.get_labels()},
            "platform_attributes":{label.id:next((attribute.id for attribute in label.attributes if attribute.name == "platform_id"), None) for label in task.get_labels()},
            "shapes":shapes, "fingerprint":fingerprint(shapes),
        }
