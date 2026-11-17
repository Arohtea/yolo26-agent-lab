"""验证评估、错误图定位、曲线工件与实验对比接口。"""

import hashlib
import json
from PIL import Image

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import DatasetExport
from yolo_agent.models.evaluation import EvaluationInvalidation, EvaluationTask
from yolo_agent.models.experiment import Experiment
from yolo_agent.schemas.evaluation import EvaluationCreate, EvaluationRead
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.services.dataset_evidence import verified_split_image
from yolo_agent.services.evaluation import EvaluationService
from yolo_agent.services.training import validate_export
from yolo_agent.settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["验证评估"])


@router.post("/evaluations/{task_id}/cancel")
def cancel_evaluation(task_id: str, request: Request):
    """主动终止验证评估，保留原训练权重和停止记录。"""
    from yolo_agent.services.task_control import cancel_task
    return cancel_task(request.app.state.engine, get_settings(), "evaluation", task_id)


def evaluation_service(request: Request) -> EvaluationService:
    """复用应用数据库，不在请求中执行推理。

    :param request: 当前请求。
    :returns: 异步评估服务。
    """
    return EvaluationService(request.app.state.engine, get_settings())


def checked_report(task_id: str, service: EvaluationService) -> tuple:
    """读取指纹一致的完成报告，所有下载都以此为白名单。

    :param task_id: 评估任务。
    :param service: 当前服务。
    :returns: 任务、报告路径、报告内容。
    """
    task = service.get(task_id)
    with Session(service.engine) as session:
        if session.get(EvaluationInvalidation, task_id):
            raise DatasetError(409, "该报告已声明失效，请查看任务中的失效原因及重新评估的报告")
    if task.status != "completed" or not task.report_path:
        raise DatasetError(409, "评估报告尚未就绪")
    path = owned_path(service.settings, task.report_path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != task.report_sha256:
        raise DatasetError(409, "评估报告缺失或指纹变化")
    return task, path, json.loads(path.read_text())


@router.post("/models/{model_id}/evaluate", response_model=EvaluationRead, status_code=202)
def create_evaluation(model_id: str, data: EvaluationCreate, service: EvaluationService=Depends(evaluation_service)):
    """提交固定验证集评估，不允许提供数据路径或测试集。

    :param model_id: 已注册权重。
    :param data: 有界诊断参数。
    :param service: 队列服务。
    :returns: 异步评估任务。
    """
    return service.create(model_id, data)


@router.get("/evaluations", response_model=list[EvaluationRead])
def list_evaluations(experiment_id: str|None=None, limit: int=Query(50,ge=1,le=200), offset: int=Query(0,ge=0), service: EvaluationService=Depends(evaluation_service)):
    """分页列出评估状态、样本支持量和摘要。

    :param experiment_id: 可选实验过滤。
    :param limit: 最大条数。
    :param offset: 分页偏移。
    :param service: 业务服务。
    :returns: 评估任务列表。
    """
    statement = select(EvaluationTask).order_by(EvaluationTask.created_at.desc()).limit(limit).offset(offset)
    if experiment_id:
        statement = statement.where(EvaluationTask.experiment_id==experiment_id)
    with Session(service.engine) as session:
        result = []
        for task in session.scalars(statement):
            if task.status in {"queued", "running", "canceling"}:
                task = service.get(task.id)
            notice = session.get(EvaluationInvalidation, task.id)
            result.append(EvaluationRead.model_validate(task).model_copy(update={"invalidated_reason":notice.reason if notice else None}))
        return result


@router.get("/evaluations/{task_id}", response_model=EvaluationRead)
def evaluation_status(task_id: str, service: EvaluationService=Depends(evaluation_service)):
    """查询真实计算状态，异常退出会收敛为 failed。

    :param task_id: 评估任务。
    :param service: 业务服务。
    :returns: 状态快照。
    """
    task = service.get(task_id)
    with Session(service.engine) as session:
        notice = session.get(EvaluationInvalidation, task.id)
        return EvaluationRead.model_validate(task).model_copy(update={"invalidated_reason":notice.reason if notice else None})


@router.get("/evaluations/{task_id}/report")
def evaluation_report(task_id: str, service: EvaluationService=Depends(evaluation_service)):
    """返回逐类 AP、真实 PR 曲线、混淆矩阵与证据建议。

    :param task_id: 评估任务。
    :param service: 业务服务。
    :returns: 指纹校验后的完整报告。
    """
    return checked_report(task_id, service)[2]


@router.get("/evaluations/{task_id}/errors")
def evaluation_errors(task_id: str, limit: int=Query(20,ge=1,le=50), offset: int=Query(0,ge=0), service: EvaluationService=Depends(evaluation_service)):
    """流式读取有限页错误图片及人审框与预测框，坐标为归一化左上角宽高。

    :param task_id: 评估任务。
    :param limit: 最大图片数。
    :param offset: 错误图片偏移。
    :param service: 业务服务。
    :returns: 真实错误图片页。
    """
    _, path, report = checked_report(task_id, service)
    error_path = path.parent / "errors.jsonl"
    if not error_path.is_file():
        raise DatasetError(409, "错误清单缺失或指纹变化")
    with error_path.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=report["artifacts"]["errors.jsonl"]:
            raise DatasetError(409, "错误清单指纹变化")
    result = []
    with error_path.open() as stream:
        for index,line in enumerate(stream):
            if index >= offset+limit:
                break
            if index >= offset:
                result.append(json.loads(line))
    return {"total":report["summary"]["errors"]["error_image_count"],"offset":offset,"items":result}


@router.get("/evaluations/{task_id}/images/{image_id}", response_class=FileResponse)
def evaluation_image(task_id: str, image_id: str, service: EvaluationService=Depends(evaluation_service)):
    """只提供该报告固定验证集内的图片，不支持任意文件访问。

    :param task_id: 已完成评估。
    :param image_id: 封存图片 ID。
    :param service: 业务服务。
    :returns: 验证图片。
    """
    task, _, _ = checked_report(task_id, service)
    with Session(service.engine) as session:
        export = session.get(DatasetExport, task.dataset_export_id)
        if export is None:
            raise DatasetError(409, "评估绑定的数据工件不存在")
    manifest = validate_export(service.settings, export, verify_files=False)
    path = verified_split_image(service.settings, export, manifest, "val", image_id)
    with Image.open(path) as image:
        media_type = {"PNG": "image/png", "JPEG": "image/jpeg", "MPO": "image/jpeg", "WEBP": "image/webp"}.get(image.format)
    if media_type is None:
        raise DatasetError(409, "不支持的验证图片编码")
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-cache"})


@router.get("/evaluations/{task_id}/artifacts/{artifact:path}", response_class=FileResponse)
def evaluation_artifact(task_id: str, artifact: str, service: EvaluationService=Depends(evaluation_service)):
    """仅下载已登记且哈希一致的评估工件，包括官方 PR/混淆矩阵图片。

    :param task_id: 完成评估任务。
    :param artifact: 报告内登记的相对工件名。
    :param service: 业务服务。
    :returns: 已验证工件。
    """
    _, report_path, report = checked_report(task_id, service)
    if artifact not in report["artifacts"]:
        raise DatasetError(404, "工件未登记")
    path = (report_path.parent / artifact).resolve()
    if not path.is_relative_to(report_path.parent) or not path.is_file():
        raise DatasetError(409, "工件缺失或路径越界")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=report["artifacts"][artifact]:
            raise DatasetError(409, "工件指纹变化")
    return FileResponse(path, filename=path.name)


@router.get("/comparisons/experiments")
def compare_experiments(ids: list[str]=Query(min_length=2,max_length=8), service: EvaluationService=Depends(evaluation_service)):
    """比较同项目实验，明确标记不同数据版本不可直接归因。

    :param ids: 两到八个实验 ID。
    :param service: 业务服务。
    :returns: 配置差异、指标和可比性。
    """
    with Session(service.engine) as session:
        experiments = [session.get(Experiment,identifier) for identifier in ids]
        if any(item is None for item in experiments):
            raise DatasetError(404, "部分实验不存在")
        if len({item.project_id for item in experiments})!=1:
            raise DatasetError(422, "只能对比同一项目的实验")
        items = []
        for experiment in experiments:
            evaluation = session.scalar(select(EvaluationTask).where(EvaluationTask.experiment_id==experiment.id,EvaluationTask.status=="completed",~EvaluationTask.id.in_(select(EvaluationInvalidation.task_id))).order_by(EvaluationTask.created_at.desc()).limit(1))
            items.append({"id":experiment.id,"name":experiment.name,"status":experiment.status,"parameters":experiment.parameters,"changed_variables":experiment.changed_variables,"optimization_reason":experiment.optimization_reason,"dataset_export_id":experiment.dataset_export_id,"evaluation_id":evaluation.id if evaluation else None,"metrics":evaluation.summary if evaluation else experiment.final_metrics,"metric_source":"independent_validation" if evaluation else "training_validator"})
        return {"same_dataset":len({item.dataset_export_id for item in experiments})==1,"test_used":False,"items":items}
