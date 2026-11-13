"""类别管理、真实自动标注及明确的人工审核入口。"""

from collections import Counter, defaultdict

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from yolo_agent.models.annotation import Annotation, AnnotationRun, ClassSchema, DatasetExport, ImageReview
from yolo_agent.models.dataset import Dataset, DatasetTask, DatasetVersion, Project, VersionImage
from yolo_agent.models.experiment import Experiment, ModelArtifact
from yolo_agent.schemas.annotation import AnnotationRead, AnnotationRunRead, AutoAnnotateCreate, ClassSchemaCreate, ClassSchemaRead, CvatReviewConfirm, ExportRead, ImageReviewCreate
from yolo_agent.schemas.agent import ClassDraftCreate
from yolo_agent.schemas.dataset import TaskRead
from yolo_agent.services.annotations import AnnotationService, annotation_digest
from yolo_agent.services.annotation_quality import source_quality
from yolo_agent.services.datasets import DatasetError, DatasetService, fail_dataset_task, owned_path
from yolo_agent.settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["标注与人工审核"])


def annotation_service(request: Request) -> AnnotationService:
    """绑定当前 API 资源。

    :param request: 当前请求。
    :returns: 标注服务。
    """
    return AnnotationService(request.app.state.engine, get_settings())


@router.get("/annotation-sources")
def annotation_sources(project_id: str, service: AnnotationService = Depends(annotation_service)):
    """返回真实可用模型及其有训练正样本支持的类别，不创建或下载模型。"""
    from yolo_agent.schemas.annotation import COCO_CLASSES
    from yolo_agent.services.annotators import coco_provenance, project_model_entry

    models, unavailable, coco = [], [], []
    with Session(service.engine) as session:
        if session.get(Project, project_id) is None:
            raise DatasetError(404, "项目不存在")
        identities = list(session.scalars(select(ModelArtifact.id).join(Experiment, Experiment.id == ModelArtifact.experiment_id).where(Experiment.project_id == project_id, Experiment.status == "completed", ModelArtifact.kind == "best").order_by(ModelArtifact.created_at.desc()).limit(50)))
        for identity in identities:
            try:
                entry = project_model_entry(session, service.settings, project_id, identity)
                models.append({key: entry[key] for key in ("model_id", "experiment_id", "base_model", "sha256", "covered_classes")})
            except (DatasetError, OSError, ValueError) as error:
                unavailable.append({"model_id": identity, "reason": str(error)})
    for name in ("yolo26n.pt", "yolo26s.pt"):
        try:
            coco.append({"name": name, "available": True, "sha256": coco_provenance(service.settings, name)["sha256"]})
        except (DatasetError, OSError, ValueError) as error:
            coco.append({"name": name, "available": False, "reason": str(error)})
    return {"coco_classes": list(COCO_CLASSES), "coco_models": coco, "default_coco_model": "yolo26n.pt", "project_models": models, "unavailable_project_models": unavailable, "project_model_limit": 50, "test_project_model_policy": "test 图片不使用项目模型，自动回退 Grounding DINO"}


@router.post("/class-schemas", response_model=ClassSchemaRead, status_code=201)
def create_schema(data: ClassSchemaCreate, service: AnnotationService = Depends(annotation_service)) -> ClassSchema:
    """由操作者明确创建类别体系版本，不注册为 Agent 自主操作。

    :param data: 类别名称、连续编号与英文提示词。
    :param service: 标注业务服务。
    :returns: 不可覆盖的类别体系。
    """
    with Session(service.engine, expire_on_commit=False) as session, session.begin():
        if session.get(Project, data.project_id) is None:
            raise DatasetError(404, "项目不存在")
        schema = ClassSchema(**data.model_dump())
        session.add(schema)
    return schema


@router.post("/class-schemas/draft")
def draft_schema(data: ClassDraftCreate, service: AnnotationService = Depends(annotation_service)) -> dict:
    """根据需求说明生成类别草案，只返回给操作者确认，不创建类别。

    :param data: 项目与检测需求。
    :param service: 标注业务服务。
    :returns: 带连续编号的类别草案与模型说明。
    """
    from yolo_agent.agents.planner import draft_classes

    with Session(service.engine) as session:
        if session.get(Project, data.project_id) is None:
            raise DatasetError(404, "项目不存在")
    return draft_classes(service.settings, data.requirement)


@router.get("/class-schemas", response_model=list[ClassSchemaRead])
def list_schemas(project_id: str | None = None, limit: int=Query(50,ge=1,le=200), offset: int=Query(0,ge=0), service: AnnotationService = Depends(annotation_service)) -> list[ClassSchema]:
    """查询可复用的类别定义。

    :param project_id: 可选项目过滤。
    :param limit: 单页条数。
    :param offset: 页偏移。
    :param service: 业务服务。
    :returns: 类别体系列表。
    """
    statement = select(ClassSchema).order_by(ClassSchema.created_at.desc()).limit(limit).offset(offset)
    if project_id:
        statement = statement.where(ClassSchema.project_id == project_id)
    with Session(service.engine) as session:
        return list(session.scalars(statement))


@router.post("/annotation-runs", status_code=202)
def create_run(data: AutoAnnotateCreate, service: AnnotationService = Depends(annotation_service)) -> dict:
    """提交真实模型推理，返回批次与后台任务标识。

    :param data: 图片版本、类别体系和受限检测策略。
    :param service: 业务服务。
    :returns: 批次状态及可复用任务查询地址。
    """
    run, task = service.create_run(data)
    return {"run":AnnotationRunRead.model_validate(run).model_dump(),"task":TaskRead.model_validate(task).model_dump()}


@router.get("/annotation-runs", response_model=list[AnnotationRunRead])
def list_runs(version_id: str | None = None, project_id: str|None=None, limit: int = Query(50, ge=1, le=200), offset: int=Query(0,ge=0), service: AnnotationService = Depends(annotation_service)) -> list[AnnotationRun]:
    """查询标注批次历史。

    :param version_id: 可选图片版本过滤。
    :param project_id: 可选项目归属。
    :param limit: 最大条数。
    :param offset: 页偏移。
    :param service: 业务服务。
    :returns: 标注批次。
    """
    statement = select(AnnotationRun).order_by(AnnotationRun.created_at.desc()).limit(limit).offset(offset)
    if project_id:
        statement=statement.join(DatasetVersion,DatasetVersion.id==AnnotationRun.version_id).join(Dataset,Dataset.id==DatasetVersion.dataset_id).where(Dataset.project_id==project_id)
    if version_id:
        statement = statement.where(AnnotationRun.version_id == version_id)
    with Session(service.engine) as session:
        return list(session.scalars(statement))


@router.get("/annotation-runs/{run_id}", response_model=AnnotationRunRead)
def get_run(run_id: str, service: AnnotationService = Depends(annotation_service)) -> AnnotationRun:
    """读取真实推理进度和审核状态。

    :param run_id: 批次 ID。
    :param service: 业务服务。
    :returns: 批次信息。
    """
    with Session(service.engine) as session:
        run = session.get(AnnotationRun, run_id)
        if run is None:
            raise DatasetError(404, "标注批次不存在")
        task_id = session.scalar(select(DatasetTask.id).where(DatasetTask.kind == "auto_annotate", DatasetTask.parameters["run_id"].as_string() == run_id).order_by(DatasetTask.created_at.desc()).limit(1)) if run.status in {"queued", "running"} else None
    if task_id:
        task = DatasetService(service.engine, service.settings).get_task(task_id)
        if task.status == "failed":
            fail_dataset_task(service.engine, task.id, task.error or "推理任务已终结")
        with Session(service.engine) as session:
            return session.get(AnnotationRun, run_id)
    return run


@router.get("/annotation-runs/{run_id}/tasks", response_model=list[TaskRead])
def annotation_tasks(run_id: str, limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0), service: AnnotationService = Depends(annotation_service)) -> list[DatasetTask]:
    """列出推理、CVAT 与导出任务的真实状态，失败记录不被后续重试覆盖。

    :param run_id: 标注批次。
    :param limit: 本页最大条数。
    :param offset: 分页偏移。
    :param service: 当前业务资源。
    :returns: 相关后台任务及失败原因。
    """
    get_run(run_id, service)
    with Session(service.engine) as session:
        tasks = list(session.scalars(select(DatasetTask).where(DatasetTask.parameters["run_id"].as_string() == run_id).order_by(DatasetTask.created_at.desc(), DatasetTask.id).limit(limit).offset(offset)))
    datasets = DatasetService(service.engine, service.settings)
    return [datasets.get_task(task.id) if task.status in {"queued", "running"} else task for task in tasks]


@router.get("/annotation-runs/{run_id}/quality")
def quality(run_id: str, service: AnnotationService = Depends(annotation_service)) -> dict:
    """区分候选标签统计与已审统计，不把置信度当成标注准确率。

    :param run_id: 标注批次。
    :param service: 业务服务。
    :returns: 类别级状态、低置信度及逐图漏标审核信息。
    """
    get_run(run_id, service)
    with Session(service.engine) as session:
        boxes = list(session.scalars(select(Annotation).where(Annotation.run_id == run_id)))
        reviews = list(session.scalars(select(ImageReview).where(ImageReview.run_id == run_id)))
        run = session.get(AnnotationRun, run_id)
        schema = session.get(ClassSchema, run.class_schema_id)
        per_class = {category["name"]:Counter() for category in schema.classes}
        with_boxes, quarantined = set(), set()
        flags = Counter()
        for box in boxes:
            per_class[box.class_name][box.review_status] += 1
            flags.update(box.flags)
            if box.review_status != "rejected":
                with_boxes.add(box.image_id)
            if box.flags and box.review_status == "pending":
                quarantined.add(box.image_id)
        return {"run_id":run_id,"review_states":dict(Counter(review.status for review in reviews)),"needs_missed_object_check":[review.image_id for review in reviews if not review.missed_objects_checked],"no_boxes_image_ids":[review.image_id for review in reviews if review.image_id not in with_boxes],"quarantined_image_ids":sorted(quarantined),"flag_counts":dict(flags),"per_class":{name:dict(counts) for name,counts in per_class.items()},**source_quality(session, run, boxes),"accuracy":None,"accuracy_reason":"候选置信度不等于准确率；质量需结合人工审核状态、漏标检查和独立核验结果判断"}


@router.get("/annotation-runs/{run_id}/images/{image_id}")
def image_annotations(run_id: str, image_id: str, service: AnnotationService = Depends(annotation_service)) -> dict:
    """提供人工审核看到的具体框和并发校验摘要。

    :param run_id: 标注批次。
    :param image_id: 图片标识。
    :param service: 业务服务。
    :returns: 真实框、显示尺寸、审核记录和摘要。
    """
    with Session(service.engine) as session:
        run = session.get(AnnotationRun, run_id)
        if run is None:
            raise DatasetError(404, "标注批次不存在")
        member = session.get(VersionImage, (run.version_id, image_id))
        if member is None:
            raise DatasetError(404, "图片不属于批次")
        boxes = list(session.scalars(select(Annotation).where(Annotation.run_id == run_id, Annotation.image_id == image_id).order_by(Annotation.id)))
        review = session.get(ImageReview, (run_id, image_id))
        return {"image":member.snapshot,"boxes":[AnnotationRead.model_validate(box).model_dump() for box in boxes],"annotation_digest":annotation_digest(boxes),"review":{ "status":review.status,"missed_objects_checked":review.missed_objects_checked,"reviewer":review.reviewer} if review else None}


@router.post("/annotation-runs/{run_id}/images/{image_id}/review")
def review_image(run_id: str, image_id: str, data: ImageReviewCreate, service: AnnotationService = Depends(annotation_service)) -> dict:
    """操作者逐框处理并确认漏标检查，Agent 不持有此工具。

    :param run_id: 标注批次。
    :param image_id: 图片标识。
    :param data: 人工处理决定及审核者。
    :param service: 业务服务。
    :returns: 最新审核摘要。
    """
    return service.review_image(run_id, image_id, data)


@router.post("/annotation-runs/{run_id}/cvat", response_model=TaskRead, status_code=202)
def push_cvat(run_id: str, service: AnnotationService = Depends(annotation_service)):
    """异步上传方向一致的图片和待审核框到 CVAT。

    :param run_id: 标注批次。
    :param service: 业务服务。
    :returns: CVAT 上传任务。
    """
    return service.schedule(run_id, "cvat_push")


@router.post("/annotation-runs/{run_id}/cvat/sync", response_model=TaskRead, status_code=202)
def pull_cvat(run_id: str, data: CvatReviewConfirm, service: AnnotationService = Depends(annotation_service)):
    """同步 CVAT 人工验收完成的内容，不自动更改 CVAT 作业状态。

    :param run_id: 标注批次。
    :param data: 审核者及漏标检查确认。
    :param service: 业务服务。
    :returns: 审核同步任务。
    """
    return service.schedule(run_id, "cvat_pull", data.model_dump())


@router.post("/annotation-runs/{run_id}/exports", response_model=TaskRead, status_code=202)
def export_yolo(run_id: str, service: AnnotationService = Depends(annotation_service)):
    """仅在所有图片和检测框完成审核后创建可训练工件。

    :param run_id: 标注批次。
    :param service: 业务服务。
    :returns: 导出任务；待审核数据返回 409。
    """
    return service.schedule(run_id, "yolo_export")


@router.get("/dataset-exports", response_model=list[ExportRead])
def list_exports(run_id: str | None = None, project_id: str|None=None, limit: int=Query(50,ge=1,le=200), offset: int=Query(0,ge=0), service: AnnotationService = Depends(annotation_service)) -> list[DatasetExport]:
    """查询可供训练绑定的已审核工件。

    :param run_id: 可选标注批次过滤。
    :param project_id: 可选项目归属。
    :param limit: 单页条数。
    :param offset: 页偏移。
    :param service: 业务服务。
    :returns: 工件摘要。
    """
    statement = select(DatasetExport).order_by(DatasetExport.created_at.desc()).limit(limit).offset(offset)
    if project_id:
        statement=statement.join(DatasetVersion,DatasetVersion.id==DatasetExport.version_id).join(Dataset,Dataset.id==DatasetVersion.dataset_id).where(Dataset.project_id==project_id)
    if run_id:
        statement = statement.where(DatasetExport.run_id == run_id)
    with Session(service.engine) as session:
        return list(session.scalars(statement))


@router.get("/dataset-exports/{export_id}/download", response_class=FileResponse)
def download_export(export_id: str, service: AnnotationService = Depends(annotation_service)) -> FileResponse:
    """下载带文件指纹和审核信息的标准 YOLO 数据集。

    :param export_id: 已审核工件 ID。
    :param service: 业务服务。
    :returns: ZIP 数据集工件。
    """
    import hashlib

    with Session(service.engine) as session:
        export = session.get(DatasetExport, export_id)
        if export is None:
            raise DatasetError(404, "工件不存在")
        path = owned_path(service.settings, export.archive_path)
        if not path.is_file():
            raise DatasetError(409, "数据工件缺失")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != export.archive_sha256:
                raise DatasetError(409, "工件哈希校验失败")
        return FileResponse(path, media_type="application/zip", filename=f"yolo-dataset-{export.id}.zip")
