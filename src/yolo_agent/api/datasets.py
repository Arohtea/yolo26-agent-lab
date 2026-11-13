"""项目和数据资产接口，耗时检查仅通过 RQ 提交。"""

from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from yolo_agent.models.dataset import Dataset, DatasetImage, DatasetTask, DatasetVersion, Project, VersionImage
from yolo_agent.schemas.dataset import AnalysisCreate, DatasetCreate, DatasetRead, ImageGroups, ImageRead, ProjectCreate, ProjectRead, TaskRead, VersionCreate, VersionImageRead, VersionRead
from yolo_agent.services.datasets import DatasetError, DatasetService, owned_path
from yolo_agent.schemas.dataset import LabeledImportOptions
from yolo_agent.services.dataset_transfer import submit_labeled_import, submit_dataset_export, download_dataset_export
from yolo_agent.settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["数据集管理"])


def dataset_service(request: Request) -> DatasetService:
    """复用应用引擎，避免为每个请求创建连接池。

    :param request: 当前请求。
    :returns: 数据集业务服务。
    """
    return DatasetService(request.app.state.engine, get_settings())


@router.post("/projects", response_model=ProjectRead, status_code=201)
def create_project(data: ProjectCreate, service: DatasetService = Depends(dataset_service)) -> Project:
    """创建后续数据集的业务归属。

    :param data: 项目名称和说明。
    :param service: 当前进程业务服务。
    :returns: 已保存的项目。
    """
    with Session(service.engine, expire_on_commit=False) as session, session.begin():
        project = Project(**data.model_dump())
        session.add(project)
    return project


@router.get("/projects", response_model=list[ProjectRead])
def list_projects(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), service: DatasetService = Depends(dataset_service)) -> list[Project]:
    """分页查询项目。

    :param limit: 本页最大条数。
    :param offset: 起始位置。
    :param service: 当前进程业务服务。
    :returns: 项目列表。
    """
    with Session(service.engine) as session:
        return list(session.scalars(select(Project).order_by(Project.created_at, Project.id).limit(limit).offset(offset)))


@router.post("/datasets", response_model=DatasetRead, status_code=201)
def create_dataset(data: DatasetCreate, service: DatasetService = Depends(dataset_service)) -> Dataset:
    """在项目中创建数据集，同项目内名称唯一。

    :param data: 项目 ID、数据集名称与说明。
    :param service: 当前进程业务服务。
    :returns: 已保存的数据集。
    """
    try:
        with Session(service.engine, expire_on_commit=False) as session, session.begin():
            if session.get(Project, data.project_id) is None:
                raise DatasetError(404, "项目不存在")
            dataset = Dataset(**data.model_dump())
            session.add(dataset)
        return dataset
    except IntegrityError as exc:
        raise DatasetError(409, "该项目已存在同名数据集") from exc


@router.get("/datasets", response_model=list[DatasetRead])
def list_datasets(project_id: str | None = None, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), service: DatasetService = Depends(dataset_service)) -> list[Dataset]:
    """按业务项目分页查询数据集。

    :param project_id: 可选项目过滤条件。
    :param limit: 本页最大条数。
    :param offset: 起始位置。
    :param service: 当前进程业务服务。
    :returns: 数据集列表。
    """
    statement = select(Dataset).order_by(Dataset.created_at, Dataset.id).limit(limit).offset(offset)
    if project_id:
        statement = statement.where(Dataset.project_id == project_id)
    with Session(service.engine) as session:
        return list(session.scalars(statement))


@router.get("/datasets/{dataset_id}", response_model=DatasetRead)
def get_dataset(dataset_id: str, service: DatasetService = Depends(dataset_service)) -> Dataset:
    """查询指定数据集，不把未分析图片数伪装成质量结论。

    :param dataset_id: 数据集 ID。
    :param service: 当前进程业务服务。
    :returns: 数据集基本信息。
    """
    with Session(service.engine) as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise DatasetError(404, "数据集不存在")
        return dataset


@router.post("/datasets/{dataset_id}/imports", response_model=TaskRead, status_code=202)
def import_images(dataset_id: str, files: list[UploadFile] = File(...), metadata: str = Form("{}"), service: DatasetService = Depends(dataset_service)) -> DatasetTask:
    """批量接收图片与逐图采集分组，返回异步任务。

    :param dataset_id: 数据集 ID。
    :param files: 支持 JPG、JPEG、PNG、WEBP 的 multipart 文件列表。
    :param metadata: 文件名到 video_id、scene_id、batch_id 的 JSON 对象。
    :param service: 当前进程业务服务。
    :returns: 已入队的导入任务；202 不表示解码检查已完成。
    """
    try:
        if len(metadata) > 512 * 1024:
            raise DatasetError(413, "分组元数据超过大小限制")
        groups = TypeAdapter(dict[str, ImageGroups]).validate_json(metadata)
        return service.import_images(dataset_id, files, groups)
    except ValidationError as exc:
        problems = [f"{'.'.join(str(part) for part in issue['loc']) or 'metadata'}: {issue['msg']}" for issue in exc.errors(include_input=False, include_context=False)[:3]]
        raise DatasetError(422, "分组元数据无效：" + "；".join(problems)) from exc
    finally:
        for file in files:
            file.file.close()


@router.post("/datasets/{dataset_id}/labeled-imports", response_model=TaskRead, status_code=202,
    openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
        "type": "object", "required": ["files"], "properties": {
            "files": {"type": "array", "items": {"type": "string", "format": "binary"}},
            "options": {"type": "string", "default": "{}", "description": "JSON：accepted_reviewed_source、reviewer、confirmation"}}}}}}})
async def import_labeled_dataset(dataset_id: str, request: Request, service: DatasetService = Depends(dataset_service)) -> DatasetTask:
    """上传 YOLO 文件夹或单个 ZIP，保留原划分；默认待审核，返回异步任务。

    files 保留相对路径。options 是 LabeledImportOptions 的 JSON；明确继承外部审核时必须填写审核者及来源说明。
    """
    # 文件夹可有数万对图片/标签，显式设置 multipart 上限并由上下文关闭临时文件。
    async with request.form(max_files=service.settings.max_dataset_images * 2 + 200, max_fields=4) as form:
        files = form.getlist("files")
        if not files or any(not isinstance(file, StarletteUploadFile) for file in files):
            raise DatasetError(422, "请选择 YOLO 数据集文件夹或 ZIP")
        raw = form.get("options", "{}")
        if not isinstance(raw, str) or len(raw) > 16 * 1024:
            raise DatasetError(422, "导入选项必须是小于 16 KiB 的 JSON")
        try:
            options = LabeledImportOptions.model_validate_json(raw)
        except ValidationError as error:
            raise DatasetError(422, "导入选项无效：" + "；".join(item["msg"] for item in error.errors(include_input=False, include_context=False))) from error
        return await run_in_threadpool(submit_labeled_import, service, dataset_id, files, options)


@router.post("/dataset-versions/{version_id}/exports", response_model=TaskRead, status_code=202)
def export_dataset_version(version_id: str, service: DatasetService = Depends(dataset_service)) -> DatasetTask:
    """导出版本图片及最新已完成批次的标注，保留审核状态；不授予训练资格。"""
    return submit_dataset_export(service, version_id)


@router.get("/dataset-tasks/{task_id}/download", response_class=FileResponse)
def download_dataset_task(task_id: str, service: DatasetService = Depends(dataset_service)) -> FileResponse:
    """下载经过 SHA-256 校验的数据集 ZIP；任务未完成时返回 409。"""
    return FileResponse(download_dataset_export(service, task_id), media_type="application/zip", filename=f"yolo-dataset-{task_id}.zip")


@router.get("/datasets/{dataset_id}/images", response_model=list[ImageRead])
def list_images(dataset_id: str, status: Literal["pending", "valid", "corrupt", "failed"] | None = None, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), service: DatasetService = Depends(dataset_service)) -> list[DatasetImage]:
    """分页查看图片检查结果及原始采集分组。

    :param dataset_id: 数据集 ID。
    :param status: 可选图片状态过滤。
    :param limit: 本页最大条数。
    :param offset: 起始位置。
    :param service: 当前进程业务服务。
    :returns: 图片元数据列表。
    """
    get_dataset(dataset_id, service)
    statement = select(DatasetImage).where(DatasetImage.dataset_id == dataset_id).order_by(DatasetImage.created_at, DatasetImage.id).limit(limit).offset(offset)
    if status:
        statement = statement.where(DatasetImage.status == status)
    with Session(service.engine) as session:
        return list(session.scalars(statement))


@router.get("/images/{image_id}/preview", response_class=FileResponse)
def preview_image(image_id: str, service: DatasetService = Depends(dataset_service)) -> FileResponse:
    """仅按平台图片 ID 返回经过解码的 JPEG 缩略图。

    :param image_id: 图片 ID。
    :param service: 当前进程业务服务。
    :returns: 有效图片的定向预览。
    """
    with Session(service.engine) as session:
        image = session.get(DatasetImage, image_id)
        if image is None:
            raise DatasetError(404, "图片不存在")
        if image.status != "valid" or not image.preview_path:
            raise DatasetError(409, "图片未通过完整检查，无法提供预览")
        path = owned_path(service.settings, image.preview_path)
        if not path.is_file():
            raise DatasetError(409, "图片预览缺失")
        return FileResponse(path, media_type="image/jpeg")


@router.post("/datasets/{dataset_id}/analyses", response_model=TaskRead, status_code=202)
def analyze_dataset(dataset_id: str, data: AnalysisCreate, service: DatasetService = Depends(dataset_service)) -> DatasetTask:
    """异步分析指定时刻的数据集图片集合。

    :param dataset_id: 数据集 ID。
    :param data: 感知哈希阈值。
    :param service: 当前进程业务服务。
    :returns: 可轮询的分析任务。
    """
    return service.schedule(dataset_id, "analysis", data)


@router.get("/dataset-tasks/{task_id}", response_model=TaskRead)
def get_task(task_id: str, service: DatasetService = Depends(dataset_service)) -> DatasetTask:
    """查询持久任务状态、进度、结果摘要及失败原因。

    :param task_id: 数据集任务 ID。
    :param service: 当前进程业务服务。
    :returns: 当前任务状态。
    """
    return service.get_task(task_id)


@router.get("/dataset-tasks/{task_id}/report", response_class=FileResponse)
def get_report(task_id: str, service: DatasetService = Depends(dataset_service)) -> FileResponse:
    """返回已完成任务的真实质量报告。

    :param task_id: 导入、分析或版本任务 ID。
    :param service: 当前进程业务服务。
    :returns: JSON 分析报告。
    """
    task = service.get_task(task_id)
    if task.status != "completed" or not task.report_path:
        raise DatasetError(409, "任务尚未成功完成，报告不可用")
    path = owned_path(service.settings, task.report_path)
    if not path.is_file():
        raise DatasetError(409, "任务报告缺失")
    return FileResponse(path, media_type="application/json", filename=f"dataset-report-{task_id}.json")


@router.get("/datasets/{dataset_id}/tasks", response_model=list[TaskRead])
def list_tasks(dataset_id: str, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), service: DatasetService = Depends(dataset_service)) -> list[DatasetTask]:
    """保留跨阶段可追踪的任务历史，单个活跃任务可查询其详细状态。

    :param dataset_id: 数据集 ID。
    :param limit: 本页最大条数。
    :param offset: 起始位置。
    :param service: 当前进程业务服务。
    :returns: 任务历史。
    """
    get_dataset(dataset_id, service)
    with Session(service.engine) as session:
        tasks = list(session.scalars(select(DatasetTask).where(DatasetTask.dataset_id == dataset_id).order_by(DatasetTask.created_at.desc(), DatasetTask.id).limit(limit).offset(offset)))
    # 每个数据集至多一个活跃任务，复用已有状态收敛，避免列表永久显示旧排队状态。
    return [service.get_task(task.id) if task.status in {"queued", "running"} else task for task in tasks]


@router.post("/datasets/{dataset_id}/versions", response_model=TaskRead, status_code=202)
def create_version(dataset_id: str, data: VersionCreate, service: DatasetService = Depends(dataset_service)) -> DatasetTask:
    """异步去重、分组划分并封存数据集版本。

    :param dataset_id: 数据集 ID。
    :param data: 划分比例、种子和采集分组策略。
    :param service: 当前进程业务服务。
    :returns: 版本创建任务，最终版本 ID 在成功结果中。
    """
    return service.schedule(dataset_id, "version", data)


@router.get("/datasets/{dataset_id}/versions", response_model=list[VersionRead])
def list_versions(dataset_id: str, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), service: DatasetService = Depends(dataset_service)) -> list[DatasetVersion]:
    """查询数据集的不可变历史版本。

    :param dataset_id: 数据集 ID。
    :param limit: 本页最大条数。
    :param offset: 起始位置。
    :param service: 当前进程业务服务。
    :returns: 版本摘要列表。
    """
    get_dataset(dataset_id, service)
    with Session(service.engine) as session:
        return list(session.scalars(select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id).order_by(DatasetVersion.number.desc()).limit(limit).offset(offset)))


@router.get("/dataset-versions/{version_id}", response_model=VersionRead)
def get_version(version_id: str, service: DatasetService = Depends(dataset_service)) -> DatasetVersion:
    """读取不可变版本的策略和划分统计。

    :param version_id: 版本 ID。
    :param service: 当前进程业务服务。
    :returns: 已封存版本摘要。
    """
    with Session(service.engine) as session:
        version = session.get(DatasetVersion, version_id)
        if version is None:
            raise DatasetError(404, "数据集版本不存在")
        return version


@router.get("/dataset-versions",response_model=list[VersionRead])
def project_versions(project_id: str|None=None,limit: int=Query(50,ge=1,le=200),offset: int=Query(0,ge=0),service: DatasetService=Depends(dataset_service)):
    """按项目分页浏览跨数据集版本，供训练和标注选择真实输入。

    :param project_id: 可选项目。
    :param limit: 页大小。
    :param offset: 页偏移。
    :param service: 数据服务。
    :returns: 已封存版本列表。
    """
    statement=select(DatasetVersion).order_by(DatasetVersion.created_at.desc()).limit(limit).offset(offset)
    if project_id:
        statement=statement.join(Dataset).where(Dataset.project_id==project_id)
    with Session(service.engine) as session:
        return list(session.scalars(statement))


@router.get("/dataset-versions/{version_id}/manifest", response_class=FileResponse)
def get_manifest(version_id: str, service: DatasetService = Depends(dataset_service)) -> FileResponse:
    """下载经过 SHA-256 完整性验证的版本清单。

    :param version_id: 版本 ID。
    :param service: 当前进程业务服务。
    :returns: 包含文件指纹、原始采集分组和划分结果的 JSON 快照。
    """
    return FileResponse(service.manifest_path(version_id), media_type="application/json", filename=f"dataset-version-{version_id}.json")


@router.get("/dataset-versions/{version_id}/images", response_model=list[VersionImageRead])
def list_version_images(version_id: str, split: Literal["train", "val", "test"] | None = None, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), service: DatasetService = Depends(dataset_service)) -> list[dict]:
    """按子集分页读取冻结的成员元数据。

    :param version_id: 版本 ID。
    :param split: 可选 train、val 或 test 过滤条件。
    :param limit: 本页最大条数。
    :param offset: 起始位置。
    :param service: 当前进程业务服务。
    :returns: 版本创建时的成员快照，不随数据集后续导入变化。
    """
    get_version(version_id, service)
    statement = select(VersionImage.snapshot).where(VersionImage.version_id == version_id).order_by(VersionImage.image_id).limit(limit).offset(offset)
    if split:
        statement = statement.where(VersionImage.split == split)
    with Session(service.engine) as session:
        return list(session.scalars(statement))
