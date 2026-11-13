"""实验与训练管理，实际计算经 RQ 执行。"""

import hashlib

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from yolo_agent.models.experiment import Experiment, ExperimentMetric, ModelArtifact, TrainingTask
from yolo_agent.schemas.experiment import ExperimentCreate, ExperimentRead, ModelExportCreate, ModelExportRead, ModelRead, TrainingParameters, TrainingTaskRead, training_defaults, training_parameter_catalog
from yolo_agent.services.exports import ExportService
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.services.training import TrainingService
from yolo_agent.services.chart_artifacts import CHARTS
from yolo_agent.settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["实验与训练"])


@router.get("/training/parameters")
def training_parameters():
    """训练参数定义、默认值与界面分组共享同一校验来源。"""
    return {"schema": TrainingParameters.model_json_schema(), "defaults": training_defaults().model_dump(), "groups": training_parameter_catalog(), "platform_managed": ["data", "device", "workers", "amp", "cache", "project", "name", "resume", "time", "val", "save", "pretrained", "classes", "single_cls"], "reference": "https://docs.ultralytics.com/zh/modes/train/"}


def training_service(request: Request) -> TrainingService:
    """绑定当前应用数据库。

    :param request: 当前请求。
    :returns: 训练编排服务。
    """
    return TrainingService(request.app.state.engine,get_settings())


@router.post("/experiments",response_model=ExperimentRead,status_code=201)
def create_experiment(data: ExperimentCreate,service: TrainingService=Depends(training_service)):
    """创建绑定已审核数据的实验，不自动开始计算。

    :param data: 受限训练配置及来源。
    :param service: 训练服务。
    :returns: 已冻结配置的实验。
    """
    return service.create_experiment(data)


@router.get("/experiments",response_model=list[ExperimentRead])
def list_experiments(project_id: str|None=None,limit: int=Query(50,ge=1,le=200),offset: int=Query(0,ge=0),service: TrainingService=Depends(training_service)):
    """分页查询可比较的历史实验。

    :param project_id: 项目过滤条件。
    :param limit: 页大小。
    :param offset: 偏移。
    :param service: 训练服务。
    :returns: 实验列表。
    """
    statement=select(Experiment).order_by(Experiment.created_at.desc()).limit(limit).offset(offset)
    if project_id:
        statement=statement.where(Experiment.project_id==project_id)
    with Session(service.engine) as session:
        return list(session.scalars(statement))


@router.get("/experiments/{experiment_id}",response_model=ExperimentRead)
def get_experiment(experiment_id: str,service: TrainingService=Depends(training_service)):
    """查询原始配置、解析后的设备配置及真实指标。

    :param experiment_id: 实验标识。
    :param service: 训练服务。
    :returns: 实验详情。
    """
    with Session(service.engine) as session:
        experiment=session.get(Experiment,experiment_id)
        if experiment is None:
            raise DatasetError(404,"实验不存在")
        return experiment


@router.post("/experiments/{experiment_id}/train",response_model=TrainingTaskRead,status_code=202)
def start_training(experiment_id: str,service: TrainingService=Depends(training_service)):
    """提交独立训练任务，单机资源占用冲突返回 409。

    :param experiment_id: 实验标识。
    :param service: 训练服务。
    :returns: 已入队任务。
    """
    return service.start(experiment_id)


@router.post("/experiments/{experiment_id}/resume",response_model=TrainingTaskRead,status_code=202)
def resume_training(experiment_id: str,service: TrainingService=Depends(training_service)):
    """从本实验最后一个合法中断检查点继续。

    :param experiment_id: 实验标识。
    :param service: 训练服务。
    :returns: 带父任务引用的新训练任务。
    """
    return service.start(experiment_id,resume=True)


@router.get("/experiments/{experiment_id}/tasks",response_model=list[TrainingTaskRead])
def experiment_tasks(experiment_id: str,service: TrainingService=Depends(training_service)):
    """保留每次启动、失败与恢复的独立记录。

    :param experiment_id: 实验标识。
    :param service: 训练服务。
    :returns: 训练尝试历史。
    """
    get_experiment(experiment_id,service)
    with Session(service.engine) as session:
        return list(session.scalars(select(TrainingTask).where(TrainingTask.experiment_id==experiment_id).order_by(TrainingTask.created_at.desc()).limit(200)))


@router.get("/training-tasks/{task_id}",response_model=TrainingTaskRead)
def get_training_task(task_id: str,service: TrainingService=Depends(training_service)):
    """读取真实进度、运行设备与停止状态。

    :param task_id: 训练任务。
    :param service: 训练服务。
    :returns: 当前任务快照。
    """
    return service.get_task(task_id)


@router.post("/training-tasks/{task_id}/cancel",response_model=TrainingTaskRead)
def cancel_training(task_id: str,service: TrainingService=Depends(training_service)):
    """请求在批次边界停止，完成验证及检查点保存后进入 stopped。

    :param task_id: 训练任务。
    :param service: 训练服务。
    :returns: 已停止或正在停止的状态。
    """
    return service.cancel(task_id)


@router.get("/training-tasks/{task_id}/logs",response_class=PlainTextResponse)
def training_logs(task_id: str,lines: int=Query(200,ge=1,le=2000),service: TrainingService=Depends(training_service)):
    """只读取日志末尾的有限窗口，避免大日志阻塞服务。

    :param task_id: 训练任务。
    :param lines: 最多返回的末尾行数。
    :param service: 训练服务。
    :returns: 最多 256 KiB 的日志尾部。
    """
    task=service.get_task(task_id)
    path=owned_path(service.settings,task.log_path)
    if not path.is_file():
        return PlainTextResponse("")
    with path.open("rb") as stream:
        stream.seek(0,2)
        stream.seek(max(0,stream.tell()-256*1024))
        content=stream.read().decode("utf-8",errors="replace")
    return PlainTextResponse("\n".join(content.splitlines()[-lines:]))


@router.get("/experiments/{experiment_id}/metrics")
def experiment_metrics(experiment_id: str,task_id: str|None=None,limit: int=Query(1000,ge=1,le=10000),offset: int=Query(0,ge=0),service: TrainingService=Depends(training_service)):
    """返回真实逐 epoch 指标，恢复尝试通过 task_id 区分。

    :param experiment_id: 实验标识。
    :param task_id: 可选训练尝试过滤。
    :param limit: 本页最多条数。
    :param offset: 偏移。
    :param service: 训练服务。
    :returns: 指标数据；未执行训练时为空列表。
    """
    get_experiment(experiment_id,service)
    statement=select(ExperimentMetric).where(ExperimentMetric.experiment_id==experiment_id).order_by(ExperimentMetric.epoch,ExperimentMetric.id).limit(limit).offset(offset)
    if task_id:
        statement=statement.where(ExperimentMetric.task_id==task_id)
    with Session(service.engine) as session:
        return [{"task_id":row.task_id,"epoch":row.epoch,"name":row.name,"value":row.value,"created_at":row.created_at} for row in session.scalars(statement)]


@router.get("/experiments/{experiment_id}/charts")
def training_charts(experiment_id: str, service: TrainingService = Depends(training_service)):
    """仅列出最近一次训练尝试已经写出的官方图，不拿旧尝试冒充当前结果。"""
    get_experiment(experiment_id, service)
    with Session(service.engine) as session:
        task = session.scalar(select(TrainingTask).where(TrainingTask.experiment_id == experiment_id).order_by(TrainingTask.created_at.desc()).limit(1))
        if task is None:
            return []
        root = owned_path(service.settings, f"training/{task.id}/run")
        return [{"name": name, "task_id": task.id, "url": f"/api/v1/training-tasks/{task.id}/charts/{name}"}
                for name, filename in CHARTS.items() if (root / filename).is_file()]


@router.get("/training-tasks/{task_id}/charts/{chart}", response_class=FileResponse)
def training_chart(task_id: str, chart: str, service: TrainingService = Depends(training_service)):
    """按固定图名展示真实训练结果图，可在新标签页查看原尺寸。"""
    service.get_task(task_id)
    if chart not in CHARTS:
        raise DatasetError(404, "不支持的图表")
    path = owned_path(service.settings, f"training/{task_id}/run/{CHARTS[chart]}")
    if not path.is_file():
        raise DatasetError(404, "该次训练尚未生成此图")
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


@router.get("/models",response_model=list[ModelRead])
def list_models(experiment_id: str|None=None,project_id: str|None=None,limit: int=Query(100,ge=1,le=200),offset: int=Query(0,ge=0),service: TrainingService=Depends(training_service)):
    """查询注册的候选权重，候选状态不等同于生产发布。

    :param experiment_id: 实验过滤。
    :param project_id: 项目过滤。
    :param limit: 最大返回条数。
    :param offset: 页偏移。
    :param service: 训练服务。
    :returns: 注册工件。
    """
    statement=select(ModelArtifact).order_by(ModelArtifact.created_at.desc()).limit(limit).offset(offset)
    if experiment_id:
        statement=statement.where(ModelArtifact.experiment_id==experiment_id)
    if project_id:
        statement=statement.join(Experiment).where(Experiment.project_id==project_id)
    with Session(service.engine) as session:
        return list(session.scalars(statement))


@router.get("/models/{model_id}/download",response_class=FileResponse)
def download_model(model_id: str,service: TrainingService=Depends(training_service)):
    """按固定工件 ID 校验并下载权重，不允许读取任意文件。

    :param model_id: 注册工件标识。
    :param service: 训练服务。
    :returns: 指纹一致的权重文件。
    """
    with Session(service.engine) as session:
        model=session.get(ModelArtifact,model_id)
        if model is None:
            raise DatasetError(404,"模型工件不存在")
        path=owned_path(service.settings,model.path)
        if not path.is_file():
            raise DatasetError(409,"模型工件缺失")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest()!=model.sha256:
                raise DatasetError(409,"模型工件指纹变化")
        return FileResponse(path,media_type="application/octet-stream",filename=f"{model.kind}-{model.id}.pt")


def export_service(request: Request) -> ExportService:
    """绑定当前应用数据库。

    :param request: 当前请求。
    :returns: 导出编排服务。
    """
    return ExportService(request.app.state.engine,get_settings())


@router.post("/models/{model_id}/exports",response_model=ModelExportRead,status_code=202)
def create_model_export(model_id: str,data: ModelExportCreate,service: ExportService=Depends(export_service)):
    """把注册权重转换为部署格式，转换和校验在 Worker 中进行。

    :param model_id: 注册权重。
    :param data: 目标格式与原因。
    :param service: 导出服务。
    :returns: 已入队的导出任务。
    """
    return service.create(model_id,data)


@router.get("/model-exports",response_model=list[ModelExportRead])
def list_model_exports(project_id: str|None=None,model_id: str|None=None,service: ExportService=Depends(export_service)):
    """按项目或单个权重列出导出历史。

    :param project_id: 项目过滤。
    :param model_id: 权重过滤。
    :param service: 导出服务。
    :returns: 最新在前的导出记录。
    """
    if not project_id and not model_id:
        raise DatasetError(422,"请指定 project_id 或 model_id")
    statement=select(ModelArtifact.id)
    if model_id:
        statement=statement.where(ModelArtifact.id==model_id)
    if project_id:
        statement=statement.join(Experiment).where(Experiment.project_id==project_id)
    with Session(service.engine) as session:
        ids=list(session.scalars(statement))
    return [service.get(item.id) if item.status in {"queued","running"} else item for item in service.list(model_ids=ids)]


@router.get("/model-exports/{export_id}/download",response_class=FileResponse)
def download_model_export(export_id: str,service: ExportService=Depends(export_service)):
    """校验指纹后下载已通过运行时加载检查的导出文件。

    :param export_id: 导出任务。
    :param service: 导出服务。
    :returns: 部署格式文件。
    """
    record=service.get(export_id)
    if record.status!="completed" or not record.path:
        raise DatasetError(409,"导出尚未完成")
    path=owned_path(service.settings,record.path)
    if not path.is_file():
        raise DatasetError(409,"导出文件缺失")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=record.sha256:
            raise DatasetError(409,"导出文件指纹变化")
    return FileResponse(path,media_type="application/octet-stream",filename=path.name)
