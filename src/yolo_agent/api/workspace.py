"""管理界面聚合查询与显式人工发布入口。"""

import hashlib

import yaml

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yolo_agent.models.agent import AgentRun
from yolo_agent.models.annotation import AnnotationRun, DatasetExport
from yolo_agent.models.dataset import Dataset, DatasetImage, DatasetVersion, Project
from yolo_agent.models.experiment import Experiment, ModelArtifact, TrainingTask
from yolo_agent.models.publication import ModelPublication, ProjectProductionModel
from yolo_agent.schemas.agent import agent_defaults
from yolo_agent.schemas.experiment import OPTIMIZERS, ModelRead, training_defaults, training_parameter_catalog
from yolo_agent.schemas.publication import PublicationRead, PublishConfirm
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.services.publication import PublicationService
from yolo_agent.settings import get_settings

router=APIRouter(prefix="/api/v1",tags=["管理工作区"])


def publication_service(request: Request) -> PublicationService:
    """绑定人工发布业务服务，不注册为 Agent 工具。

    :param request: 当前请求。
    :returns: 发布服务。
    """
    return PublicationService(request.app.state.engine,get_settings())


@router.get("/workspace/overview")
def overview(request: Request, project_id: str|None=None):
    """从业务库统计真实工作量，不以分页列表长度冒充总数。

    :param request: 当前应用资源。
    :param project_id: 可选当前项目。
    :returns: 计数、当前任务、真实候选和不含凭据的默认配置。
    """
    settings=get_settings()
    with Session(request.app.state.engine) as session:
        project=session.get(Project,project_id) if project_id else None
        if project_id and project is None:
            raise DatasetError(404,"项目不存在")
        datasets=select(Dataset.id)
        experiments=select(Experiment.id)
        agents=select(AgentRun.id)
        if project_id:
            datasets=datasets.where(Dataset.project_id==project_id)
            experiments=experiments.where(Experiment.project_id==project_id)
            agents=agents.where(AgentRun.project_id==project_id)
        versions=select(DatasetVersion.id).where(DatasetVersion.dataset_id.in_(datasets))
        counts={"projects":session.scalar(select(func.count()).select_from(Project)),"datasets":session.scalar(select(func.count()).select_from(Dataset).where(Dataset.id.in_(datasets))),"images":session.scalar(select(func.count()).select_from(DatasetImage).where(DatasetImage.dataset_id.in_(datasets))),"corrupt_images":session.scalar(select(func.count()).select_from(DatasetImage).where(DatasetImage.dataset_id.in_(datasets),DatasetImage.status=="corrupt")),"dataset_versions":session.scalar(select(func.count()).select_from(DatasetVersion).where(DatasetVersion.id.in_(versions))),"reviewed_exports":session.scalar(select(func.count()).select_from(DatasetExport).where(DatasetExport.version_id.in_(versions))),"pending_annotation_runs":session.scalar(select(func.count()).select_from(AnnotationRun).where(AnnotationRun.version_id.in_(versions),AnnotationRun.status=="pending_review")),"experiments":session.scalar(select(func.count()).select_from(Experiment).where(Experiment.id.in_(experiments))),"completed_experiments":session.scalar(select(func.count()).select_from(Experiment).where(Experiment.id.in_(experiments),Experiment.status=="completed")),"model_artifacts":session.scalar(select(func.count()).select_from(ModelArtifact).where(ModelArtifact.experiment_id.in_(experiments))),"active_agent_runs":session.scalar(select(func.count()).select_from(AgentRun).where(AgentRun.id.in_(agents),AgentRun.status.in_(["queued","running","waiting","canceling"])))}
        task=session.scalar(select(TrainingTask).where(TrainingTask.experiment_id.in_(experiments),TrainingTask.status.in_(["queued","running","canceling"])).order_by(TrainingTask.created_at.desc()).limit(1))
        latest=session.scalar(select(AgentRun).where(AgentRun.id.in_(agents)).order_by(AgentRun.created_at.desc()).limit(1))
        best=session.scalar(select(AgentRun).where(AgentRun.id.in_(agents),AgentRun.status=="completed",AgentRun.best_model_id.is_not(None)).order_by(AgentRun.created_at.desc()).limit(1))
        pointer=session.get(ProjectProductionModel,project_id) if project_id else None
        stopped=session.scalars(select(AgentRun).where(AgentRun.id.in_(agents),AgentRun.state["stop_notice"]["code"].as_string()=="stagnation").order_by(AgentRun.created_at.desc()).limit(5))
        notices=[{"run_id":item.id,"name":item.name,"status":item.status,"notice":item.state["stop_notice"]} for item in stopped]
        return {"project":{"id":project.id,"name":project.name,"description":project.description} if project else None,"counts":counts,"active_training":{"id":task.id,"experiment_id":task.experiment_id,"status":task.status,"current_epoch":task.current_epoch,"total_epochs":task.total_epochs,"device":task.device} if task else None,"latest_agent_run":{"id":latest.id,"name":latest.name,"phase":latest.phase,"status":latest.status,"state":latest.state} if latest else None,"autonomous_best":{"run_id":best.id,"model_id":best.best_model_id,"name":best.name} if best else None,"production_model_id":pointer.model_id if pointer else None,"notices":notices,"configuration":{"training_defaults":training_defaults().model_dump(),"training_parameter_groups":training_parameter_catalog(),"training_optimizers":list(OPTIMIZERS),"agent_defaults":agent_defaults().model_dump(),"search_space":yaml.safe_load(settings.optimization_config_path.read_text())["search_space"],"prepared_models":[name for name in ("yolo26n.pt","yolo26s.pt","yolo26m.pt") if (settings.training_model_dir/name).is_file() and (settings.training_model_dir/name).with_suffix(".provenance.json").is_file()],"upload_limits":{"files":settings.max_upload_files,"file_bytes":None,"batch_bytes":None,"labeled_bytes":None},"llm_model":settings.llm_model,"llm_configured":bool(settings.llm_model and settings.llm_api_key.get_secret_value()),"cvat_url":settings.cvat_public_url or settings.cvat_url,"mlflow_url":settings.mlflow_public_url or settings.mlflow_tracking_uri}}


@router.get("/models/{model_id}/publication-preview")
def publication_preview(model_id: str,service: PublicationService=Depends(publication_service)):
    """只读预览待人工决定的发布证据。

    :param model_id: 注册权重。
    :param service: 业务服务。
    :returns: 指标、局限、指纹和旧生产模型。
    """
    return service.preview(model_id)


@router.get("/models/{model_id}",response_model=ModelRead)
def model_detail(model_id: str,service: PublicationService=Depends(publication_service)):
    """按固定 ID 查询历史或当前权重，不依赖列表分页位置。

    :param model_id: 注册工件。
    :param service: 数据资源。
    :returns: 不含任意内部路径的模型记录。
    """
    with Session(service.engine) as session:
        model=session.get(ModelArtifact,model_id)
        if model is None:
            raise DatasetError(404,"模型不存在")
        return model


@router.post("/models/{model_id}/publish",response_model=PublicationRead,status_code=201)
def publish_model(model_id: str,data: PublishConfirm,service: PublicationService=Depends(publication_service)):
    """仅在明确人工确认内容与当前状态一致时更新生产指针。

    :param model_id: 已验证模型。
    :param data: 必须为 true 的确认、操作者、理由和指纹。
    :param service: 发布服务。
    :returns: 只追加的发布审计。
    """
    return service.publish(model_id,data)


@router.get("/model-publications",response_model=list[PublicationRead])
def list_publications(project_id: str|None=None,limit: int=Query(50,ge=1,le=200),service: PublicationService=Depends(publication_service)):
    """查看人工发布历史。

    :param project_id: 可选项目。
    :param limit: 返回上限。
    :param service: 业务服务。
    :returns: 不可覆盖审计。
    """
    statement=select(ModelPublication).order_by(ModelPublication.created_at.desc()).limit(limit)
    if project_id:
        statement=statement.where(ModelPublication.project_id==project_id)
    with Session(service.engine) as session:
        return list(session.scalars(statement))


@router.get("/model-publications/{publication_id}/download",response_class=FileResponse)
def publication_download(publication_id: str,service: PublicationService=Depends(publication_service)):
    """只提供固定发布版本的哈希一致权重副本。

    :param publication_id: 已确认发布标识。
    :param service: 业务服务。
    :returns: 发布权重，不支持任意路径下载。
    """
    with Session(service.engine) as session:
        publication=session.get(ModelPublication,publication_id)
        if publication is None:
            raise DatasetError(404,"发布版本不存在")
        path=owned_path(service.settings,publication.weight_path)
        if not path.is_file():
            raise DatasetError(409,"发布工件缺失")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest()!=publication.weight_sha256:
                raise DatasetError(409,"发布工件指纹变化")
        return FileResponse(path,media_type="application/octet-stream",filename=f"production-{publication.id}.pt")
