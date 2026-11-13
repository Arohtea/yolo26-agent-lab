"""人工发布采用独立只读工件与条件更新，不由 Agent 自动选择生产模型。"""

import hashlib
import json
import shutil

from sqlalchemy import Engine, select, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from yolo_agent.models.dataset import new_id, utc_now
from yolo_agent.models.evaluation import EvaluationInvalidation, EvaluationTask
from yolo_agent.models.experiment import Experiment, ModelArtifact
from yolo_agent.models.publication import ModelPublication, ProjectProductionModel
from yolo_agent.schemas.evaluation import EvaluationCreate
from yolo_agent.schemas.publication import PublishConfirm
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.settings import Settings


class PublicationService:
    """只允许明确人工确认的、经过有效验证的注册权重进入生产指针。"""

    def __init__(self, engine: Engine, settings: Settings):
        """绑定数据库与本地工件存储。

        :param engine: 业务数据库。
        :param settings: 固定存储根目录。
        :returns: 无。
        """
        self.engine,self.settings=engine,settings

    def preview(self, model_id: str) -> dict:
        """提供人工决策的真实指标、样本支持与当前生产状态，不执行发布。

        :param model_id: 已注册的实际权重。
        :returns: 指纹一致的候选预览及明确局限。
        """
        with Session(self.engine) as session:
            model=session.get(ModelArtifact,model_id)
            if model is None:
                raise DatasetError(404,"模型工件不存在")
            experiment=session.get(Experiment,model.experiment_id)
            if experiment.status!="completed":
                raise DatasetError(409,"只有已完成实验且经过独立验证的权重可发布")
            evaluations=list(session.scalars(select(EvaluationTask).where(EvaluationTask.model_id==model_id,EvaluationTask.status=="completed",~EvaluationTask.id.in_(select(EvaluationInvalidation.task_id))).order_by(EvaluationTask.created_at.desc())))
            evaluation=next((row for row in evaluations if row.parameters==EvaluationCreate().model_dump()),None)
            if evaluation is None or evaluation.summary.get("map50_95") is None:
                raise DatasetError(409,"缺少默认验证协议的有效报告，请先评估该权重")
            pointer=session.get(ProjectProductionModel,experiment.project_id)
            report_path=owned_path(self.settings,evaluation.report_path)
            if not report_path.is_file() or hashlib.sha256(report_path.read_bytes()).hexdigest()!=evaluation.report_sha256:
                raise DatasetError(409,"验证报告缺失或指纹变化")
            report=json.loads(report_path.read_text())
            if report.get("test_used") is not False or report.get("split")!="val":
                raise DatasetError(409,"发布评估必须来自固定验证集")
            path=owned_path(self.settings,model.path)
            if not path.is_file():
                raise DatasetError(409,"注册权重文件缺失")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream,"sha256").hexdigest()!=model.sha256:
                    raise DatasetError(409,"注册权重指纹变化")
            return {"model_id":model.id,"project_id":experiment.project_id,"experiment_id":experiment.id,"kind":model.kind,"sha256":model.sha256,"evaluation_id":evaluation.id,"evaluation_report_sha256":evaluation.report_sha256,"metrics":evaluation.summary,"per_class":report["per_class"],"dataset_export_id":experiment.dataset_export_id,"current_model_id":pointer.model_id if pointer else None,"approval_required":True,"test_used":False,"warnings":report["limitations"]+["发布会更新本项目生产指针，样本支持量与错误应由操作者确认"]}

    def publish(self, model_id: str, confirmation: PublishConfirm) -> ModelPublication:
        """校验显式确认后封存副本，以条件更新避免覆盖另一位操作者的新发布。

        :param model_id: 选定注册模型。
        :param confirmation: 人工名称、理由、指纹和旧生产指针。
        :returns: 不可覆盖的发布记录。
        """
        preview=self.preview(model_id)
        if confirmation.expected_sha256!=preview["sha256"]:
            raise DatasetError(409,"待发布权重与确认指纹不一致，请重新核对")
        if confirmation.expected_current_model_id!=preview["current_model_id"]:
            raise DatasetError(409,"生产版本已变化，请刷新后重新确认替换对象")
        if preview["current_model_id"]==model_id:
            raise DatasetError(409,"该模型已经是当前生产版本")
        publication_id=new_id()
        directory=owned_path(self.settings,f"production/{publication_id}")
        directory.mkdir(parents=True)
        with Session(self.engine) as session:
            source=owned_path(self.settings,session.get(ModelArtifact,model_id).path)
        weight=directory/"model.pt"
        shutil.copyfile(source,weight)
        with weight.open("rb") as stream:
            digest=hashlib.file_digest(stream,"sha256").hexdigest()
        if digest!=preview["sha256"]:
            raise DatasetError(409,"封存过程中权重变化，发布未生效")
        manifest=directory/"manifest.json"
        manifest.write_text(json.dumps({**preview,"publication_id":publication_id,"reviewer":confirmation.reviewer,"reason":confirmation.reason,"approved":True,"created_at":utc_now()},ensure_ascii=False,indent=2)+"\n")
        weight.chmod(0o444)
        manifest.chmod(0o444)
        try:
            with Session(self.engine,expire_on_commit=False) as session,session.begin():
                evaluation=session.get(EvaluationTask,preview["evaluation_id"])
                if session.get(EvaluationInvalidation,evaluation.id) or evaluation.report_sha256!=preview["evaluation_report_sha256"]:
                    raise DatasetError(409,"确认期间评估已失效，发布未生效")
                publication=ModelPublication(id=publication_id,project_id=preview["project_id"],model_id=model_id,previous_model_id=preview["current_model_id"],evaluation_id=evaluation.id,reviewer=confirmation.reviewer,reason=confirmation.reason,weight_path=weight.relative_to(self.settings.storage_dir.resolve()).as_posix(),weight_sha256=digest,manifest_path=manifest.relative_to(self.settings.storage_dir.resolve()).as_posix(),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
                session.add(publication)
                session.flush()
                if preview["current_model_id"] is None:
                    session.add(ProjectProductionModel(project_id=preview["project_id"],model_id=model_id,publication_id=publication.id))
                    session.flush()
                else:
                    changed=session.execute(update(ProjectProductionModel).where(ProjectProductionModel.project_id==preview["project_id"],ProjectProductionModel.model_id==preview["current_model_id"]).values(model_id=model_id,publication_id=publication.id,updated_at=utc_now()))
                    if changed.rowcount!=1:
                        raise DatasetError(409,"另一位操作者已更新生产版本，请刷新")
                    session.get(ModelArtifact,preview["current_model_id"]).status="retired"
                session.get(ModelArtifact,model_id).status="published"
        except (IntegrityError,OperationalError) as error:
            raise DatasetError(409,"发布状态发生竞争，生产指针未被本次请求覆盖，请刷新") from error
        return publication
