"""自主运行的创建、进度、人工控制、日志与最终报告。"""

import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from yolo_agent.models.agent import AgentEvent, AgentIteration, AgentRun
from yolo_agent.models.annotation import AnnotationRun, ClassSchema, DatasetExport
from yolo_agent.models.dataset import Dataset, DatasetVersion, Project
from yolo_agent.schemas.agent import AgentOperatorMessage, AgentResume, AgentRunCreate, AgentRunRead, AgentWake, OperatorInstructionRead, OperatorInstructionRevoke, TrainingPromptDraft, TrainingPromptOptimize, TrainingSetup
from yolo_agent.services.agents import AgentService
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.settings import get_settings

router=APIRouter(prefix="/api/v1",tags=["Agent 自主运行"])


def agent_service(request: Request) -> AgentService:
    """为 API 请求绑定业务资源。

    :param request: 当前应用请求。
    :returns: 不执行长计算的 Agent 服务。
    """
    return AgentService(request.app.state.engine,get_settings())


@router.post("/training-prompts/optimize", response_model=TrainingPromptDraft)
def optimize_prompt(data: TrainingPromptOptimize, service: AgentService = Depends(agent_service)):
    """仅润色需求文本并返回建议，不写库、不创建流程或训练任务。"""
    from yolo_agent.agents.planner import optimize_training_prompt

    with Session(service.engine) as session:
        project = session.get(Project, data.project_id)
        if project is None:
            raise DatasetError(404, "项目不存在")
        context = {"project_name": project.name, "operator_parameters": data.initial_parameters.model_dump(exclude_unset=True), "policy": data.policy.model_dump(), "model_locked": len(data.policy.allowed_models) == 1}
        if data.dataset_export_id:
            row = session.execute(select(Dataset, ClassSchema).select_from(DatasetExport)
                .join(DatasetVersion, DatasetExport.version_id == DatasetVersion.id)
                .join(Dataset, DatasetVersion.dataset_id == Dataset.id)
                .join(AnnotationRun, DatasetExport.run_id == AnnotationRun.id)
                .join(ClassSchema, AnnotationRun.class_schema_id == ClassSchema.id)
                .where(DatasetExport.id == data.dataset_export_id, Dataset.project_id == project.id,
                    Dataset.deleted_at.is_(None), ClassSchema.project_id == project.id)).first()
            if row is None:
                raise DatasetError(404, "所选训练数据不存在或不属于当前项目")
            dataset, schema = row
            context["dataset"] = {"name": dataset.name, "classes": [{"class_id": item["class_id"], "name": item["name"]} for item in schema.classes]}
    return optimize_training_prompt(service.settings, data.requirement, context)


@router.post("/agent-runs",response_model=AgentRunRead,status_code=202)
def create_run(data: AgentRunCreate,service: AgentService=Depends(agent_service)):
    """创建冻结预算的流程，由独立控制器推进。

    :param data: 数据入口和操作者预算。
    :param service: 业务服务。
    :returns: 已持久化运行。
    """
    return service.create(data)


@router.get("/agent-runs",response_model=list[AgentRunRead])
def list_runs(project_id: str|None=None,limit: int=Query(50,ge=1,le=200),offset: int=Query(0,ge=0),service: AgentService=Depends(agent_service)):
    """分页读取当前阶段、等待原因与历史结果。

    :param project_id: 项目过滤。
    :param limit: 页大小。
    :param offset: 偏移。
    :param service: 业务服务。
    :returns: 运行列表。
    """
    statement=select(AgentRun).order_by(AgentRun.created_at.desc()).limit(limit).offset(offset)
    if project_id:
        statement=statement.where(AgentRun.project_id==project_id)
    with Session(service.engine) as session:
        return list(session.scalars(statement))


@router.get("/agent-runs/{run_id}",response_model=AgentRunRead)
def get_run(run_id: str,service: AgentService=Depends(agent_service)):
    """查询业务进度，检查点内容不作为可写 API 暴露。

    :param run_id: 运行标识。
    :param service: 业务服务。
    :returns: 状态快照。
    """
    return service.get(run_id)


@router.get("/agent-runs/{run_id}/budget")
def run_budget(run_id: str,service: AgentService=Depends(agent_service)):
    """查看全部尝试消耗的真实预算。

    :param run_id: 业务运行。
    :param service: 业务服务。
    :returns: 实验数、用时和剩余预算。
    """
    return service.budget(run_id)


@router.post("/agent-runs/{run_id}/wake", response_model=AgentRunRead, status_code=202)
def wake_run(run_id: str, data: AgentWake, service: AgentService = Depends(agent_service)):
    """提交立即唤醒请求及可选问题，由现有控制器恢复同一检查点。"""
    return service.wake(run_id, data.question, sleep_until=data.sleep_until)


@router.get("/agent-runs/{run_id}/operator-messages", response_model=list[AgentOperatorMessage])
def operator_messages(run_id: str, service: AgentService = Depends(agent_service)):
    """读取最近十次唤醒请求与回答，独立于长流程事件分页。"""
    service.get(run_id)
    return service.operator_messages(run_id)


@router.get("/agent-runs/{run_id}/operator-instructions", response_model=list[OperatorInstructionRead])
def operator_instructions(run_id: str, service: AgentService = Depends(agent_service)):
    """当前流程的持续指令，独立于问答分页与检查点摘要。"""
    return service.operator_instructions(run_id)


@router.post("/agent-runs/{run_id}/operator-instructions/{key}/revoke")
def revoke_operator_instruction(run_id: str, key: str, data: OperatorInstructionRevoke, service: AgentService = Depends(agent_service)):
    """只撤销操作者看到的指令版本，不覆盖后来提交的新要求。"""
    return service.revoke_operator_instruction(run_id, key, data.instruction_id)


@router.post("/agent-runs/{run_id}/pause",response_model=AgentRunRead)
def pause_run(run_id: str,service: AgentService=Depends(agent_service)):
    """请求暂停后续编排，当前计算任务仍有自己的时间上限。

    :param run_id: 当前运行。
    :param service: 业务服务。
    :returns: 暂停请求状态。
    """
    return service.control(run_id,"pause")


@router.post("/agent-runs/{run_id}/stop",response_model=AgentRunRead)
def stop_run(run_id: str,service: AgentService=Depends(agent_service)):
    """停止本运行的计算并保留检查点、候选权重与报告。

    :param run_id: 当前运行。
    :param service: 业务服务。
    :returns: 正在停止或已停止的状态。
    """
    return service.control(run_id,"stop")


@router.post("/agent-runs/{run_id}/resume",response_model=AgentRunRead)
def resume_run(run_id: str,data: AgentResume|None=None,service: AgentService=Depends(agent_service)):
    """显式恢复暂停、人审等待或故障，不允许更改冻结预算。

    :param run_id: 业务运行。
    :param data: 人工审核或恢复说明。
    :param service: 业务服务。
    :returns: 等待同一检查点继续的状态。
    """
    return service.control(run_id,"resume",data.model_dump() if data else None)


@router.post("/agent-runs/{run_id}/training-setup",response_model=AgentRunRead)
def setup_training(run_id: str,data: TrainingSetup,service: AgentService=Depends(agent_service)):
    """人审完成后开始训练，首轮参数可选，预算只能收紧。

    :param run_id: 停在开始训练步骤的运行。
    :param data: 首轮参数片段与预算片段。
    :param service: 业务服务。
    :returns: 排队继续的运行。
    """
    return service.setup_training(run_id,data)


@router.get("/agent-runs/{run_id}/events")
def run_events(run_id: str,after_id: int=Query(0,ge=0),limit: int=Query(100,ge=1,le=500),service: AgentService=Depends(agent_service)):
    """按游标读取节点、模型决策、工具结果和人工操作。

    :param run_id: 业务运行。
    :param after_id: 已读取事件 ID。
    :param limit: 本次最多条数。
    :param service: 业务服务。
    :returns: 不包含模型内部思考或凭据的事件。
    """
    service.get(run_id)
    with Session(service.engine) as session:
        rows=list(session.scalars(select(AgentEvent).where(AgentEvent.run_id==run_id,AgentEvent.id>after_id).order_by(AgentEvent.id).limit(limit)))
        return [{"id":row.id,"node":row.node,"event_type":row.event_type,"level":row.level,"payload":row.payload,"created_at":row.created_at} for row in rows]


@router.get("/agent-runs/{run_id}/iterations")
def run_iterations(run_id: str,service: AgentService=Depends(agent_service)):
    """列出每轮父子实验、计划、Optuna 与真实结论。

    :param run_id: 业务运行。
    :param service: 业务服务。
    :returns: 不可覆盖计划及验证结论。
    """
    service.get(run_id)
    with Session(service.engine) as session:
        return [{"iteration":row.number,"experiment_id":row.experiment_id,"plan":row.plan,"optuna_trial_number":row.optuna_trial_number,"evaluation_id":row.evaluation_id,"conclusion":row.conclusion,"created_at":row.created_at} for row in session.scalars(select(AgentIteration).where(AgentIteration.run_id==run_id).order_by(AgentIteration.number))]


@router.get("/agent-runs/{run_id}/report")
def run_report(run_id: str,service: AgentService=Depends(agent_service)):
    """验证指纹后返回最终报告，不把未完成运行伪装成成功。

    :param run_id: 业务运行。
    :param service: 业务服务。
    :returns: 机器报告，生产发布状态独立展示。
    """
    run=service.get(run_id)
    if not run.report_path:
        raise DatasetError(409,"流程尚未生成最终报告")
    path=owned_path(service.settings,run.report_path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=run.report_sha256:
        raise DatasetError(409,"最终报告缺失或指纹变化")
    return json.loads(path.read_text())


@router.get("/agent-runs/{run_id}/report.md",response_class=FileResponse)
def human_report(run_id: str,service: AgentService=Depends(agent_service)):
    """下载固定运行的人工可读报告，原始结论以机器报告指纹为准。

    :param run_id: 已封存运行。
    :param service: 业务服务。
    :returns: docs/reports 中对应的 Markdown，不开放通用文件下载。
    """
    run=service.get(run_id)
    if not run.report_path:
        raise DatasetError(409,"报告尚未生成")
    path=Path("docs/reports")/f"agent-{run.id}.md"
    if not path.is_file():
        raise DatasetError(404,"人工报告文件不存在，请读取机器报告")
    return FileResponse(path,media_type="text/markdown; charset=utf-8",filename=path.name)
