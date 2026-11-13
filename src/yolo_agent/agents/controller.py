"""有租约的独立控制进程，恢复官方检查点，不在 API 中运行图。"""

import logging
import signal
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from rq.command import send_stop_job_command
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.serializers import JSONSerializer
from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from yolo_agent.agents.graph import WorkflowNodes
from yolo_agent.agents.reporting import finalize_run
from yolo_agent.models.agent import AgentIteration, AgentRun
from yolo_agent.models.database import create_database_engine
from yolo_agent.models.dataset import DatasetTask, utc_now
from yolo_agent.models.evaluation import EvaluationTask
from yolo_agent.models.experiment import TrainingTask
from yolo_agent.services.agents import AgentService
from yolo_agent.services.datasets import fail_dataset_task
from yolo_agent.services.evaluation import fail_evaluation
from yolo_agent.settings import Settings
from yolo_agent.tools.workflow import WorkflowPaused, WorkflowStopped, WorkflowTools
from yolo_agent.utils.logging import configure_logging
from yolo_agent.workers.queue import redis_connection

logger=logging.getLogger(__name__)


def _time_after(seconds: float) -> str:
    return (datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat()


class AgentController:
    """计算等待先查持久状态，仅就绪时恢复图，避免空转和重复副作用。"""

    def __init__(self, settings: Settings):
        """为当前控制进程建立资源和唯一租约身份。

        :param settings: 轮询、模型调用与持久化配置。
        :returns: 无。
        """
        self.settings=settings
        self.engine=create_database_engine(settings)
        self.service=AgentService(self.engine,settings)
        self.owner=str(uuid4())

    def _training_tasks(self, run_id: str):
        with Session(self.engine) as session:
            return list(session.scalars(select(TrainingTask).join(AgentIteration,AgentIteration.experiment_id==TrainingTask.experiment_id).where(AgentIteration.run_id==run_id,TrainingTask.status.in_(["queued","running","canceling"]))))

    def _cancel_training(self, run_id: str) -> bool:
        tools=WorkflowTools(self.service,run_id)
        tasks=self._training_tasks(run_id)
        for task in tasks:
            current=tools.training.get_task(task.id)
            if current.status in {"queued","running","canceling"}:
                tools.training.cancel(task.id)
        return bool(self._training_tasks(run_id))

    def _cancel_pending(self, run_id: str) -> bool:
        run=self.service.get(run_id)
        waiting=run.state.get("waiting",{})
        kind,task_id=waiting.get("kind"),waiting.get("task_id")
        if kind not in {"dataset","evaluation"} or not task_id:
            return False
        with Session(self.engine) as session:
            task=session.get(DatasetTask if kind=="dataset" else EvaluationTask,task_id)
            if task is None or task.status not in {"queued","running"}:
                return False
            if kind=="evaluation":
                owned=session.scalar(select(AgentIteration.run_id).where(AgentIteration.run_id==run_id,AgentIteration.experiment_id==task.experiment_id))
            else:
                tools=WorkflowTools(self.service,run_id)
                owned=task.dataset_id==run.dataset_id and (task.id in {tools.identity("analysis"),tools.identity("split")} or task.parameters.get("run_id")==run.state.get("annotation_run_id"))
            if not owned:
                raise PermissionError("不能取消非本运行提交的任务")
        with redis_connection(self.settings) as connection:
            try:
                job=Job.fetch(task_id,connection=connection,serializer=JSONSerializer)
                if job.get_status(refresh=True)=="started":
                    send_stop_job_command(connection,task_id)
                    return True
                job.cancel()
            except NoSuchJobError:
                pass
        if kind=="dataset":
            fail_dataset_task(self.engine,task_id,"操作者停止 Agent 流程")
        else:
            fail_evaluation(self.engine,task_id,"操作者停止 Agent 流程")
        return False

    def _ready(self, run_id: str, payload: dict) -> bool | str:
        """判断中断是否可以恢复；休眠类返回唤醒原因，交给模型作为工具结果。"""
        kind=payload.get("kind")
        tools=WorkflowTools(self.service,run_id)
        if kind=="dataset":
            task=tools.datasets.get_task(payload["task_id"])
            if task.status=="queued" and (datetime.now(timezone.utc)-datetime.fromisoformat(task.created_at)).total_seconds()>self.settings.agent_queue_wait_seconds:
                self._cancel_pending(run_id)
                raise TimeoutError("任务排队超过等待上限，已撤销计算，等待人工检查 Worker")
            return task.status not in {"queued","running","canceling"}
        if kind=="sleep":
            getters={"training":(tools.training.get_task,"训练"),"evaluation":(tools.evaluation.get,"评估"),"export":(tools.exports.get,"导出")}
            for ref in payload.get("watch",[]):
                getter,label=getters[ref["kind"]]
                task=getter(ref["id"])
                if task.status not in {"queued","running","canceling"}:
                    return f"{label}任务 {ref['id'][:8]} 已结束，状态 {task.status}"
            return "到达设定的唤醒时间" if datetime.fromisoformat(payload["until"])<=datetime.now(timezone.utc) else False
        run=self.service.get(run_id)
        if kind=="human_review":
            return bool(run.review_confirmation)
        if kind=="training_setup":
            return bool(run.state.get("training_setup"))
        if kind=="agent_review":
            return "review_request" not in run.state
        raise ValueError("未知图中断类型")

    def advance(self, run_id: str) -> None:
        """推进一个已获得租约的业务运行，直到完成或下一次中断。

        :param run_id: 唯一图线程。
        :returns: 无；任务、等待或最终状态均持久化。
        """
        run=self.service.get(run_id)
        budget=self.service.budget(run_id)
        active=self._training_tasks(run_id)
        if run.stop_requested or (budget["remaining_seconds"]<1 and active):
            reason="操作者停止" if run.stop_requested else "训练用时预算耗尽"
            training_wait=self._cancel_training(run_id)
            other_wait=self._cancel_pending(run_id)
            if training_wait or other_wait:
                self.service.patch(run_id,status="canceling",stop_requested=True,state={"stop_reason":reason})
                return
            finalize_run(self.service,run_id,run.state.get("stop_reason",reason),status="stopped")
            return
        if run.pause_requested:
            self.service.patch(run_id,status="paused")
            return
        if run.status=="paused":
            return
        if not run.started_at:
            self.service.patch(run_id,started_at=utc_now())
        path=self.settings.storage_dir.resolve()/"agents"/"checkpoints.sqlite"
        path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(path,check_same_thread=False,timeout=5)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            saver=SqliteSaver(connection)
            graph=WorkflowNodes(self.service,run_id).compile(saver)
            config={"configurable":{"thread_id":run_id},"recursion_limit":1000}
            snapshot=graph.get_state(config)
            if snapshot.interrupts:
                payload=snapshot.interrupts[0].value
                ready=self._ready(run_id,payload)
                if not ready:
                    return
                input_value=Command(resume={"ready":True,"reason":ready if isinstance(ready,str) else None})
                self.service.event(run_id,self.service.get(run_id).phase,"checkpoint_resumed",{"interrupt":payload,"checkpoint_id":snapshot.config["configurable"].get("checkpoint_id")})
            else:
                input_value=None if snapshot.values else {"run_id":run_id,"iteration":0}
            graph.invoke(input_value,config,durability="sync")
            snapshot=graph.get_state(config)
            self.service.patch(run_id,state={"graph_next":list(snapshot.next),"checkpoint_id":snapshot.config["configurable"].get("checkpoint_id")})

    def tick(self) -> int:
        """以数据库条件更新领取到期工作，租约失效前其它控制器不可接管。

        :returns: 本次检查的运行数量。
        """
        now=utc_now()
        with Session(self.engine) as session:
            ids=list(session.scalars(select(AgentRun.id).where(AgentRun.status.in_(["queued","running","waiting","canceling","paused"]),or_(AgentRun.next_poll_at.is_(None),AgentRun.next_poll_at<=now),or_(AgentRun.lease_until.is_(None),AgentRun.lease_until<=now)).order_by(AgentRun.created_at).limit(20)))
        processed=0
        for run_id in ids:
            with Session(self.engine) as session,session.begin():
                result=session.execute(update(AgentRun).where(AgentRun.id==run_id,or_(AgentRun.lease_until.is_(None),AgentRun.lease_until<=utc_now())).values(lease_owner=self.owner,lease_until=_time_after(300)))
                claimed=result.rowcount==1
            if not claimed:
                continue
            processed+=1
            try:
                self.advance(run_id)
            except WorkflowPaused as error:
                self.service.patch(run_id,status="paused",pause_requested=True,state={"pause_reason":str(error)})
                self.service.event(run_id,"operator","paused",{"reason":str(error)})
            except WorkflowStopped as error:
                self.service.patch(run_id,status="canceling",stop_requested=True,state={"stop_reason":str(error)})
            except Exception as error:
                run=self.service.get(run_id)
                self.service.patch(run_id,status="failed",error=f"{type(error).__name__}: {error}"[:4000],retry_count=run.retry_count+1)
                self.service.event(run_id,run.phase,"failed",{"error_type":type(error).__name__,"message":str(error)[:4000]},level="error")
                logger.exception("Agent 节点失败，保留检查点等待人工恢复",extra={"task_id":run_id})
            finally:
                with Session(self.engine) as session,session.begin():
                    session.execute(update(AgentRun).where(AgentRun.id==run_id,AgentRun.lease_owner==self.owner).values(lease_owner=None,lease_until=None,next_poll_at=_time_after(self.settings.agent_poll_seconds)))
        return processed

    def close(self) -> None:
        """释放当前进程连接，图检查点与计算任务不会被删除。

        :returns: 无。
        """
        self.engine.dispose()


def run_controller(settings: Settings) -> None:
    """运行可由 Supervisor/systemd 管理的控制器。

    :param settings: 项目独立环境配置。
    :returns: 收到停止信号后释放租约并退出。
    """
    import os
    os.environ["LANGSMITH_TRACING"]="false"
    os.environ["LANGCHAIN_TRACING_V2"]="false"
    configure_logging(settings,"agent-controller")
    stop=threading.Event()
    for name in (signal.SIGTERM,signal.SIGINT):
        signal.signal(name,lambda *_:stop.set())
    controller=AgentController(settings)
    try:
        while not stop.is_set():
            controller.tick()
            stop.wait(settings.agent_poll_seconds)
    finally:
        controller.close()
