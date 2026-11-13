"""LangGraph 持久编排：前半段固定节点直到人工审核门禁，训练段由模型通过工具循环决策。"""

import json
import operator
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from yolo_agent.agents.context import assemble, build_dossier
from yolo_agent.agents.optimization import stagnation_recommendations
from yolo_agent.agents.planner import SYSTEM_PROMPT, call_model
from yolo_agent.agents.reporting import finalize_run
from yolo_agent.models.annotation import AnnotationRun, DatasetExport
from yolo_agent.models.dataset import utc_now
from yolo_agent.schemas.agent import AgentPolicy
from yolo_agent.services.agents import AgentService
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.services.training import validate_export
from yolo_agent.tools.workflow import CONTROL_TOOLS, WorkflowPaused, WorkflowStopped, WorkflowTools

TOOL_RESULT_CHARS = 12_000


class WorkflowState(TypedDict, total=False):
    """检查点只保存 JSON 业务身份、对话与小型状态，不保存 ORM、模型对象或图像。"""

    run_id: str
    analysis_task: str
    annotation_task: str
    annotation_attempt: int
    upload_attempt: int
    sync_attempt: int
    export_attempt: int
    operation_retry: bool
    messages: Annotated[list[dict], operator.add]
    route: str
    sleep: dict | None
    review: dict | None
    finish: dict | None
    stop_reason: str


class WorkflowNodes:
    """前半段每个逻辑角色对应固定节点，训练段由模型调用受约束的工具。"""

    def __init__(self, service: AgentService, run_id: str):
        """绑定单次运行，避免模型选择其它项目的资源。

        :param service: 控制器资源。
        :param run_id: 图线程与业务运行 ID。
        :returns: 无。
        """
        self.service,self.run_id=service,run_id
        self.tools=WorkflowTools(service,run_id)
        self.policy=AgentPolicy.model_validate(service.get(run_id).policy)

    def _enter(self, node: str, state: WorkflowState):
        self.tools.guard()
        old=self.service.get(self.run_id)
        run=self.service.patch(self.run_id,phase=node,status="running")
        if old.phase!=node:
            self.service.event(self.run_id,node,"entered",{})
        return run

    def _wait(self, kind: str, task_id: str):
        task=self.tools.datasets.get_task(task_id)
        if task.status in {"queued","running","canceling"}:
            payload={"kind":kind,"task_id":task_id}
            self.service.patch(self.run_id,status="waiting",state={"waiting":payload})
            interrupt(payload)
            task=self.tools.datasets.get_task(task_id)
        return task

    def _dataset_inspector(self, state: WorkflowState):
        run=self._enter("dataset_inspector",state)
        if run.dataset_export_id:
            with Session(self.service.engine) as session:
                export=session.get(DatasetExport,run.dataset_export_id)
                manifest=validate_export(self.service.settings,export,verify_files=False)
                root=owned_path(self.service.settings,export.directory)
            counts={}
            for split in ("train","val"):
                count=Counter()
                for file in (root/"labels"/split).glob("*.txt"):
                    for line in file.read_text().splitlines():
                        if line.strip():
                            count[int(line.split()[0])]+=1
                counts[split]={item["name"]:count[item["class_id"]] for item in manifest["classes"]}
            dataset={"fully_human_reviewed":True,"classes":manifest["classes"],"split_counts":manifest["summary"]["split_counts"],"class_counts":counts,"test_labels_read":False,"entry":"reviewed_export"}
            self.service.patch(self.run_id,state={"dataset":dataset})
            self.service.event(self.run_id,"dataset_inspector","reused_immutable_export",dataset)
            return {}
        task=self.tools.analyze()
        return {"analysis_task":task.id}

    def _data_quality(self, state: WorkflowState):
        run=self._enter("data_quality_check",state)
        if run.dataset_export_id:
            return {}
        task=self._wait("dataset",state["analysis_task"])
        if task.status!="completed":
            raise RuntimeError(f"数据分析失败：{task.error}")
        if task.result["valid_images"]<3:
            raise ValueError("有效图片不足，无法建立互斥子集")
        self.service.patch(self.run_id,state={"dataset":{"fully_human_reviewed":False,"analysis":task.result,"test_labels_read":False}})
        return {}

    def _dataset_split(self, state: WorkflowState):
        run=self._enter("dataset_split",state)
        if run.dataset_version_id:
            return {}
        task=self.tools.split()
        task=self._wait("dataset",task.id)
        if task.status!="completed":
            raise RuntimeError(f"分组划分失败：{task.error}")
        self.service.patch(self.run_id,dataset_version_id=task.result["version_id"])
        self.service.event(self.run_id,"dataset_split","version_sealed",task.result)
        return {}

    def _auto_annotation(self, state: WorkflowState):
        run=self._enter("auto_annotation",state)
        if run.dataset_export_id:
            return {}
        annotation,task=self.tools.annotate(state.get("annotation_attempt",0))
        self.service.patch(self.run_id,state={"annotation_run_id":annotation.id})
        return {"annotation_task":task.id}

    def _annotation_quality(self, state: WorkflowState):
        run=self._enter("annotation_quality",state)
        if run.dataset_export_id:
            return {}
        task=self._wait("dataset",state["annotation_task"])
        if task.status!="completed":
            attempt=state.get("annotation_attempt",0)
            if attempt>=self.policy.max_retries:
                raise RuntimeError(f"自动标注重试耗尽：{task.error}")
            self.service.event(self.run_id,"annotation_quality","retry_scheduled",{"failed_task":task.id,"attempt":attempt+1})
            return {"annotation_attempt":attempt+1,"operation_retry":True}
        self.service.event(self.run_id,"annotation_quality","proposals_require_human_review",task.result)
        return {"operation_retry":False}

    def _cvat_upload(self, state: WorkflowState):
        run=self._enter("cvat_upload",state)
        if run.dataset_export_id:
            return {}
        # 每个等待节点只对应一个 interrupt，避免恢复值因条件变化而错配。
        attempt=state.get("upload_attempt",0)
        task=self.tools.annotation_operation("cvat_push",attempt=attempt)
        task=self._wait("dataset",task.id)
        if task.status!="completed":
            if attempt>=self.policy.max_retries:
                raise RuntimeError(f"CVAT 上传重试耗尽：{task.error}")
            return {"upload_attempt":attempt+1,"operation_retry":True}
        return {"operation_retry":False}

    def _cvat_review(self, state: WorkflowState):
        run=self._enter("cvat_review",state)
        if run.dataset_export_id:
            return {}
        if not run.review_confirmation:
            with Session(self.service.engine) as session:
                annotation=session.get(AnnotationRun,run.state["annotation_run_id"])
                task_id=annotation.cvat_task_id
            base_url=self.service.settings.cvat_public_url or self.service.settings.cvat_url
            payload={"kind":"human_review","annotation_run_id":run.state["annotation_run_id"],"cvat_task_id":task_id,"url":f"{base_url.rstrip('/')}/tasks/{task_id}","requires":["acceptance/completed","reviewer","missed_objects_checked"]}
            self.service.patch(self.run_id,status="needs_review",state={"waiting":payload})
            interrupt(payload)
        return {}

    def _annotation_validation(self, state: WorkflowState):
        run=self._enter("annotation_validation",state)
        if run.dataset_export_id:
            return {}
        attempt=state.get("sync_attempt",0)
        task=self.tools.annotation_operation("cvat_pull",attempt=attempt)
        task=self._wait("dataset",task.id)
        if task.status!="completed":
            if attempt>=self.policy.max_retries:
                raise RuntimeError(f"人工回传验证失败：{task.error}")
            return {"sync_attempt":attempt+1,"operation_retry":True}
        self.service.event(self.run_id,"annotation_validation","human_annotations_accepted",task.result)
        return {"operation_retry":False}

    def _reviewed_export(self, state: WorkflowState):
        run=self._enter("reviewed_export",state)
        if run.dataset_export_id:
            return {}
        attempt=state.get("export_attempt",0)
        task=self.tools.annotation_operation("yolo_export",attempt=attempt)
        task=self._wait("dataset",task.id)
        if task.status!="completed":
            if attempt>=self.policy.max_retries:
                raise RuntimeError(f"人审工件导出失败：{task.error}")
            return {"export_attempt":attempt+1,"operation_retry":True}
        with Session(self.service.engine) as session:
            export=session.scalar(select(DatasetExport).where(DatasetExport.run_id==run.state["annotation_run_id"]))
            annotation=session.get(AnnotationRun,export.run_id)
            export_id,schema_id=export.id,annotation.class_schema_id
        self.service.patch(self.run_id,dataset_export_id=export_id,class_schema_id=schema_id)
        self._dataset_inspector(state)
        return {"operation_retry":False}

    def _training_setup(self, state: WorkflowState):
        run=self._enter("training_setup",state)
        # 从原图启动的运行在人审后停下，由操作者确认首轮参数与预算后才开始训练。
        if not run.state.get("training_setup"):
            payload={"kind":"training_setup","dataset_export_id":run.dataset_export_id}
            self.service.patch(self.run_id,status="needs_review",state={"waiting":payload})
            interrupt(payload)
        return {}

    def _stop_reason(self, run, progress: dict, budget: dict, turn: int) -> str | None:
        """模型之外的硬性停止条件：停滞、模型调用上限、训练预算。"""
        if progress["triggered"]:
            message=f"连续 {progress['rounds_without_improvement']} 轮可比较实验未达到明显提升阈值（固定 val mAP50-95 绝对提升 {self.policy.minimum_improvement:g}），已主动停止后续训练。"
            previous=run.state.get("stop_notice")
            if not previous or previous.get("progress")!=progress:
                notice={"code":"stagnation","message":message,"created_at":utc_now(),"progress":progress,"recommendations":stagnation_recommendations(progress.get("latest_issues",[]))}
                self.service.patch(self.run_id,state={"stop_notice":notice})
                self.service.event(self.run_id,"agent","stagnation_detected",notice,level="warning")
            return message
        if turn>=self.policy.max_model_calls:
            return f"模型调用达到本运行上限 {self.policy.max_model_calls} 次"
        if budget["remaining_seconds"]<1 and not self.tools._active_training():
            return "训练用时预算耗尽"
        return None

    def _agent(self, state: WorkflowState):
        run=self._enter("agent",state)
        self.tools.guard("optimize")
        self.tools.sync_iterations()
        history=state.get("messages",[])
        turn=sum(message["role"]=="assistant" for message in history)
        progress,budget=self.tools.stagnation(),self.service.budget(self.run_id)
        self.service.patch(self.run_id,state={"stagnation":progress,"budget":budget,"model_calls":turn,"iteration":budget["experiments"]})
        reason=self._stop_reason(run,progress,budget,turn)
        if reason:
            return {"route":"finish","stop_reason":reason}
        messages,stats=assemble(self.service.settings,SYSTEM_PROMPT,history,build_dossier(self.tools,model_calls=turn))
        message,metadata=call_model(self.service.settings,messages,self.tools.specs(),on_response=lambda metadata:self.service.event(self.run_id,"agent","model_response_received",{"turn":turn,"metadata":metadata,"context":stats}))
        # 调用 ID 统一按轮次与位置改写：重放时据此推导业务身份，也避免兼容服务返回重复 ID。
        for index,call in enumerate(message["tool_calls"]):
            call["id"]=f"t{turn}c{index}"
        self.service.event(self.run_id,"agent","model_decision",{"turn":turn,"content":(message["content"] or "")[:4000],"tool_calls":[{"name":call["function"]["name"],"arguments":call["function"]["arguments"][:2000]} for call in message["tool_calls"]]})
        return {"messages":[message],"route":"act"}

    def _act(self, state: WorkflowState):
        self._enter("agent_tools",state)
        message=state["messages"][-1]
        turn=sum(item["role"]=="assistant" for item in state["messages"])-1
        results,update=[],{"route":"agent"}
        for call in message["tool_calls"]:
            name,raw=call["function"]["name"],call["function"]["arguments"]
            arguments=raw
            try:
                arguments=json.loads(raw or "{}")
                if name in CONTROL_TOOLS:
                    args=self.tools.control_args(name,arguments)
                    if update["route"]!="agent":
                        raise ValueError("一次回复只能包含一个 sleep、request_human_review 或 finish")
                    if name=="sleep":
                        until=(datetime.now(timezone.utc)+timedelta(minutes=args["minutes"])).isoformat()
                        update.update(route="sleep",sleep={"call_id":call["id"],"until":until,**args})
                    elif name=="request_human_review":
                        update.update(route="review",review={"call_id":call["id"],**args})
                    else:
                        update.update(route="finish",finish=args,stop_reason=args["reason"])
                    self.service.event(self.run_id,"agent","tool_call",{"turn":turn,"tool":name,"arguments":arguments,"ok":True,"result":"已接受"})
                    continue
                output,ok=self.tools.dispatch(name,arguments,key=call["id"]),True
            except (WorkflowPaused,WorkflowStopped):
                raise
            except (DatasetError,PermissionError,ValueError,ValidationError) as error:
                # 参数或业务约束错误回给模型修正，不让整个运行失败。
                output,ok={"error":str(error)[:2000]},False
            images=output.pop("_images",[])
            content=json.dumps(output,ensure_ascii=False,default=str)
            if len(content)>TOOL_RESULT_CHARS:
                content=content[:TOOL_RESULT_CHARS]+"…（结果过长已截断，可用 get_experiment 等工具分项读取）"
            results.append({"role":"tool","tool_call_id":call["id"],"content":content,**({"_images":images} if images else {})})
            self.service.event(self.run_id,"agent","tool_call",{"turn":turn,"tool":name,"arguments":arguments,"ok":ok,"result":content[:2000]})
        return {"messages":results,**update}

    def _sleep(self, state: WorkflowState):
        self._enter("agent_sleep",state)
        sleep=state["sleep"]
        payload={"kind":"sleep","until":sleep["until"],"minutes":sleep["minutes"],"reason":sleep["reason"],"watch":self.tools.active_task_refs()}
        self.service.patch(self.run_id,status="waiting",state={"waiting":payload})
        woke=interrupt(payload) or {}
        result={"woke_at":utc_now(),"reason":woke.get("reason") or "到达设定的唤醒时间"}
        self.service.event(self.run_id,"agent","woke",result)
        return {"messages":[{"role":"tool","tool_call_id":sleep["call_id"],"content":json.dumps(result,ensure_ascii=False)}],"sleep":None}

    def _review(self, state: WorkflowState):
        run=self._enter("agent_review",state)
        review=state["review"]
        acknowledgement=run.state.get("human_acknowledgement") or {}
        if acknowledgement.get("call_id")!=review["call_id"]:
            payload={"kind":"agent_review","call_id":review["call_id"],"reason":review["reason"],"data_actions":review["data_actions"],"message":"Agent 请求人工处理，恢复前需要操作者确认"}
            self.service.patch(self.run_id,status="needs_review",state={"waiting":payload,"review_request":payload})
            self.service.event(self.run_id,"agent","human_review_requested",payload,level="warning")
            interrupt(payload)
            acknowledgement=self.service.get(self.run_id).state.get("human_acknowledgement") or {}
        result={"operator":{key:acknowledgement.get(key) for key in ("reviewer","reason")}}
        return {"messages":[{"role":"tool","tool_call_id":review["call_id"],"content":json.dumps(result,ensure_ascii=False)}],"review":None}

    def _best_model(self, state: WorkflowState):
        self._enter("best_model_selection",state)
        self.tools.guard("compare")
        # 结束前收敛本运行仍在进行的训练，避免运行封存后继续占用训练槽。
        for task in self.tools._active_training():
            try:
                self.tools.training.cancel(task.id)
            except DatasetError:
                pass
        self.service.patch(self.run_id,state={"agent_finish":state.get("finish")})
        return {}

    def _final_report(self, state: WorkflowState):
        self._enter("final_report",state)
        finalize_run(self.service,self.run_id,state.get("stop_reason") or "Agent 决定结束")
        return {}

    def compile(self, checkpointer):
        """建立 StateGraph，并由官方检查点器保存每个节点状态。

        :param checkpointer: 官方 SqliteSaver。
        :returns: 可同步推进、可中断恢复的编译图。
        """
        graph=StateGraph(WorkflowState)
        nodes={"dataset_inspector":self._dataset_inspector,"data_quality":self._data_quality,"dataset_split":self._dataset_split,"auto_annotation":self._auto_annotation,"annotation_quality":self._annotation_quality,"cvat_upload":self._cvat_upload,"cvat_review":self._cvat_review,"annotation_validation":self._annotation_validation,"reviewed_export":self._reviewed_export,"training_setup":self._training_setup,"agent":self._agent,"agent_tools":self._act,"agent_sleep":self._sleep,"agent_review":self._review,"best_model":self._best_model,"final_report":self._final_report}
        for name,node in nodes.items():
            if name=="agent":
                graph.add_node(name,node,retry_policy=RetryPolicy(max_attempts=self.policy.max_retries+1,initial_interval=1,max_interval=5,retry_on=(APIConnectionError,APITimeoutError,InternalServerError,RateLimitError,ValueError)))
            else:
                graph.add_node(name,node)
        edges=[(START,"dataset_inspector"),("dataset_inspector","data_quality"),("data_quality","dataset_split"),("dataset_split","auto_annotation"),("auto_annotation","annotation_quality"),("cvat_review","annotation_validation"),("training_setup","agent"),("agent_sleep","agent"),("agent_review","agent"),("best_model","final_report"),("final_report",END)]
        for left,right in edges:
            graph.add_edge(left,right)
        graph.add_conditional_edges("annotation_quality",lambda state:state.get("operation_retry",False),{True:"auto_annotation",False:"cvat_upload"})
        graph.add_conditional_edges("cvat_upload",lambda state:state.get("operation_retry",False),{True:"cvat_upload",False:"cvat_review"})
        graph.add_conditional_edges("annotation_validation",lambda state:state.get("operation_retry",False),{True:"annotation_validation",False:"reviewed_export"})
        graph.add_conditional_edges("reviewed_export",lambda state:state.get("operation_retry",False),{True:"reviewed_export",False:"training_setup"})
        graph.add_conditional_edges("agent",lambda state:state["route"],{"act":"agent_tools","finish":"best_model"})
        graph.add_conditional_edges("agent_tools",lambda state:state["route"],{"agent":"agent","sleep":"agent_sleep","review":"agent_review","finish":"best_model"})
        compiled=graph.compile(checkpointer=checkpointer)
        # 视觉推理较慢，单步时限覆盖模型超时乘以重试次数。
        compiled.step_timeout=self.service.settings.llm_timeout_seconds*(self.policy.max_retries+1)+180
        return compiled
