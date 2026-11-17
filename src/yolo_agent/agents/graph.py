"""LangGraph 持久编排：前半段固定节点直到人工审核门禁，训练段由模型通过工具循环决策。"""

import json
import operator
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, RetryPolicy, interrupt
from langchain_core.messages import convert_to_openai_messages
from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, RateLimitError
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from yolo_agent.agents.context import MAX_IMAGES, assemble, build_dossier
from yolo_agent.agents.context_memory import ContextMemory, context_signature
from yolo_agent.agents.optimization import stagnation_recommendations
from yolo_agent.agents.planner import SYSTEM_PROMPT, call_model, summarize_context
from yolo_agent.agents.reporting import finalize_run
from yolo_agent.models.annotation import AnnotationRun, DatasetExport
from yolo_agent.models.dataset import utc_now
from yolo_agent.schemas.agent import AgentPolicy
from yolo_agent.services.agents import AgentService
from yolo_agent.services.agent_skills import SKILL_RESULT_CHARS, SKILL_TOOLS
from yolo_agent.services.datasets import DatasetError, owned_path
from yolo_agent.services.context_budget import ContextBudgetError, budget_for_run, context_profile, context_length_rejected, images_rejected
from yolo_agent.services.training import validate_export
from yolo_agent.tools.workflow import CONTROL_TOOLS, WorkflowPaused, WorkflowStopped, WorkflowTools

TOOL_RESULT_CHARS = 6_000


class WorkflowState(TypedDict, total=False):
    """检查点只保存 JSON 业务身份、对话与小型状态，不保存 ORM、模型对象或图像。"""

    run_id: str
    context_memory: dict
    context_ready: str
    context_compacting: bool
    context_image_limit: int
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
        setup=run.state.get("training_setup") or {}
        preparation=run.state.get("workflow_mode")=="preparation" or bool(run.state.get("annotation_run_id"))
        authorized=bool(setup) and (not preparation or (
            setup.get("source")=="operator" and setup.get("dataset_export_id")==run.dataset_export_id))
        if not authorized:
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

    def _record_response(self, metadata, stats, *, turn, kind):
        self.service.event(self.run_id, "agent", "model_response_received", {"turn": turn, "kind": kind, "metadata": metadata, "context": stats})
        usage = metadata.get("usage") or {}
        actual = usage.get("prompt_tokens") or usage.get("input_tokens")
        current = self.service.get(self.run_id)
        budget = budget_for_run(self.service.settings, current.state)
        estimated = stats.get("estimated_input_tokens", 0)
        scale = budget.scale * actual / estimated * 1.03 if actual and estimated and actual > estimated else budget.scale
        self.service.patch(self.run_id, state={"context_profile": context_profile(self.service.settings), "token_estimation_scale": scale,
            "context_usage": {**stats, "actual_input_tokens": actual, "kind": kind}})

    def _context_overflow(self):
        run = self.service.get(self.run_id)
        budget = budget_for_run(self.service.settings, run.state)
        if budget.window_factor <= 0.1:
            raise ContextBudgetError("模型服务在最小有效窗口下仍拒绝上下文长度，请核对该模型的实际窗口配置；已保留历史、持续指令和图片能力。")
        self.service.patch(self.run_id, state={"context_profile": context_profile(self.service.settings), "context_window_factor": max(0.1, budget.window_factor * 0.7)})
        self.service.event(self.run_id, "agent", "context_window_rejected", {"action": "缩小有效窗口并重新摘要，保留图片能力和操作者指令"}, level="warning")

    def _prepare_context(self, state: WorkflowState):
        run = self._enter("agent", state)
        self.tools.guard("optimize")
        settings = self.service.settings
        manager = ContextMemory(settings, run.state)

        def summary_model(messages):
            request = convert_to_openai_messages(messages)
            estimate = manager.budget.check(request)
            turn = self.service.reserve_model_call(self.run_id, kind="summary", reserve_decision=True)
            try:
                return summarize_context(settings, messages, on_response=lambda metadata: self._record_response(metadata,
                    {**manager.budget.info(), "estimated_input_tokens": estimate, "images": 0}, turn=turn, kind="summary"))
            except BadRequestError as error:
                if context_length_rejected(error):
                    self._context_overflow()
                    raise ValueError("摘要请求超过模型服务的实际窗口，保留原始历史并缩小预算后重试") from None
                raise

        try:
            history, memory, stats = manager.compact_once(state.get("messages", []), build_dossier(self.tools, model_calls=run.state.get("model_calls", 0)),
                self.tools.specs(), SYSTEM_PROMPT, state.get("context_memory"), summary_model,
                may_summarize=run.state.get("model_calls", 0) + 2 <= self.policy.max_model_calls,
                force=state.get("context_compacting", False), include_images=not run.state.get("image_inputs_rejected", False))
        except ContextBudgetError as error:
            return {"route": "finish", "stop_reason": str(error)}
        fresh = self.service.get(self.run_id)
        more = stats["needs_more"] and fresh.state.get("model_calls", 0) + 2 <= self.policy.max_model_calls
        if stats["estimated_input_tokens"] > manager.budget.input_limit and not more:
            return {"route": "finish", "stop_reason": "上下文仍超出安全窗口，模型调用预算不足以继续摘要；原始对话与持续指令已保留。"}
        self.service.patch(self.run_id, state={"context_usage": stats})
        if stats["compacted"]:
            self.service.event(self.run_id, "agent", "context_compacted", {key: stats[key] for key in ("estimated_input_tokens", "history_units", "kept_units", "compactions", "summarized_units")})
        return {"messages": Overwrite(history), "context_memory": memory, "context_image_limit": stats["image_limit"],
            "context_compacting": more, "context_ready": context_signature(history, settings, fresh.state), "route": "context" if more else "agent"}

    def _agent(self, state: WorkflowState):
        run=self._enter("agent",state)
        self.tools.guard("optimize")
        self.tools.sync_iterations()
        history=state.get("messages",[])
        # 调用预算独立于被裁剪的历史，超时与格式错误的尝试也计入持久预算。
        turn=max(int(run.state.get("model_calls", 0)), sum(message["role"]=="assistant" for message in history))
        progress,budget=self.tools.stagnation(),self.service.budget(self.run_id)
        self.service.patch(self.run_id,state={"stagnation":progress,"budget":budget,"model_calls":turn,"iteration":budget["experiments"]})
        reason=self._stop_reason(run,progress,budget,turn)
        if reason:
            return {"route":"finish","stop_reason":reason}
        if run.state.get("tool_error_streak", 0) >= 3:
            return {"route":"finish","stop_reason":"连续三轮工具调用未成功，已停止自动尝试并保留结果，请检查工具错误记录"}
        if state.get("context_ready") != context_signature(history, self.service.settings, run.state):
            return {"route": "context"}
        specs = self.tools.specs()
        try:
            messages,stats=assemble(self.service.settings,SYSTEM_PROMPT,history,build_dossier(self.tools,model_calls=turn), include_images=not run.state.get("image_inputs_rejected",False),
                summary=(state.get("context_memory") or {}).get("summary", ""), tools=specs,
                budget=budget_for_run(self.service.settings, run.state), image_limit=state.get("context_image_limit", MAX_IMAGES))
        except ContextBudgetError:
            return {"route": "context", "context_compacting": True}
        stats["compactions"] = (state.get("context_memory") or {}).get("compactions", 0)
        turn = self.service.reserve_model_call(self.run_id, kind="decision")
        try:
            message,metadata=call_model(self.service.settings,messages,specs,on_response=lambda metadata:self._record_response(metadata, stats, turn=turn, kind="decision"))
        except BadRequestError as error:
            if context_length_rejected(error):
                try:
                    self._context_overflow()
                except ContextBudgetError as error:
                    return {"route": "finish", "stop_reason": str(error)}
                raise ValueError("模型服务拒绝本次上下文长度，将缩小窗口并重新摘要；图片能力保留") from None
            if stats["images"] and images_rejected(error):
                self.service.patch(self.run_id,state={"image_inputs_rejected":True})
                self.service.event(self.run_id,"agent","image_request_rejected",{"action":"移除图像输入，继续使用结构化指标和混淆矩阵证据"},level="warning")
                raise ValueError("模型拒绝带图请求，下一次有限重试将使用结构化训练档案") from None
            raise
        # 调用 ID 统一按轮次与位置改写：重放时据此推导业务身份，也避免兼容服务返回重复 ID。
        for index,call in enumerate(message["tool_calls"]):
            call["id"]=f"t{turn}c{index}"
        self.service.event(self.run_id,"agent","model_decision",{"turn":turn,"content":(message["content"] or "")[:4000],"tool_calls":[{"name":call["function"]["name"],"arguments":call["function"]["arguments"][:2000]} for call in message["tool_calls"]]})
        return {"messages":[message],"route":"act"}

    def _act(self, state: WorkflowState):
        self._enter("agent_tools",state)
        message=state["messages"][-1]
        turn=max(0,int(self.service.get(self.run_id).state.get("model_calls", 1))-1)
        results,outcomes,update=[],[],{"route":"agent"}
        for call in message["tool_calls"]:
            name,raw=call["function"]["name"],call["function"]["arguments"]
            arguments=raw
            try:
                arguments=json.loads(raw or "{}")
                if name in CONTROL_TOOLS:
                    args=self.tools.control_args(name,arguments)
                    if self.service.operator_messages(self.run_id, pending_only=True):
                        raise ValueError("有未回答的操作者问题，请先读取所需证据并调用 answer_operator 回答，再休眠、请求人工处理或结束")
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
                    outcomes.append(True)
                    continue
                output,ok=self.tools.dispatch(name,arguments,key=call["id"]),True
            except (WorkflowPaused,WorkflowStopped):
                raise
            except (DatasetError,PermissionError,ValueError,ValidationError) as error:
                # 参数或业务约束错误回给模型修正，不让整个运行失败。
                output,ok={"error":str(error)[:2000]},False
            images=output.pop("_images",[])
            outcomes.append(ok)
            content=json.dumps(output,ensure_ascii=False,default=str)
            result_limit = SKILL_RESULT_CHARS if name in SKILL_TOOLS else TOOL_RESULT_CHARS
            if len(content)>result_limit:
                content=json.dumps({"result_too_large": True, "tool": name, "characters": len(content),
                    "message": "结果超过单次阅读上限，未返回不完整 JSON。请减小 limit、分批 names/parameter_names，或指定 section、class_names、metric_names 后重新读取。",
                    "available_fields": list(output)}, ensure_ascii=False)
                images=[]
            results.append({"role":"tool","tool_call_id":call["id"],"content":content,**({"_images":images} if images else {})})
            self.service.event(self.run_id,"agent","tool_call",{"turn":turn,"tool":name,"arguments":arguments,"ok":ok,"result":content[:2000]})
        failures = bool(outcomes) and not any(outcomes)
        run = self.service.get(self.run_id)
        self.service.patch(self.run_id,state={"tool_error_streak":run.state.get("tool_error_streak",0)+1 if failures else 0})
        return {"messages": results, **update}

    def _sleep(self, state: WorkflowState):
        self._enter("agent_sleep",state)
        sleep=state["sleep"]
        payload={"kind":"sleep","until":sleep["until"],"minutes":sleep["minutes"],"reason":sleep["reason"],"watch":self.tools.active_task_refs()}
        self.service.patch(self.run_id,status="waiting",state={"waiting":payload})
        woke=interrupt(payload) or {}
        result={"woke_at":utc_now(),"reason":woke.get("reason") or "到达设定的唤醒时间"}
        request = self.service.wake_request(self.run_id, sleep["until"])
        if request:
            result["operator_request"] = request
            result["reason"] = "操作者手动唤醒"
        self.service.patch(self.run_id, state={"waiting": {}})
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
        pending=self.tools.active_task_refs()
        if pending:
            # finish 可能与 export_model 同轮提交，必须等文件校验或停止真正结束后再封存报告。
            payload={"kind":"finish_tasks","watch":pending}
            self.service.patch(self.run_id,status="waiting",state={"waiting":payload})
            interrupt(payload)
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
        nodes["agent_context"] = self._prepare_context
        for name,node in nodes.items():
            if name in {"agent", "agent_context"}:
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
        graph.add_conditional_edges("agent",lambda state:state["route"],{"act":"agent_tools","context":"agent_context","finish":"best_model"})
        graph.add_conditional_edges("agent_context",lambda state:state["route"],{"agent":"agent","context":"agent_context","finish":"best_model"})
        graph.add_conditional_edges("agent_tools",lambda state:state["route"],{"agent":"agent","sleep":"agent_sleep","review":"agent_review","finish":"best_model"})
        compiled=graph.compile(checkpointer=checkpointer)
        # 视觉推理较慢，单步时限覆盖模型超时乘以重试次数。
        compiled.step_timeout=self.service.settings.llm_timeout_seconds*(self.policy.max_retries+1)+180
        return compiled
