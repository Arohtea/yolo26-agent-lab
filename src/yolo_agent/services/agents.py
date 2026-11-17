"""Agent 运行入口、人工控制、冻结预算与可查询审计。"""

from datetime import datetime, timezone

import yaml

from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from yolo_agent.models.agent import AgentEvent, AgentIteration, AgentRun
from yolo_agent.models.annotation import AnnotationRun, ClassSchema, DatasetExport
from yolo_agent.models.dataset import Dataset, DatasetVersion, utc_now
from yolo_agent.models.experiment import TrainingTask
from yolo_agent.schemas.agent import AgentPolicy, AgentRunCreate, OperatorInstructionChange, TrainingSetup
from yolo_agent.schemas.experiment import TrainingParameters
from yolo_agent.services.datasets import DatasetError
from yolo_agent.services.training import validate_export
from yolo_agent.settings import Settings


def fit_initial(settings: Settings, parameters: dict, explicit: set[str], policy: AgentPolicy) -> dict:
    """操作者未填写的默认值收进预算与搜索空间交给模型调整，填写的值越界则直接拒绝。

    :param settings: 搜索空间配置位置。
    :param parameters: 合并后的完整首轮参数。
    :param explicit: 操作者明确填写的字段。
    :param policy: 本次运行预算。
    :returns: 可直接作为首轮起点的参数。
    """
    space=yaml.safe_load(settings.optimization_config_path.read_text())["search_space"]
    sizes=[int(value) for value in space["imgsz"] if int(value)<=policy.max_imgsz and int(value)%32==0] or [policy.max_imgsz]
    try:
        values=TrainingParameters.model_validate(parameters).model_dump()
    except ValueError as error:
        raise DatasetError(422,f"首轮参数不合法：{error}") from error
    if "model" not in explicit and values["model"] not in policy.allowed_models:
        values["model"]=policy.allowed_models[0]
    if "epochs" not in explicit:
        values["epochs"]=min(values["epochs"],policy.max_epochs_per_experiment)
    if "imgsz" not in explicit:
        feasible = []
        for size in sizes:
            try:
                candidate = TrainingParameters.model_validate({**values, "imgsz": size})
            except ValueError:
                continue
            if candidate.max_training_imgsz() <= policy.max_imgsz:
                feasible.append(size)
        if not feasible:
            raise DatasetError(422,"当前 multi_scale 在输入尺寸预算内没有可用的搜索尺寸")
        if values["imgsz"] not in feasible:
            values["imgsz"]=max(feasible)
    for name in ("optimizer","mosaic"):
        if name not in explicit and values[name] not in space[name]:
            values[name]=space[name][0]
    for name in ("lr0","weight_decay"):
        if name not in explicit:
            values[name]=min(max(values[name],space[name]["low"]),space[name]["high"])
    problems=[name for name,bad in {"model":values["model"] not in policy.allowed_models,"epochs":values["epochs"]>policy.max_epochs_per_experiment,"imgsz":values["imgsz"] not in sizes,"optimizer":values["optimizer"] not in space["optimizer"],"mosaic":values["mosaic"] not in space["mosaic"],"lr0":not space["lr0"]["low"]<=values["lr0"]<=space["lr0"]["high"],"weight_decay":not space["weight_decay"]["low"]<=values["weight_decay"]<=space["weight_decay"]["high"]}.items() if bad]
    if problems:
        raise DatasetError(422,f"首轮参数超出本次预算或搜索空间：{', '.join(problems)}")
    if TrainingParameters.model_validate(values).max_training_imgsz() > policy.max_imgsz:
        raise DatasetError(422,"multi_scale 实际最大训练尺寸超过本次输入尺寸预算，请降低 imgsz 或 multi_scale")
    return values


class AgentService:
    """控制面不执行模型训练，所有控制操作都留下事件。"""

    def __init__(self, engine: Engine, settings: Settings):
        """绑定当前进程资源。

        :param engine: 业务数据库。
        :param settings: 模型与存储配置。
        :returns: 无。
        """
        self.engine, self.settings = engine, settings

    def create(self, data: AgentRunCreate) -> AgentRun:
        """从真实数据归属推导项目，冻结一次自主运行预算。

        :param data: 已导入数据或已审核工件及操作者限制。
        :returns: 等待控制器执行的运行记录。
        """
        if not self.settings.llm_model or not self.settings.llm_api_key.get_secret_value():
            raise DatasetError(503,"请先配置兼容模型服务")
        if data.dataset_id:
            from yolo_agent.services.annotators import validate_source_plan

            with Session(self.engine) as session:
                schema = session.get(ClassSchema, data.class_schema_id)
                dataset = session.get(Dataset, data.dataset_id)
                if schema is None or dataset is None or dataset.deleted_at or schema.project_id != dataset.project_id:
                    raise DatasetError(422,"数据与类别体系必须存在且属于同一项目")
                validate_source_plan(session, self.settings, dataset.project_id, schema.classes, data.annotation_sources, coco_model=data.annotation_coco_model)
        with Session(self.engine,expire_on_commit=False) as session, session.begin():
            export = session.get(DatasetExport,data.dataset_export_id) if data.dataset_export_id else None
            version = session.get(DatasetVersion,export.version_id) if export else None
            if data.dataset_export_id and (export is None or version is None):
                raise DatasetError(404,"审核后工件不存在")
            if export:
                validate_export(self.settings,export,verify_files=False)
                annotation = session.get(AnnotationRun,export.run_id)
                if annotation.status!="exported" or not version.sealed:
                    raise DatasetError(409,"数据尚未封存")
                schema_id = annotation.class_schema_id
            else:
                schema_id = data.class_schema_id
            dataset = session.get(Dataset,version.dataset_id if version else data.dataset_id)
            schema = session.get(ClassSchema,schema_id) if schema_id else None
            if dataset is None or dataset.deleted_at or schema is None or schema.project_id!=dataset.project_id:
                raise DatasetError(422,"数据与类别体系必须存在且属于同一项目")
            # 操作者明确填写的参数只约束首轮，之后由模型调整；未填写的项从一开始就由模型决定。
            fields = sorted(data.initial_parameters.model_fields_set)
            state = {"workflow_mode":"training" if export else "preparation","operator_fields":fields,"annotation_sources":data.model_dump(mode="json")["annotation_sources"],"annotation_box_threshold":data.annotation_box_threshold,"annotation_coco_model":data.annotation_coco_model}
            # 只有显式从已审训练数据创建的运行才授权训练；原图入口始终停在训练确认处。
            if export:
                state["training_setup"] = {"at":utc_now(),"source":"create","dataset_export_id":export.id}
            initial = fit_initial(self.settings,data.initial_parameters.model_dump(),set(fields),data.policy)
            run = AgentRun(project_id=dataset.project_id,dataset_id=dataset.id,dataset_version_id=version.id if version else None,class_schema_id=schema_id,dataset_export_id=export.id if export else None,name=data.name,requirement=data.requirement,policy=data.policy.model_dump(),initial_parameters=initial,state=state)
            session.add(run)
            session.flush()
            session.add(AgentEvent(run_id=run.id,node="operator",event_type="created",payload={"policy":run.policy,"entry":"reviewed_export" if export else "raw_dataset"}))
            if export:
                # 新训练独立管理；关闭已交接的准备流程，避免留下重复的“等待开始训练”。
                for prepared in session.scalars(select(AgentRun).where(AgentRun.id != run.id,
                    AgentRun.dataset_export_id == export.id, AgentRun.phase == "training_setup",
                    AgentRun.status == "needs_review")):
                    if prepared.state.get("workflow_mode") == "preparation" or prepared.state.get("annotation_run_id"):
                        prepared.status, prepared.finished_at = "completed", utc_now()
                        prepared.state = {**prepared.state, "preparation_complete": True,
                            "training_successor_id": run.id, "waiting": {}}
                        session.add(AgentEvent(run_id=prepared.id,node="training_setup",event_type="data_prepared",
                            payload={"training_run_id":run.id,"message":"数据准备已完成，训练由新的独立流程管理"}))
        return run

    def get(self, run_id: str) -> AgentRun:
        """读取不依赖进程内存的运行快照。

        :param run_id: 运行标识。
        :returns: 已脱离会话的业务记录。
        """
        with Session(self.engine) as session:
            run=session.get(AgentRun,run_id)
            if run is None:
                raise DatasetError(404,"Agent 运行不存在")
            session.expunge(run)
            return run

    def event(self, run_id: str, node: str, event_type: str, payload: dict, *, level: str="info") -> None:
        """追加决策或固定工具结果，不记录凭据及模型内部思考。

        :param run_id: 业务运行。
        :param node: 图节点。
        :param event_type: 事件类型。
        :param payload: 已脱敏结构化信息。
        :param level: 事件等级。
        :returns: 无。
        """
        with Session(self.engine) as session, session.begin():
            session.add(AgentEvent(run_id=run_id,node=node,event_type=event_type,payload=payload,level=level))

    def patch(self, run_id: str, *, state: dict | None=None, **fields) -> AgentRun:
        """短事务更新进度，预算和输入不在可更新字段中。

        :param run_id: 业务运行。
        :param state: 合并到状态缓存中的片段。
        :param fields: 内部阶段、状态或已推导的数据关联。
        :returns: 最新快照。
        """
        allowed={"status","phase","dataset_version_id","dataset_export_id","class_schema_id","started_at","finished_at","error","retry_count","best_model_id","report_path","report_sha256","review_confirmation","pause_requested","stop_requested"}
        if set(fields)-allowed:
            raise ValueError("禁止修改 Agent 受保护字段")
        with Session(self.engine) as session,session.begin():
            run=session.get(AgentRun,run_id)
            if state is not None:
                run.state={**run.state,**state}
            for key,value in fields.items():
                setattr(run,key,value)
        return self.get(run_id)

    def control(self, run_id: str, action: str, confirmation: dict | None=None) -> AgentRun:
        """人工暂停、停止与恢复，不允许模型触发此管理入口。

        :param run_id: 当前运行。
        :param action: pause、stop 或 resume。
        :param confirmation: 人审或故障恢复时的明确操作者信息。
        :returns: 控制请求后的状态。
        """
        with Session(self.engine) as session,session.begin():
            run=session.get(AgentRun,run_id)
            if run is None:
                raise DatasetError(404,"Agent 运行不存在")
            if run.status in {"completed","stopped"}:
                raise DatasetError(409,"已封存运行不能再次执行，请创建新运行")
            if action=="pause":
                run.pause_requested=True
            elif action=="stop":
                run.stop_requested=True
                run.pause_requested=False
                run.status="canceling"
            elif action=="resume":
                if run.status not in {"paused","needs_review","failed"}:
                    raise DatasetError(409,"当前状态不允许恢复")
                if run.status in {"needs_review","failed"} and not confirmation:
                    raise DatasetError(422,"人工审核或故障恢复必须提供操作者与说明")
                if run.phase in {"cvat_review","annotation_validation"} and (not confirmation or not confirmation.get("missed_objects_checked")):
                    raise DatasetError(422,"需要明确确认 CVAT 审核及漏标检查")
                if run.phase in {"cvat_review","annotation_validation"}:
                    run.review_confirmation={**confirmation,"_revision":(run.review_confirmation or {}).get("_revision",0)+1}
                run.pause_requested=False
                run.status="queued"
                run.error=None
                run.retry_count=0
                if run.phase=="agent_review":
                    state=dict(run.state)
                    request=state.pop("review_request",None) or {}
                    state["human_acknowledgement"]={**confirmation,"call_id":request.get("call_id")}
                    run.state=state
            else:
                raise DatasetError(422,"未知控制操作")
            session.add(AgentEvent(run_id=run_id,node="operator",event_type=action,payload=confirmation or {}))
        return self.get(run_id)

    def wake(self, run_id: str, question: str = "", *, sleep_until: str | None = None) -> AgentRun:
        """原子登记本次休眠的唤醒请求，复用控制器轮询，不接触图检查点。

        先以条件更新取得数据库写锁，再追加事件，避免同时点击产生重复请求。
        问题保存在只追加的事件中，不会被控制器的进度缓存更新覆盖。
        """
        with Session(self.engine) as session, session.begin():
            run = session.scalars(update(AgentRun).where(AgentRun.id == run_id,
                AgentRun.status == "waiting", AgentRun.phase == "agent_sleep",
                AgentRun.stop_requested.is_(False), AgentRun.pause_requested.is_(False))
                .values(next_poll_at=utc_now()).returning(AgentRun)).first()
            if run is None:
                if session.get(AgentRun, run_id) is None:
                    raise DatasetError(404, "Agent 运行不存在")
                raise DatasetError(409, "仅可唤醒休眠中的 Agent；暂停或人工审核请使用相应恢复入口")
            waiting = run.state.get("waiting") or {}
            until = waiting.get("until")
            if waiting.get("kind") != "sleep" or not until or (sleep_until is not None and sleep_until != until):
                raise DatasetError(409, "Agent 的休眠状态已变化，请刷新后重新唤醒；问题尚未提交")
            policy = AgentPolicy.model_validate(run.policy)
            if run.state.get("model_calls", 0) >= policy.max_model_calls:
                raise DatasetError(409, "模型调用预算已耗尽，不能继续唤醒 Agent")
            existing = session.scalars(select(AgentEvent).where(AgentEvent.run_id == run_id,
                AgentEvent.event_type == "wake_requested", AgentEvent.payload["sleep_until"].as_string() == until)).first()
            if existing is not None:
                if existing.payload.get("question", "") != question:
                    raise DatasetError(409, "本次休眠已收到唤醒请求，请等待 Agent 处理后再提问")
            else:
                session.add(AgentEvent(run_id=run_id, node="operator", event_type="wake_requested",
                    payload={"question": question, "sleep_until": until}))
        return self.get(run_id)

    def wake_request(self, run_id: str, sleep_until: str) -> dict | None:
        """读取匹配本次休眠的请求，旧请求不能唤醒后续休眠。"""
        with Session(self.engine) as session:
            row = session.scalars(select(AgentEvent).where(AgentEvent.run_id == run_id,
                AgentEvent.event_type == "wake_requested", AgentEvent.payload["sleep_until"].as_string() == sleep_until)
                .order_by(AgentEvent.id.desc()).limit(1)).first()
            return {"request_id": row.id, "question": row.payload.get("question", "")} if row else None

    def operator_messages(self, run_id: str, *, pending_only: bool = False, limit: int = 10) -> list[dict]:
        """从持久事件读取问题和回答；未答问题持续进入每轮档案，直至明确回答。"""
        with Session(self.engine) as session:
            query = select(AgentEvent).where(AgentEvent.run_id == run_id, AgentEvent.event_type == "wake_requested")
            if pending_only:
                answered = select(AgentEvent.payload["request_id"].as_integer()).where(AgentEvent.run_id == run_id,
                    AgentEvent.event_type == "operator_answer").correlate(None)
                query = query.where(AgentEvent.payload["question"].as_string() != "", AgentEvent.id.not_in(answered))
            rows = list(session.scalars(query.order_by(AgentEvent.id if pending_only else AgentEvent.id.desc()).limit(limit)))
            answers = list(session.scalars(select(AgentEvent).where(AgentEvent.run_id == run_id,
                AgentEvent.event_type == "operator_answer", AgentEvent.payload["request_id"].as_integer().in_([row.id for row in rows])))) if rows else []
            by_request = {row.payload["request_id"]: row for row in answers}
            instructions = self._instruction_ledger(session, run_id)
            result = [{"request_id": row.id, "sleep_until": row.payload["sleep_until"],
                "question": row.payload.get("question", ""), "created_at": row.created_at,
                "answer": by_request[row.id].payload["answer"] if row.id in by_request else None,
                "answered_at": by_request[row.id].created_at if row.id in by_request else None,
                "instruction_key": by_request[row.id].payload.get("instruction_key") if row.id in by_request else None,
                "instruction_active": any(item["request_id"] == row.id for item in instructions.values())} for row in rows]
            return result if pending_only else list(reversed(result))

    @staticmethod
    def _instruction_ledger(session, run_id):
        events = session.scalars(select(AgentEvent).where(AgentEvent.run_id == run_id,
            AgentEvent.event_type.in_(["operator_instruction_set", "operator_instruction_revoked"])).order_by(AgentEvent.id))
        active = {}
        for event in events:
            data = event.payload
            key = data["key"]
            if event.event_type == "operator_instruction_set":
                active[key] = {"instruction_id": event.id, "key": key, "request_id": data["request_id"],
                    "text": data["text"], "sleep_minutes": data.get("sleep_minutes"), "created_at": event.created_at}
            elif active.get(key, {}).get("instruction_id") == data["instruction_id"]:
                active.pop(key)
        return active

    def operator_instructions(self, run_id: str) -> list[dict]:
        self.get(run_id)
        with Session(self.engine) as session:
            return list(self._instruction_ledger(session, run_id).values())

    def revoke_operator_instruction(self, run_id: str, key: str, instruction_id: int) -> dict:
        with Session(self.engine) as session, session.begin():
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            run = session.get(AgentRun, run_id)
            if run is None:
                raise DatasetError(404, "Agent 运行不存在")
            if run.status in {"completed", "stopped"} or run.stop_requested:
                raise DatasetError(409, "流程已结束或正在停止，不能修改持续指令")
            source = session.get(AgentEvent, instruction_id)
            if source is None or source.run_id != run_id or source.event_type != "operator_instruction_set" or source.payload.get("key") != key:
                raise DatasetError(404, "这条持续指令不属于当前流程")
            active = self._instruction_ledger(session, run_id).get(key)
            if active and active["instruction_id"] != instruction_id:
                raise DatasetError(409, "指令已被后续要求替换，请刷新后确认要撤销的版本")
            if active:
                session.add(AgentEvent(run_id=run_id, node="operator", event_type="operator_instruction_revoked",
                    payload={"key": key, "instruction_id": instruction_id, "source": "operator"}))
        return {"status": "revoked", "key": key, "instruction_id": instruction_id}

    def answer_operator(self, run_id: str, request_id: int, answer: str, *, instruction_action="none", instruction_key=None, sleep_minutes=None) -> dict:
        """记录 Agent 对本运行问题的可见回答，检查点重放不会重复写入。"""
        change = OperatorInstructionChange(instruction_action=instruction_action, instruction_key=instruction_key, sleep_minutes=sleep_minutes)
        with Session(self.engine) as session, session.begin():
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            request = session.get(AgentEvent, request_id)
            if request is None or request.run_id != run_id or request.event_type != "wake_requested" or not request.payload.get("question"):
                raise DatasetError(422, "只能回答本运行中真实存在的操作者问题")
            existing = session.scalars(select(AgentEvent).where(AgentEvent.run_id == run_id,
                AgentEvent.event_type == "operator_answer", AgentEvent.payload["request_id"].as_integer() == request_id)).first()
            if existing is None:
                active = self._instruction_ledger(session, run_id)
                if change.instruction_action == "remember":
                    if change.instruction_key not in active and len(active) >= 32:
                        raise DatasetError(409, "持续指令已达32条，请先撤销不再需要的指令；问题原文仍保留")
                    session.add(AgentEvent(run_id=run_id, node="agent", event_type="operator_instruction_set", payload={"key": change.instruction_key,
                        "request_id": request_id, "text": request.payload["question"], "sleep_minutes": change.sleep_minutes}))
                elif change.instruction_action == "forget" and change.instruction_key in active:
                    session.add(AgentEvent(run_id=run_id, node="agent", event_type="operator_instruction_revoked", payload={"key": change.instruction_key,
                        "instruction_id": active[change.instruction_key]["instruction_id"], "request_id": request_id, "source": "operator_message"}))
                session.add(AgentEvent(run_id=run_id, node="agent", event_type="operator_answer",
                    payload={"request_id": request_id, "answer": answer, **change.model_dump()}))
            else:
                answer = existing.payload["answer"]
                change = OperatorInstructionChange.model_validate({key: existing.payload.get(key) for key in ("instruction_action", "instruction_key", "sleep_minutes") if key in existing.payload})
        return {"request_id": request_id, "answer": answer, "status": "answered", **change.model_dump()}

    def reserve_model_call(self, run_id: str, *, kind: str, reserve_decision: bool = False) -> int:
        """摘要与决策共用冻结调用预算；请求失败也算一次尝试。"""
        with Session(self.engine) as session, session.begin():
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            run = session.get(AgentRun, run_id)
            policy = AgentPolicy.model_validate(run.policy)
            used = int(run.state.get("model_calls", 0))
            if run.stop_requested or run.pause_requested:
                raise DatasetError(409, "操作者已暂停或停止，不能发起新的模型请求")
            if used + 1 + int(reserve_decision) > policy.max_model_calls:
                raise DatasetError(409, "模型调用预算不足，不能额外发起摘要或决策请求")
            run.state = {**run.state, "model_calls": used + 1}
            session.add(AgentEvent(run_id=run_id, node="agent", event_type="model_call_reserved", payload={"turn": used, "kind": kind}))
        return used

    def setup_training(self, run_id: str, data: TrainingSetup) -> AgentRun:
        """人审完成后开始训练：首轮参数可选，预算只能收紧，不能超过创建时的上限。

        :param run_id: 停在开始训练步骤的运行。
        :param data: 操作者填写的首轮参数片段与预算片段。
        :returns: 已排队继续的运行。
        """
        with Session(self.engine) as session,session.begin():
            run=session.get(AgentRun,run_id)
            if run is None:
                raise DatasetError(404,"Agent 运行不存在")
            if run.phase!="training_setup" or run.status!="needs_review":
                raise DatasetError(409,"运行当前不在开始训练步骤")
            if data.dataset_export_id != run.dataset_export_id:
                raise DatasetError(422,"请明确选择本流程已审核的训练数据；选择其他数据集请创建新的训练流程")
            old=AgentPolicy.model_validate(run.policy)
            try:
                policy=AgentPolicy.model_validate({**run.policy,**data.policy})
            except ValueError as error:
                raise DatasetError(422,f"预算设置不合法：{error}") from error
            looser=[key for key in ("max_experiments","max_training_hours","max_retries","max_gpu_memory_ratio","stagnation_rounds","max_epochs_per_experiment","max_imgsz","max_model_calls") if getattr(policy,key)>getattr(old,key)]
            if policy.minimum_improvement<old.minimum_improvement:
                looser.append("minimum_improvement")
            if not set(policy.allowed_models)<=set(old.allowed_models) or not set(policy.allowed_operations)<=set(old.allowed_operations):
                looser.append("allowed_models/allowed_operations")
            if looser:
                raise DatasetError(422,f"开始训练时预算只能收紧：{', '.join(looser)} 超过创建时的上限")
            initial=fit_initial(self.settings,{**run.initial_parameters,**data.initial_parameters},set(data.initial_parameters),policy)
            run.policy,run.initial_parameters=policy.model_dump(),initial
            run.state={**run.state,"operator_fields":sorted(data.initial_parameters),"training_setup":{"at":utc_now(),"source":"operator","dataset_export_id":run.dataset_export_id,"initial_parameters":data.initial_parameters,"policy":data.policy}}
            run.status,run.error="queued",None
            session.add(AgentEvent(run_id=run_id,node="operator",event_type="training_setup",payload=data.model_dump()))
        return self.get(run_id)

    def budget(self, run_id: str) -> dict:
        """统计本运行全部训练尝试，包括失败、恢复及当前耗时。

        :param run_id: 业务运行。
        :returns: 已创建实验数、真实训练用时和剩余预算。
        """
        run=self.get(run_id)
        with Session(self.engine) as session:
            iterations=list(session.scalars(select(AgentIteration).where(AgentIteration.run_id==run_id).order_by(AgentIteration.number)))
            tasks=list(session.scalars(select(TrainingTask).join(AgentIteration,AgentIteration.experiment_id==TrainingTask.experiment_id).where(AgentIteration.run_id==run_id)))
        seconds=0.0
        for task in tasks:
            elapsed=task.elapsed_seconds
            if task.status in {"running","canceling"} and task.started_at:
                elapsed=max(elapsed,(datetime.now(timezone.utc)-datetime.fromisoformat(task.started_at)).total_seconds())
            seconds+=max(0,elapsed)
        limit=run.policy["max_training_hours"]*3600
        return {"experiments":len(iterations),"training_seconds":seconds,"remaining_seconds":max(0,limit-seconds),"max_experiments":run.policy["max_experiments"],"max_training_seconds":limit}
