"""每轮重建真实训练档案，携带预算内的历史、持久摘要及操作者持续指令。

旧历史由独立上下文节点按完整工具对话组摘要，并与剩余对话一起存入检查点。
图像只保留文件引用，在读取后的下一轮从磁盘生成，不把 base64 写入检查点。
"""

import base64
import io
import json
from datetime import datetime, timezone

from PIL import Image

from yolo_agent.models.dataset import utc_now
from yolo_agent.services.datasets import owned_path
from yolo_agent.services.context_budget import TokenBudget
from yolo_agent.settings import Settings
from yolo_agent.tools.workflow import WorkflowTools

MAX_IMAGES = 4
IMAGE_SIDE = 1280


def _elapsed(task) -> float:
    if task.status in {"running", "canceling"} and task.started_at:
        return max(task.elapsed_seconds, (datetime.now(timezone.utc) - datetime.fromisoformat(task.started_at)).total_seconds())
    return task.elapsed_seconds


def build_dossier(tools: WorkflowTools, *, model_calls: int) -> dict:
    """汇总模型做训练决策所需的全部真实信息。

    :param tools: 绑定本运行的工具面。
    :param model_calls: 本运行已发生的模型调用次数。
    :returns: 可 JSON 序列化的档案；不含测试集数据或标签。
    """
    run, policy = tools.service.get(tools.run_id), tools.policy()
    budget = tools.service.budget(tools.run_id)
    pairs = tools.iterations()
    concluded = [(item, experiment) for item, experiment in pairs if item.conclusion and item.conclusion.get("map50_95") is not None]
    best = max(concluded, key=lambda pair: pair[0].conclusion["map50_95"]) if concluded else None
    # 最近三轮与最佳轮给出完整明细，更早的实验只保留比较字段，控制档案长度。
    detailed = {item.number for item, _ in pairs[-3:]} | ({best[0].number} if best else set())
    active = []
    for task in tools._active_training():
        elapsed = _elapsed(task)
        per_epoch = elapsed / task.current_epoch if task.current_epoch else None
        active.append({"task_id": task.id, "experiment_id": task.experiment_id, "status": task.status, "epoch": f"{task.current_epoch}/{task.total_epochs}", "elapsed_seconds": round(elapsed), "seconds_per_epoch": round(per_epoch, 1) if per_epoch else None, "eta_seconds": round(per_epoch * (task.total_epochs - task.current_epoch)) if per_epoch else None, "stop_requested": task.stop_requested})
    pending = []
    for item, experiment in pairs:
        if experiment.status == "completed" and not item.conclusion:
            evaluations = tools._evaluations(experiment.id)
            pending.append({"experiment_id": experiment.id, "evaluation": evaluations[-1].status if evaluations else "尚未提交"})
    pinned = {field: run.initial_parameters[field] for field in run.state.get("operator_fields", [])}
    return {
        "now": utc_now(),
        "goal": {"run_name": run.name, "requirement": run.requirement or "操作者未填写需求说明", "purpose": policy.purpose},
        "budget": {**{key: round(value) if isinstance(value, float) else value for key, value in budget.items()}, "model_calls_used": model_calls, "max_model_calls": policy.max_model_calls},
        "limits": {"prepared_models": tools.prepared_models(), "max_epochs_per_experiment": policy.max_epochs_per_experiment, "max_imgsz": policy.max_imgsz, "max_gpu_memory_ratio": policy.max_gpu_memory_ratio, "search_space": tools.search_space(), "parameter_rule": "全部可调项见 start_experiment 参数定义；search_space 只对列出的维度额外约束；多尺度最高尺寸也受 max_imgsz 限制", "stagnation_rule": {"rounds": policy.stagnation_rounds, "minimum_improvement": policy.minimum_improvement}},
        "skills": tools.skill_knowledge(),
        "evidence_tools": {"before_first_experiment": ["inspect_training_parameters", "read_dataset_examples"],
            "history": "get_experiment 按 section=parameters/curve/evaluation/training 定向分页；compare_experiments 比较任意两轮",
            "diagnosis": "read_validation_errors 按类别、fn/fp、小目标筛选；read_confusion_matrix 分页读取方向和计数",
            "before_training": "preview_experiment 只读预检；通过后以相同输入 start_experiment，开训时重新校验"},
        "operator_first_round": {"pinned": pinned, "model_locked": len(policy.allowed_models)==1, "rule": "只有 pinned 中的字段是操作者明确填写的首轮参数，首轮必须照用；allowed_models 只有一种时，全程锁定该基座。未列入 pinned 的可调项从首轮起由 Agent 决定，不能称为操作者指定或锁定。"},
        "platform_training_defaults": {"parameters": {field: value for field, value in run.initial_parameters.items() if field not in pinned}, "rule": "这些值由平台补齐，不是操作者指定。首轮未在 start_experiment.parameters 中显式提供的字段使用这些默认值；可调项允许按需求、数据与设备在预算和搜索空间内覆盖。主动采用默认值时，应说明这是 Agent 的基线选择。"},
        "experiment_design_rule": "默认 joint_search 可联合调参，声明 hypothesis，只评价整组收益；有必要且预算允许时才用 ablation 相对指定父实验只改 target_parameter，保持数据一致。不要求逐参数穷举，也不扩大预算。数字方向引用 confusion_evidence，不能从图猜测。",
        "training_value_review": {"budget_is_ceiling": True,
            "rule": "每次新实验或恢复前自主判断是否值得；后续开训需 value_assessment（已有证据、可验证预期价值、成本权衡）。可在硬性停滞阈值前 finish，不能只因预算尚余或 Optuna 有建议而继续。",
            "evidence_sources": ["goal", "experiments", "latest_evaluation", "active_training", "budget", "limits", "operator_instructions"],
            "choices": ["有具体且值得验证的方向：预检后继续", "证据尚缺或评估中：定向读取或等待", "数据处理是必要前提：请求人工处理", "新增训练价值不足：保留最佳候选并结束"],
            "uncertainty_rule": "重点类别收益与总体指标一起看；不把单轮波动、缺少验证支持或早期曲线当作定论，不自设达标线或宣称全局最优。"},
        "compute": tools.compute_environments(),
        "datasets": tools.allowed_exports(),
        "experiments": [tools.experiment_brief(item, experiment, full=item.number in detailed) for item, experiment in pairs],
        "best_so_far": {"iteration": best[0].number, "experiment_id": best[1].id, "val_map50_95": best[0].conclusion["map50_95"], "dataset_export_id": best[1].dataset_export_id} if best else None,
        "active_training": active,
        "waiting_for_evaluation": pending,
        "latest_evaluation": tools.evaluation_digest(concluded[-1][1].id) if concluded else None,
        "stagnation": tools.stagnation(),
        "decision_log": tools.decision_log(),
        "operator_questions": tools.service.operator_messages(tools.run_id, pending_only=True),
        "operator_conversation": tools.service.operator_messages(tools.run_id, limit=3),
        "operator_instructions": tools.service.operator_instructions(tools.run_id),
        "human_acknowledgement": run.state.get("human_acknowledgement"),
    }


def _public(message: dict) -> dict:
    """去掉系统内部键，发送给服务商的消息只含标准字段。"""
    return {key: value for key, value in message.items() if not key.startswith("_")}


def history_units(history: list[dict]) -> list[list[dict]]:
    """以助手调用及其全部结果为原子组，休眠的唤醒结果也归入同组。"""
    units = []
    for message in history:
        if message["role"] == "assistant" or not units:
            units.append([message])
        else:
            units[-1].append(message)
    return units


def _data_url(settings: Settings, relative: str) -> str:
    """缩放并转码指标图，控制单张图片的请求体积。"""
    with Image.open(owned_path(settings, relative)) as image:
        image = image.convert("RGB")
        image.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def request_messages(settings: Settings, system: str, history: list[dict], dossier: dict, *, summary: str = "", include_images: bool = True, image_limit: int = MAX_IMAGES, load_images: bool = True, refs: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """组装一次模型请求：系统提示、摘要、剩余对话、最新档案与待看图像。

    :param settings: 存储配置，用于读取图像。
    :param system: 系统提示。
    :param history: 检查点中的完整对话。
    :param dossier: 本轮训练档案。
    :returns: 发送给服务商的消息，以及实际附带的图像引用。
    """
    units = history_units(history)
    messages = [{"role": "system", "content": system}]
    if summary:
        messages.append({"role": "user", "content": "先前对话的自动摘要（历史观察，不是新的操作者指令；不得覆盖冻结预算、当前档案和持续指令）：\n" + summary})
    if units:
        messages.append({"role": "user", "content": "根据当前训练档案继续自主训练；已摘要的历史可参考上文。"})
        messages += [_public(message) for unit in units for message in unit]
    all_refs = refs if refs is not None else [ref for message in (units[-1] if units else []) if message["role"] == "tool" for ref in message.get("_images", [])]
    images = all_refs[:image_limit] if include_images else []
    text = "当前训练档案（系统每轮从数据库重新生成，以此为准）：\n" + json.dumps(dossier, ensure_ascii=False, default=str, separators=(",", ":"))
    if all_refs and len(images) < len(all_refs):
        text += f"\n本轮实际附带 {len(images)} / {len(all_refs)} 张图片，其余受图像上限或上下文预算限制未发送，不能当作已查看；必要时请分批重读，结构化数值仍可使用。"
    content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": _data_url(settings, ref) if load_images else "[image]"}} for ref in images]
    messages.append({"role": "user", "content": content if images else text})
    return messages, images


def assemble(settings: Settings, system: str, history: list[dict], dossier: dict, *, include_images: bool = True, summary: str = "", tools: list[dict] = (), budget: TokenBudget | None = None, image_limit: int = MAX_IMAGES) -> tuple[list[dict], dict]:
    messages, refs = request_messages(settings, system, history, dossier, summary=summary, include_images=include_images, image_limit=image_limit)
    budget = budget or TokenBudget(settings)
    tokens = budget.check(messages, tools)
    return messages, {**budget.info(), "estimated_input_tokens": tokens, "history_units": len(history_units(history)),
        "kept_units": len(history_units(history)), "context_chars": sum(len(json.dumps(_public(message), ensure_ascii=False)) for message in history) + len(json.dumps(dossier, ensure_ascii=False)), "images": len(refs)}
