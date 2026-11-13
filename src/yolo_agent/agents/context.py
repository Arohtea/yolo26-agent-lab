"""训练段上下文工程：每轮从数据库重建训练档案，裁剪对话，并按需附加指标图。

长时间训练会让对话无限增长，而且休眠、重启后模型的记忆只剩检查点里的消息。
因此事实统一来自数据库生成的档案（每轮最新、完整），对话只保留最近若干轮
工具往来用于连贯推理；被裁掉的早期决策由档案中的决策记录兜底。图像体积大，
不写入检查点，只在读取后的下一轮从磁盘生成。
"""

import base64
import io
import json
from datetime import datetime, timezone

from PIL import Image

from yolo_agent.models.dataset import utc_now
from yolo_agent.services.datasets import owned_path
from yolo_agent.settings import Settings
from yolo_agent.tools.workflow import WorkflowTools

# ponytail: 按字符数估算上下文，约 3 字符 1 token；换用 tokenizer 计数前不需要更精确。
HISTORY_CHARS = 60_000
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
        "limits": {"prepared_models": tools.prepared_models(), "max_epochs_per_experiment": policy.max_epochs_per_experiment, "max_imgsz": policy.max_imgsz, "max_gpu_memory_ratio": policy.max_gpu_memory_ratio, "search_space": tools.study().search_space(), "parameter_rule": "全部可调项见 start_experiment 参数定义；search_space 只对列出的维度额外约束；多尺度最高尺寸也受 max_imgsz 限制", "stagnation_rule": {"rounds": policy.stagnation_rounds, "minimum_improvement": policy.minimum_improvement}},
        "operator_first_round": {"pinned": pinned, "starting_parameters": run.initial_parameters, "rule": "pinned 中的值首轮必须照用，之后可以调整；其余参数由你决定"},
        "experiment_design_rule": "默认 joint_search 可联合调参，声明 hypothesis，只评价整组收益；有必要且预算允许时才用 ablation 相对指定父实验只改 target_parameter，保持数据一致。不要求逐参数穷举，也不扩大预算。数字方向引用 confusion_evidence，不能从图猜测。",
        "compute": tools.compute_environments(),
        "datasets": tools.allowed_exports(),
        "experiments": [tools.experiment_brief(item, experiment, full=item.number in detailed) for item, experiment in pairs],
        "best_so_far": {"iteration": best[0].number, "experiment_id": best[1].id, "val_map50_95": best[0].conclusion["map50_95"], "dataset_export_id": best[1].dataset_export_id} if best else None,
        "active_training": active,
        "waiting_for_evaluation": pending,
        "latest_evaluation": tools.evaluation_digest(concluded[-1][1].id) if concluded else None,
        "stagnation": tools.stagnation(),
        "decision_log": tools.decision_log(),
        "human_acknowledgement": run.state.get("human_acknowledgement"),
    }


def _public(message: dict) -> dict:
    """去掉系统内部键，发送给服务商的消息只含标准字段。"""
    return {key: value for key, value in message.items() if not key.startswith("_")}


def _data_url(settings: Settings, relative: str) -> str:
    """缩放并转码指标图，控制单张图片的请求体积。"""
    with Image.open(owned_path(settings, relative)) as image:
        image = image.convert("RGB")
        image.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def assemble(settings: Settings, system: str, history: list[dict], dossier: dict) -> tuple[list[dict], dict]:
    """组装一次模型请求：系统提示、裁剪后的对话、最新档案与待看图像。

    :param settings: 存储配置，用于读取图像。
    :param system: 系统提示。
    :param history: 检查点中的完整对话。
    :param dossier: 本轮训练档案。
    :returns: 发送给服务商的消息，以及裁剪统计。
    """
    units: list[list[dict]] = []
    for message in history:
        if message["role"] == "assistant" or not units:
            units.append([message])
        else:
            units[-1].append(message)
    # 按"助手调用 + 工具结果"整组保留，避免出现没有调用方的工具结果。
    kept, size = [], 0
    for unit in reversed(units):
        length = sum(len(json.dumps(_public(message), ensure_ascii=False)) for message in unit)
        if kept and size + length > HISTORY_CHARS:
            break
        kept.insert(0, unit)
        size += length
    dropped = len(units) - len(kept)
    messages = [{"role": "system", "content": system}]
    if kept:
        messages.append({"role": "user", "content": "开始自主训练。" + (f"较早的 {dropped} 轮对话已移出上下文，可参考档案中的决策记录。" if dropped else "")})
        messages += [_public(message) for unit in kept for message in unit]
    images = [ref for message in (kept[-1] if kept else []) if message["role"] == "tool" for ref in message.get("_images", [])][:MAX_IMAGES]
    text = "当前训练档案（系统每轮从数据库重新生成，以此为准）：\n" + json.dumps(dossier, ensure_ascii=False, default=str, separators=(",", ":"))
    content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": _data_url(settings, ref)}} for ref in images]
    messages.append({"role": "user", "content": content if images else text})
    return messages, {"history_units": len(units), "kept_units": len(kept), "context_chars": size + len(text), "images": len(images)}
