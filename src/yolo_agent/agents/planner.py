"""通过兼容 OpenAI 的成熟 SDK 调用模型：训练段的工具循环与类别草案。"""

import json
from typing import Callable

from openai import OpenAI

from yolo_agent.schemas.annotation import COCO_CLASSES, DetectionClass
from yolo_agent.services.datasets import DatasetError
from yolo_agent.settings import Settings

SYSTEM_PROMPT = """你是 YOLO26 目标检测训练平台的自主训练 Agent，通过工具循环完成"训练 → 评估 → 分析 → 调整 → 再训练 → 导出"。每次回复必须调用至少一个工具；理由用中文，引用档案中的真实数值，不要输出内部思考过程。

每轮最后一条用户消息是系统从数据库重新生成的训练档案，以它为准：其中有操作者需求、预算、数据集画像、设备、全部实验的参数差异与结果、当前训练的逐 epoch 曲线、最新 val 评估、停滞进度和你最近的决策记录。较早的对话会被裁剪，需要长期记住的观察请写进工具的 reason。

工作方法：
- 默认使用 YOLO26n，YOLO26s 为备选；选择更大模型需说明依据并遵守操作者模型白名单与资源预算。
- 首轮用 start_experiment 开始训练。operator_first_round 中操作者填写的参数首轮必须照用，其余参数由你按需求、数据规模和设备决定；之后各轮都可以调整。
- 训练很慢。启动后用 sleep 休眠，时长参考 seconds_per_epoch × 剩余 epoch；训练、评估或导出结束时系统会提前唤醒你。不要频繁短睡，模型调用次数也计入预算。
- 训练完成后系统自动提交固定 val 评估，评估完成后档案会出现结论。用 read_chart 看 results、PR 曲线和 val 场景样例；混淆矩阵的数字必须引用 confusion_evidence 或 read_confusion_matrix 的结构化事实。每条方向为 true_class→predicted_class，比例分母为 true_count；漏检来自 predicted=background 的行，true=background 的列为无真值匹配的预测。禁止根据图片自行读数、交换方向或用训练图代替独立验证数据。图像只在读取后的下一轮可见。
- 训练中途如果曲线已明显过拟合（val loss 持续上升而 train loss 下降）、长期不再提升或出现异常，可以 stop_training 节省预算，再 resume_training 或换参数开新实验。
- 下一轮按优先级排查：数据质量与标注、类别不均衡与困难样本、训练参数、模型规格。默认 comparison_mode=joint_search，允许结合 Optuna 建议联合调参，填写基于真实证据的 hypothesis，只能说整组配置更好/更差，不拆分单项贡献。确实需要验证某个高价值假设且预算允许时才做 comparison_mode=ablation：相对指定候选父实验只改 target_parameter 一项，其余字段继承、数据一致。可把候选配置某一项回退做定向消融；不要求每个参数都验证，不为归因耗尽或扩大预算。数据问题无法靠训练解决时用 request_human_review。
- 结束前用 export_model 导出最佳实验的权重（按需求中的部署目标选择 ONNX 或 TorchScript，没有说明时导出 ONNX），然后调用 finish 指定最佳实验并总结。

硬约束（工具会拒绝越界请求）：
- 测试集不可见，也不得要求查看或重新划分；只依据固定 val 协议的 mAP50-95 比较实验。换用其它数据集后验证集不同，指标不能直接比较。
- 父实验必须已有有效固定 val 结论。用 comparison、delta_parent 比较指定父实验，delta_previous 仅表示时间相邻的轮次变化。一个请求变量也可能联动多个实际因素，核对 effective_changes 与 coupled_effects；一次训练的差异只算观察结果，不能声称证明因果或统计显著。矩阵、固定阈值错误清单及 AP/P/R 协议不同，不能混算。
- start_experiment 的参数定义列出了完整可调项；档案 limits 中的模型和预算必须遵守，search_space 对列出的维度额外收紧，其余参数按工具定义范围选择。multi_scale 实际最高尺寸也不得超过 max_imgsz。auto 优化器会覆盖 lr0 和 warmup_bias_lr，研究学习率要指定明确的优化器。rect 模式可能关闭混合增强，理由应区分请求参数与训练库实际采用值。
- 类别变更、删除数据、生产发布、自定义模型架构都必须由人完成；你不能批准标注，也没有 Shell、下载或任意文件访问能力。
- 不编造指标，不预设下一轮一定更好；样本很少时要说明结论的局限。
- 预算耗尽、收益停滞或平台故障时停止，不要求扩大预算。purpose=development_validation 表示本机小规模软件闭环验收，不声称业务模型合格。"""

DRAFT_PROMPT = """你帮助操作者把目标检测需求整理成类别草案，草案会交给操作者逐项确认后才会创建。
只调用 submit_class_draft。name 用操作者的语言写简短类别名；prompt 写给 Grounding DINO 的英文名词短语（只含字母、数字、空格、逗号、连字符或撇号，以字母开头，不超过 80 字符），要具体、可见、互不重叠。
只列需求里明确要检测的目标，不要添加背景或推测的类别；需求含糊时在 notes 里说明需要操作者确认的地方。可准确对应 COCO 类别时提供 coco_class 建议，专业类别省略；此建议需操作者确认，不改变类别定义。"""


def _client(settings: Settings) -> OpenAI:
    if not settings.llm_model or not settings.llm_api_key.get_secret_value():
        raise DatasetError(503, "模型服务尚未配置")
    return OpenAI(api_key=settings.llm_api_key.get_secret_value(), base_url=settings.llm_base_url, timeout=settings.llm_timeout_seconds, max_retries=0)


def call_model(settings: Settings, messages: list[dict], tools: list[dict], *, on_response: Callable[[dict], None] | None = None) -> tuple[dict, dict]:
    """请求一轮工具调用决策，SDK 不重试，失败交给图节点的有限重试。

    :param settings: 操作者配置的地址、模型、SecretStr 密钥和超时。
    :param messages: 已裁剪并附加训练档案与图像的对话。
    :param tools: 与执行端校验同源的工具定义。
    :param on_response: 解析前保存用量的审计回调，不合规的响应也要计入用量。
    :returns: 可写入检查点的助手消息，以及用量、模型和请求标识。
    """
    with _client(settings) as client:
        response = client.chat.completions.create(model=settings.llm_model, messages=messages, tools=tools, tool_choice=settings.llm_tool_choice, max_tokens=4000, extra_body=settings.llm_extra_body or None)
    message = response.choices[0].message
    metadata = {"provider_request_id": response.id, "model": response.model, "usage": response.usage.model_dump() if response.usage else None, "finish_reason": response.choices[0].finish_reason}
    if on_response:
        on_response(metadata)
    if not message.tool_calls:
        raise ValueError("模型没有调用任何工具")
    names = {tool["function"]["name"] for tool in tools}
    calls = []
    for call in message.tool_calls:
        if call.function.name not in names:
            raise ValueError(f"模型调用了白名单外的工具：{call.function.name}")
        calls.append({"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"}})
    # 只保存可见回复和工具调用，不保存服务商返回的推理内容。
    return {"role": "assistant", "content": message.content or None, "tool_calls": calls}, metadata


def draft_classes(settings: Settings, requirement: str) -> dict:
    """把需求说明整理成待人工确认的类别草案，不创建任何类别。

    :param settings: 模型服务配置。
    :param requirement: 操作者填写的检测需求。
    :returns: 带连续 class_id 的类别列表与补充说明。
    """
    schema = {"type": "object", "properties": {"classes": {"type": "array", "minItems": 1, "maxItems": 50, "items": {"type": "object", "properties": {"name": {"type": "string", "maxLength": 100}, "prompt": {"type": "string", "pattern": DetectionClass.model_json_schema()["properties"]["prompt"]["pattern"]}}, "required": ["name", "prompt"]}}, "notes": {"type": "string", "maxLength": 1000}}, "required": ["classes"]}
    schema["properties"]["classes"]["items"]["properties"]["coco_class"] = {"type": "string", "enum": list(COCO_CLASSES), "description": "可选的 COCO 对应类；没有准确对应时省略"}
    tool = {"type": "function", "function": {"name": "submit_class_draft", "description": "提交待操作者确认的检测类别草案。", "parameters": schema}}
    with _client(settings) as client:
        response = client.chat.completions.create(model=settings.llm_model, messages=[{"role": "system", "content": DRAFT_PROMPT}, {"role": "user", "content": requirement}], tools=[tool], tool_choice={"type": "function", "function": {"name": "submit_class_draft"}}, max_tokens=1500, extra_body=settings.llm_extra_body or None)
    calls = response.choices[0].message.tool_calls
    if not calls or calls[0].function.name != "submit_class_draft":
        raise DatasetError(502, "模型没有返回类别草案")
    try:
        payload = json.loads(calls[0].function.arguments)
        classes = []
        for index, item in enumerate(payload["classes"]):
            category = DetectionClass(class_id=index, name=item["name"], prompt=item["prompt"]).model_dump()
            if item.get("coco_class") is not None:
                if item["coco_class"] not in COCO_CLASSES:
                    raise ValueError("类别草案返回未知 COCO 类名")
                category["coco_class"] = item["coco_class"]
            classes.append(category)
    except (ValueError, KeyError, TypeError) as error:
        raise DatasetError(502, f"模型返回的类别草案不合法：{error}") from error
    return {"classes": classes, "notes": str(payload.get("notes", ""))[:1000]}
