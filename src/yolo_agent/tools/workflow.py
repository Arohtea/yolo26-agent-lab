"""把既有成熟服务组合为有归属校验、预算与审计的固定工具。

Agent 只能通过本模块接触业务：前半段工具由固定图节点调用，训练段工具以
OpenAI 兼容的 function calling 暴露给模型。这里没有 Shell、任意路径、下载、
删除、审批或发布能力；所有参数都经 Pydantic 校验后再落到既有服务。
"""

import hashlib
import json
import math
from copy import deepcopy
from typing import Literal
from uuid import UUID, uuid5

from pydantic import Field, create_model
from sqlalchemy import select
from sqlalchemy.orm import Session

from yolo_agent.agents.optimization import OptimizationStudy, describe_search_space, search_distributions, stagnation_progress
from yolo_agent.agents.evidence import confusion_evidence
from yolo_agent.agents.experiment_design import comparison_design, validate_experiment_design
from yolo_agent.models.agent import AgentEvent, AgentIteration
from yolo_agent.models.annotation import AnnotationRun, DatasetExport
from yolo_agent.models.dataset import Dataset, DatasetVersion
from yolo_agent.models.evaluation import EvaluationInvalidation, EvaluationTask
from yolo_agent.models.experiment import Experiment, ExperimentMetric, ModelArtifact, ModelExport, TrainingTask
from yolo_agent.schemas.agent import AgentPolicy, OperatorInstructionChange
from yolo_agent.schemas.agent_tools import CompareExperimentsArgs, ConfusionReadArgs, DatasetExamplesArgs, ExperimentInspectArgs, ParameterInfoArgs, ValidationErrorsArgs
from yolo_agent.schemas.annotation import AutoAnnotateCreate, CvatReviewConfirm
from yolo_agent.schemas.dataset import AnalysisCreate, InputModel, VersionCreate
from yolo_agent.schemas.evaluation import EvaluationCreate
from yolo_agent.schemas.experiment import ExecutionLimits, ExperimentCreate, ModelExportCreate, TrainingParameters
from yolo_agent.services.agents import AgentService
from yolo_agent.services.agent_skills import SKILL_TOOLS, training_skills
from yolo_agent.services.annotations import AnnotationService
from yolo_agent.services.datasets import DatasetError, DatasetService, owned_path
from yolo_agent.services.evaluation import EvaluationService
from yolo_agent.services.exports import ExportService
from yolo_agent.services.training import TrainingService, validate_export
from yolo_agent.services.chart_artifacts import CHARTS
from yolo_agent.workers.queue import RUNTIME_PREFIX, redis_connection
from yolo_agent.tools.training_inspection import TrainingInspection

ACTIVE = {"queued", "running", "canceling"}
# 图名固定映射到训练库产物；训练目录只含 train/val，评估目录只含 val，测试集图像无从读取。
# 归一化面积小于 (32/640)^2 视为小目标，与评估报告的像素口径不同，只用于数据画像。
SMALL_AREA = (32 / 640) ** 2
_EXPORT_STATS: dict[tuple[str, str], dict] = {}


class WorkflowPaused(RuntimeError):
    """操作者暂停只阻止后续编排，不把已完成计算改写成失败。"""


class WorkflowStopped(RuntimeError):
    """操作者停止或预算耗尽，必须先收敛自己提交的计算。"""


class ChartArgs(InputModel):
    """只能按固定图名读取本运行实验的产物，不能给出路径。"""

    experiment_id: str
    source: Literal["training", "evaluation"]
    charts: list[Literal[tuple(CHARTS)]] = Field(min_length=1, max_length=4)


class ExperimentArgs(InputModel):
    """引用本运行创建的实验。"""

    experiment_id: str


class LogArgs(InputModel):
    """日志只返回末尾有限行。"""

    experiment_id: str
    lines: int = Field(default=80, ge=10, le=300)


class TrainingValueAssessment(InputModel):
    """由 Agent 判断训练价值，不用固定分数代替证据与成本权衡。"""

    evidence: str = Field(min_length=10, max_length=1200, description="引用已有实验、曲线或错误样例；区分事实与未知，不编造验证结果")
    expected_benefit: str = Field(min_length=10, max_length=1000, description="继续训练能验证什么尚未解决的假设；结束时说明为何当前没有值得追加训练的方向，不承诺数值收益")
    cost_tradeoff: str = Field(min_length=10, max_length=1000, description="结合已用耗时、剩余预算与数据/模型限制，说明预期价值是否值得新增成本；无法估时则说明未知")


class StartArgs(InputModel):
    """参数只写变化项；首轮基于平台默认值和操作者明确填写值，后续继承父实验。"""

    parameters: dict = Field(default_factory=dict)
    dataset_export_id: str | None = None
    parent_experiment_id: str | None = None
    comparison_mode: Literal["joint_search", "ablation"] = Field(default="joint_search", description="默认联合搜索可改多项，只评价配置整体；定向消融用于有预算时验证一个变量")
    target_parameter: str | None = Field(default=None, max_length=100)
    hypothesis: str | None = Field(default=None, min_length=10, max_length=1000)
    value_assessment: TrainingValueAssessment | None = Field(default=None, description="首轮可省略；后续开训必须提供继续训练的必要性判断，不能仅因还有预算就继续")
    reason: str = Field(min_length=10, max_length=2000)


class ReasonArgs(InputModel):
    """暂停训练需要留下可审计的理由。"""

    reason: str = Field(min_length=4, max_length=2000)


class ResumeArgs(InputModel):
    """恢复只针对本运行实验的合法检查点。"""

    experiment_id: str
    reason: str = Field(min_length=4, max_length=2000)
    value_assessment: TrainingValueAssessment | None = Field(default=None, description="真正恢复训练前必须说明剩余训练的预期价值与成本，已在运行时可省略")


class ExportArgs(InputModel):
    """部署格式为固定白名单，导出不等于发布。"""

    experiment_id: str
    kind: Literal["best", "last"] = "best"
    format: Literal["onnx", "torchscript"]
    reason: str = Field(min_length=4, max_length=2000)


class SleepArgs(InputModel):
    """训练耗时长，由模型自选休眠时长，任务结束会提前唤醒。"""

    minutes: float = Field(ge=0.5, le=720)
    reason: str = Field(min_length=4, max_length=1000)


class ReviewArgs(InputModel):
    """训练无法解决的数据问题交给人处理，模型不能自行批准。"""

    reason: str = Field(min_length=10, max_length=2000)
    data_actions: list[str] = Field(default_factory=list, max_length=10)


class FinishArgs(InputModel):
    """结束运行时给出模型认为的最佳实验，报告仍会核对证据。"""

    best_experiment_id: str | None = None
    reason: str = Field(min_length=10, max_length=2000)
    summary: str = Field(default="", max_length=4000)
    value_assessment: TrainingValueAssessment | None = Field(default=None, description="主动提前结束时给出收益递减、目标达成或现有限制的证据；停止不要求用完停滞轮数或预算")


class NoArgs(InputModel):
    """无参数工具。"""


class OperatorAnswerArgs(OperatorInstructionChange):
    """回答训练档案中真实存在的操作者问题。"""

    request_id: int = Field(ge=1)
    answer: str = Field(min_length=1, max_length=4000)


TOOLS = {
    "inspect_training_parameters": (ParameterInfoArgs, "按 names 查询本平台当前真正支持的参数、现值来源、模型锁定和本运行 Schema；技能提及不代表参数已开放。最多八项，未开放项明确返回 supported=false。"),
    "read_dataset_examples": (DatasetExamplesArgs, "首轮前也可查看已审封存 train/val 样例与真值框；按类别或 image_ids 筛选，分页最多两图。不能读取 test，原图和标注均核对封存指纹。标注可用 box_offset 继续读取。"),
    "read_validation_errors": (ValidationErrorsArgs, "按真实类名、漏检 fn/误检 fp、小目标漏检筛选固定 val 错误，按错误数排序并分页，最多两图。需要已有完成评估。只读已保存匹配结果，不重新推理；可按 image_id 与 box_offset 继续读取标注。"),
    "compare_experiments": (CompareExperimentsArgs, "直接比较本运行两个指定实验的参数与逐类指标，不限父子关系。同一数据和兼容评估协议才返回指标增量；否则明确不可比。类别与分页可定向查询，不证明单项因果。"),
    "preview_experiment": (StartArgs, "开训前只读预检候选参数、首轮约束、实验设计、数据、预算和训练槽；不创建实验、Optuna 试验或训练。与 start_experiment 共用校验；valid=true 但 can_start=false 表示配置有效、当前仍须等待。"),
    "answer_operator": (OperatorAnswerArgs, "回答真实操作者问题。明确的后续持续要求用 instruction_action=remember、稳定 instruction_key 保存原文；休眠周期固定用 key=sleep_interval 并填写 sleep_minutes。新要求用相同 key 替换旧要求；明确取消用 forget。普通提问用 none。不能借此修改冻结预算、模型白名单或批准标注。"),
    "read_chart": (ChartArgs, "读取本运行实验的图：training 来源有 results 曲线、labels 标签分布、train_batch、val 混淆矩阵与 PR/F1/P/R 曲线、val 预测样例；evaluation 来源是固定 val 协议评估的同类图。一次最多 4 张，图像只在下一轮可见，请在之后的 reason 中写下观察。"),
    "get_experiment": (ExperimentInspectArgs, "定向读取本运行实验：overview 概览；parameters 按 parameter_names 查请求值与实际值；curve 按 metric_names 和 offset/limit 查未抽样的逐 epoch 表；evaluation 按 class_names 分页读逐类指标；training 查训练尝试。避免一次索取全部结果导致截断。"),
    "read_confusion_matrix": (ConfusionReadArgs, "分页读取固定 val 混淆矩阵结构化证据：真值→预测方向、计数、真值分母与漏检比例。可用 class_names 指定类别，默认五类。禁止从图片猜轴或读数。"),
    "read_training_log": (LogArgs, "读取实验最近一次训练尝试的日志末尾，用于排查失败或异常。"),
    "suggest_parameters": (NoArgs, "向 Optuna TPE 请求下一轮受限参数建议，仅供参考，可以不采用；采样器仍能给建议不代表继续训练有价值。"),
    "start_experiment": (StartArgs, "新建实验并立即开始训练。首轮建立基线；后续必须填写 hypothesis 和 value_assessment，说明证据、预期价值及成本权衡，没有值得验证的方向时应 finish，无需等到停滞阈值。默认 joint_search 比较整组配置；必要且有预算时才做单变量 ablation。"),
    "stop_training": (ReasonArgs, "暂停本运行正在进行的训练：在批次边界保存检查点后进入 stopped，之后可用 resume_training 继续，或放弃并开新实验。"),
    "resume_training": (ResumeArgs, "从检查点继续已暂停或失败的实验，参数不变。真正恢复前必须提供 value_assessment；检查点可恢复不代表值得继续，过拟合或数据瓶颈未解决时不应无意义恢复。"),
    "evaluate": (ExperimentArgs, "对已完成实验的 best 权重提交固定 val 协议评估。训练完成后系统会自动提交，一般只在评估失败后使用。"),
    "export_model": (ExportArgs, "把实验的 best/last 权重导出为 ONNX 或 TorchScript，并用对应运行时加载校验。导出不等于生产发布。"),
    "sleep": (SleepArgs, "休眠指定分钟数。期间训练、评估或导出任务结束会提前唤醒你。"),
    "request_human_review": (ReviewArgs, "遇到训练无法解决的数据问题（标注错误、类别缺样本、验证集不具代表性）时请求人工处理，流程暂停到操作者确认。"),
    "finish": (FinishArgs, "可在停滞阈值或预算耗尽之前主动结束本次运行，保留最佳实验并封存报告。目标已有证据达成、收益不抵成本、无有据可行的新假设或现有数据/锁定模型限制收益时应考虑结束，value_assessment 解释依据和未解决的问题。不要仅因预算尚余而继续，也不能无证据宣称达标；生产发布仍由人工完成。"),
}
CONTROL_TOOLS = {"sleep", "request_human_review", "finish"}


def _round(value):
    return float(f"{value:.5g}") if isinstance(value, float) and math.isfinite(value) else value


def _clip(value, limit: int = 300):
    """决策记录只保留短文本，长理由已在事件中完整保存。"""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        return {key: _clip(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_clip(item, limit) for item in value[:20]]
    return value


class WorkflowTools:
    """此对象没有任意 Shell、URL、下载、删除、审批或发布方法。"""

    def __init__(self, service: AgentService, run_id: str):
        """绑定唯一运行，工具参数不能改变项目或数据归属。

        :param service: 当前控制器业务服务。
        :param run_id: 已冻结的运行身份。
        :returns: 无。
        """
        self.service, self.run_id = service, run_id
        self.engine, self.settings = service.engine, service.settings
        self.datasets = DatasetService(self.engine, self.settings)
        self.annotations = AnnotationService(self.engine, self.settings)
        self.training = TrainingService(self.engine, self.settings)
        self.evaluation = EvaluationService(self.engine, self.settings)
        self.exports = ExportService(self.engine, self.settings)

    def identity(self, key: str) -> str:
        """标准 UUID5 让同一节点重放时使用同一业务身份。

        :param key: 控制器生成的节点、轮次及有限尝试序号。
        :returns: 不接受模型指定的确定性 UUID。
        """
        return str(uuid5(UUID(self.run_id), key))

    def guard(self, operation: str | None = None) -> None:
        """在实际副作用之前重新检查人工控制和运行白名单。

        :param operation: 固定业务操作名；空值只检查控制信号。
        :returns: 无；拒绝时抛出明确异常。
        """
        run = self.service.get(self.run_id)
        if run.stop_requested:
            raise WorkflowStopped("操作者要求停止")
        if run.pause_requested:
            raise WorkflowPaused("操作者要求暂停")
        if operation and operation not in run.policy["allowed_operations"]:
            raise PermissionError(f"本次运行未授权工具：{operation}")

    def policy(self) -> AgentPolicy:
        """读取创建时冻结的预算，模型无法修改。

        :returns: 本运行策略。
        """
        return AgentPolicy.model_validate(self.service.get(self.run_id).policy)

    def prepared_models(self) -> list[str]:
        """只列出白名单内且来源文件齐全的本地预训练权重。

        :returns: 可用于训练的模型文件名。
        """
        root = self.settings.training_model_dir
        return [name for name in self.policy().allowed_models if (root / name).is_file() and (root / name).with_suffix(".provenance.json").is_file()]

    def study(self) -> OptimizationStudy:
        """每次按冻结策略打开同一持久 study。

        :returns: 本运行的 Optuna 研究。
        """
        return OptimizationStudy(self.settings, self.run_id, self.policy())

    def search_space(self) -> dict:
        return describe_search_space(search_distributions(self.settings, self.policy()))

    # ---- 前半段固定节点使用的工具 ----

    def analyze(self):
        """对本运行数据集调用既有分析器，不接受任意文件路径。

        :returns: 确定性分析任务。
        """
        self.guard("analyze_dataset")
        run = self.service.get(self.run_id)
        return self.datasets.schedule(run.dataset_id, "analysis", AnalysisCreate(), task_id=self.identity("analysis"))

    def compute_environments(self) -> dict:
        """读取计算 Worker 的近期报告，不把控制面 CPU 当作未来训练设备。

        :returns: 有采集时间的精简报告；缺失时返回明确原因。
        """
        reports = []
        with redis_connection(self.settings) as connection:
            for index, key in enumerate(connection.scan_iter(match=f"{RUNTIME_PREFIX}*", count=20)):
                if index >= 20:
                    break
                raw = connection.get(key)
                if raw:
                    report = json.loads(raw)
                    reports.append({name: report.get(name) for name in ("collected_at", "worker_role", "queue_name", "os", "architecture", "effective_device", "gpu_count", "gpus", "torch_version", "cuda_runtime_version", "selection_reason", "warnings")})
        return {"workers": reports, "unavailable_reason": None if reports else "没有近期计算 Worker 报告，不能假定计算设备可用", "note": "真正接单的 Worker 会重新检测设备，排队最长等待受单独上限约束"}

    def split(self):
        """沿已有分组规则生成不可变划分，测试集不参与优化。

        :returns: 版本封存任务。
        """
        self.guard("analyze_dataset")
        run = self.service.get(self.run_id)
        return self.datasets.schedule(run.dataset_id, "version", VersionCreate(name=f"Agent {self.run_id[:8]} 冻结版本"), task_id=self.identity("split"))

    def annotate(self, attempt: int = 0):
        """使用当前类别与封存版本创建真实 DINO 候选，不自动审核。

        :param attempt: 预算内的重新推理次数，失败的部分候选另批保留。
        :returns: 标注批次及持久任务。
        """
        self.guard("auto_annotate")
        run = self.service.get(self.run_id)
        return self.annotations.create_run(AutoAnnotateCreate(version_id=run.dataset_version_id, class_schema_id=run.class_schema_id, class_sources=run.state.get("annotation_sources", {}), box_threshold=run.state.get("annotation_box_threshold", 0.35), coco_model=run.state.get("annotation_coco_model", "yolo26n.pt")), run_id=self.identity("annotation_run" if not attempt else f"annotation_run:{attempt}"), task_id=self.identity("annotation" if not attempt else f"annotation:{attempt}"), execution_limits={"gpu_memory_ratio": run.policy["max_gpu_memory_ratio"]})

    def annotation_operation(self, operation: str, *, attempt: int = 0):
        """仅允许已有批次的上传、人审回传和导出。

        :param operation: cvat_push、cvat_pull 或 yolo_export。
        :param attempt: 控制器在预算内的重试序号。
        :returns: 固定标注任务。
        """
        self.guard("auto_annotate")
        run = self.service.get(self.run_id)
        parameters = {}
        if operation == "cvat_pull":
            if not run.review_confirmation:
                raise PermissionError("必须由人工恢复入口提供审核确认")
            parameters = CvatReviewConfirm.model_validate({key: value for key, value in run.review_confirmation.items() if key in CvatReviewConfirm.model_fields}).model_dump()
        revision = (run.review_confirmation or {}).get("_revision", 0)
        identity = f"{operation}:{revision}:{attempt}" if operation == "cvat_pull" else f"{operation}:{attempt}"
        return self.annotations.schedule(run.state.get("annotation_run_id", self.identity("annotation_run")), operation, parameters, task_id=self.identity(identity))

    def train(self, experiment_id: str, *, resume: bool = False):
        """只执行本运行实验，按剩余预算收紧 RQ 硬超时及显存上限。

        :param experiment_id: 本运行的固定实验。
        :param resume: 是否恢复合法检查点。
        :returns: 异步训练任务。
        """
        self.guard("train")
        run = self.service.get(self.run_id)
        self._own(experiment_id)
        remaining = self.service.budget(self.run_id)["remaining_seconds"]
        if remaining < 1:
            raise DatasetError(409, "训练用时预算已耗尽，请导出已有模型或结束运行")
        limits = ExecutionLimits(gpu_memory_ratio=min(self.settings.max_gpu_memory_ratio, run.policy["max_gpu_memory_ratio"]), timeout_seconds=max(1, min(self.settings.training_job_timeout, int(remaining))))
        return self.training.start(experiment_id, resume=resume, execution_limits=limits)

    def report(self, evaluation_id: str) -> dict:
        """只读取本运行实验的有效固定 val 报告。

        :param evaluation_id: 完成评估任务。
        :returns: 哈希一致且未失效的真实报告。
        """
        self.guard("validate")
        task = self.evaluation.get(evaluation_id)
        with Session(self.engine) as session:
            owned = session.scalar(select(AgentIteration).where(AgentIteration.run_id == self.run_id, AgentIteration.experiment_id == task.experiment_id))
            invalid = session.get(EvaluationInvalidation, evaluation_id)
        if not owned or invalid or task.status != "completed" or task.parameters != EvaluationCreate().model_dump():
            raise DatasetError(409, "只能使用本运行、默认协议且未失效的完成评估")
        if not task.report_path:
            raise DatasetError(409, "验证报告尚未生成")
        path = owned_path(self.settings, task.report_path)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != task.report_sha256:
            raise DatasetError(409, "验证报告指纹变化")
        report = json.loads(path.read_text())
        identity = {"evaluation_id": task.id, "experiment_id": task.experiment_id, "model_id": task.model_id,
            "dataset_export_id": task.dataset_export_id, "parameters": task.parameters, "split": "val", "test_used": False}
        if any(report.get(key) != value for key, value in identity.items()):
            raise PermissionError("评估协议或数据版本漂移")
        return report

    # ---- 训练段共享的只读查询 ----

    def iterations(self) -> list[tuple[AgentIteration, Experiment]]:
        """按轮次列出本运行实验。

        :returns: 迭代记录与实验快照。
        """
        with Session(self.engine) as session:
            rows = list(session.scalars(select(AgentIteration).where(AgentIteration.run_id == self.run_id).order_by(AgentIteration.number)))
            result = [(row, session.get(Experiment, row.experiment_id)) for row in rows]
            session.expunge_all()
            return result

    def _own(self, experiment_id: str) -> tuple[AgentIteration, Experiment]:
        for item, experiment in self.iterations():
            if experiment.id == experiment_id:
                return item, experiment
        raise PermissionError("只能操作本运行创建的实验")

    def _tasks(self, experiment_id: str) -> list[TrainingTask]:
        with Session(self.engine) as session:
            return list(session.scalars(select(TrainingTask).where(TrainingTask.experiment_id == experiment_id).order_by(TrainingTask.created_at)))

    def _models(self, experiment_id: str) -> list[ModelArtifact]:
        with Session(self.engine) as session:
            return list(session.scalars(select(ModelArtifact).where(ModelArtifact.experiment_id == experiment_id).order_by(ModelArtifact.created_at.desc())))

    def _evaluations(self, experiment_id: str) -> list[EvaluationTask]:
        """默认协议、未失效的评估，按创建时间排序。"""
        with Session(self.engine) as session:
            rows = session.scalars(select(EvaluationTask).where(EvaluationTask.experiment_id == experiment_id, ~EvaluationTask.id.in_(select(EvaluationInvalidation.task_id))).order_by(EvaluationTask.created_at))
            return [row for row in rows if row.parameters == EvaluationCreate().model_dump()]

    def _active_training(self) -> list[TrainingTask]:
        with Session(self.engine) as session:
            return list(session.scalars(select(TrainingTask).join(AgentIteration, AgentIteration.experiment_id == TrainingTask.experiment_id).where(AgentIteration.run_id == self.run_id, TrainingTask.status.in_(ACTIVE))))

    def active_task_refs(self) -> list[dict]:
        """休眠期间需要盯住的本运行计算，任一结束即唤醒模型。

        :returns: 任务类型与 ID。
        """
        refs = [{"kind": "training", "id": task.id} for task in self._active_training()]
        for _, experiment in self.iterations():
            refs += [{"kind": "evaluation", "id": task.id} for task in self._evaluations(experiment.id) if task.status in ACTIVE]
            model_ids = [model.id for model in self._models(experiment.id)]
            refs += [{"kind": "export", "id": record.id} for record in self.exports.list(model_ids=model_ids) if record.status in ACTIVE and record.requested_by == f"agent:{self.run_id}"]
        return refs

    def allowed_exports(self) -> list[dict]:
        """同项目、同类别定义且已人审封存的数据工件，切换数据集只能在其中选择。

        :returns: 工件身份、规模与 train/val 标注画像。
        """
        run = self.service.get(self.run_id)
        if not run.dataset_export_id:
            return []
        with Session(self.engine) as session:
            base = session.get(DatasetExport, run.dataset_export_id)
            classes = validate_export(self.settings, base, verify_files=False)["classes"]
            rows = list(session.scalars(select(DatasetExport).join(DatasetVersion, DatasetVersion.id == DatasetExport.version_id).join(Dataset, Dataset.id == DatasetVersion.dataset_id).join(AnnotationRun, AnnotationRun.id == DatasetExport.run_id).where(Dataset.project_id == run.project_id, Dataset.deleted_at.is_(None), DatasetVersion.sealed.is_(True), AnnotationRun.status == "exported").order_by(DatasetExport.created_at)))
        result = []
        for export in rows:
            try:
                manifest = validate_export(self.settings, export, verify_files=False)
            except DatasetError:
                continue
            if manifest["classes"] != classes:
                continue
            counts = manifest["summary"]["split_counts"]
            result.append({"dataset_export_id": export.id, "is_run_entry": export.id == run.dataset_export_id, "created_at": export.created_at, "images": {"train": counts.get("train", 0), "val": counts.get("val", 0)}, "label_profile": self._export_profile(export, manifest)})
        return result

    def _export_profile(self, export: DatasetExport, manifest: dict) -> dict:
        """统计 train/val 逐类框数与尺寸分布；工件不可变，按指纹缓存。"""
        key = (export.id, export.manifest_sha256)
        if key in _EXPORT_STATS:
            return _EXPORT_STATS[key]
        root = owned_path(self.settings, export.directory)
        names = {item["class_id"]: item["name"] for item in manifest["classes"]}
        profile = {}
        for split in ("train", "val"):
            areas = {class_id: [] for class_id in names}
            empty = 0
            for file in sorted((root / "labels" / split).glob("*.txt")):
                lines = [line.split() for line in file.read_text().splitlines() if line.strip()]
                empty += not lines
                for parts in lines:
                    class_id = int(parts[0])
                    if class_id in areas:
                        areas[class_id].append(float(parts[3]) * float(parts[4]))
            classes = {}
            for class_id, values in areas.items():
                values.sort()
                classes[names[class_id]] = {"boxes": len(values), "median_area_ratio": _round(values[len(values) // 2]) if values else None, "small_share": _round(sum(value < SMALL_AREA for value in values) / len(values)) if values else None}
            profile[split] = {"images_without_boxes": empty, "classes": classes}
        profile["small_definition"] = "归一化面积 < (32/640)^2，约等于 640 输入下小于 32×32 像素"
        _EXPORT_STATS[key] = profile
        return profile

    def epoch_table(self, experiment_id: str, *, limit: int = 30) -> dict:
        """合并全部训练尝试的逐 epoch 指标，恢复训练沿用原 epoch 编号。

        :param experiment_id: 实验。
        :param limit: 返回行数上限，超出时保留首尾并均匀抽样中段。
        :returns: 列名、行数据与派生信号。
        """
        with Session(self.engine) as session:
            rows = list(session.execute(select(ExperimentMetric.epoch, ExperimentMetric.name, ExperimentMetric.value).join(TrainingTask, TrainingTask.id == ExperimentMetric.task_id).where(ExperimentMetric.experiment_id == experiment_id).order_by(TrainingTask.created_at, ExperimentMetric.epoch)))
        by_epoch: dict[int, dict] = {}
        for epoch, name, value in rows:
            by_epoch.setdefault(epoch, {})[name] = value
        if not by_epoch:
            return {"columns": [], "rows": [], "epochs": 0}
        names = sorted({name for values in by_epoch.values() for name in values}, key=lambda name: (not name.startswith("metrics/"), not name.startswith("val/"), name))
        epochs = sorted(by_epoch)
        table = [[epoch, *[_round(by_epoch[epoch].get(name)) for name in names]] for epoch in epochs]
        signals = {}
        key = "metrics/mAP50-95(B)"
        if key in names:
            series = [(epoch, by_epoch[epoch].get(key)) for epoch in epochs if by_epoch[epoch].get(key) is not None]
            if series:
                best_epoch, best = max(series, key=lambda item: item[1])
                signals.update(best_epoch=best_epoch, best_map50_95=_round(best), last_map50_95=_round(series[-1][1]), epochs_since_best=epochs[-1] - best_epoch)
        for name in ("train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss"):
            series = [by_epoch[epoch].get(name) for epoch in epochs[-5:] if by_epoch[epoch].get(name) is not None]
            if len(series) >= 2:
                signals[f"{name}_change_last_{len(series)}"] = _round(series[-1] - series[0])
        if len(table) > limit:
            keep = set(range(3)) | set(range(len(table) - limit // 2, len(table)))
            step = max(1, (len(table) - len(keep)) // max(1, limit - len(keep)))
            keep |= set(range(3, len(table), step))
            table = [row for index, row in enumerate(table) if index in keep][: limit + 3]
        return {"columns": ["epoch", *names], "rows": table, "epochs": len(epochs), "downsampled": len(table) < len(epochs), "signals": signals}

    def _chart_dir(self, experiment_id: str, source: str) -> str | None:
        if source == "training":
            for task in reversed(self._tasks(experiment_id)):
                relative = f"training/{task.id}/run"
                if owned_path(self.settings, relative).is_dir():
                    return relative
            return None
        for task in reversed(self._evaluations(experiment_id)):
            if task.status == "completed":
                return f"evaluations/{task.id}/validation"
        return None

    def available_charts(self, experiment_id: str) -> dict:
        """列出实际存在的图，模型据此选择 read_chart 参数。

        :param experiment_id: 实验。
        :returns: 来源到图名的映射。
        """
        result = {}
        for source in ("training", "evaluation"):
            directory = self._chart_dir(experiment_id, source)
            if directory:
                root = owned_path(self.settings, directory)
                result[source] = [name for name, file in CHARTS.items() if (root / file).is_file()]
        return result

    def evaluation_digest(self, experiment_id: str, *, samples: int = 5) -> dict | None:
        """最近一次默认 val 评估的逐类结果、混淆矩阵与错误样本。

        :param experiment_id: 实验。
        :param samples: 错误样本条数。
        :returns: 评估摘要；没有评估时为 None。
        """
        evaluations = self._evaluations(experiment_id)
        if not evaluations:
            return None
        task = evaluations[-1]
        if task.status != "completed":
            return {"evaluation_id": task.id, "status": task.status, "error": task.error}
        report = self.report(task.id)
        error_path = owned_path(self.settings, f"evaluations/{task.id}/errors.jsonl")
        with error_path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != report["artifacts"]["errors.jsonl"]:
                raise DatasetError(409, "错误清单指纹变化")
        errors = []
        with error_path.open() as stream:
            for line in stream:
                record = json.loads(line)
                errors.append({"image_id": record["image_id"], "counts": record["counts"], "ground_truth": record["ground_truth"][:8], "predictions": record["predictions"][:8]})
                if len(errors) >= samples:
                    break
        matrix = report["confusion_matrix"]
        per_class = [{key: _round(row.get(key)) for key in ("name", "train_instances", "validation_instances", "precision", "recall", "ap50", "ap50_95", "unavailable_reason")} for row in report["per_class"]]
        return {"evaluation_id": task.id, "status": "completed", "summary": report["summary"], "metric_protocol": report["metric_protocol"], "per_class": per_class, "confusion_matrix": matrix if len(matrix["labels"]) <= 21 else "请用 read_confusion_matrix 读取方向与计数", "confusion_evidence": self._confusion_evidence(task), "error_samples": errors}

    def _confusion_evidence(self, task: EvaluationTask) -> dict:
        report = self.report(task.id)
        try:
            evidence = confusion_evidence(report)
        except ValueError as error:
            return {"status": "unavailable", "reason": str(error), "rule": "禁止从图片替代推算方向和数字"}
        return {"status": "completed", "report_sha256": task.report_sha256, **evidence}

    def read_confusion_matrix(self, experiment_id: str, *, class_names: list | None = None, offset: int = 0, limit: int = 5) -> dict:
        self.guard()
        self._own(experiment_id)
        completed = [task for task in self._evaluations(experiment_id) if task.status == "completed"]
        if not completed:
            return {"status": "unavailable", "reason": "尚无有效固定 val 评估，请等待；禁止根据训练图猜测矩阵数值"}
        evidence = self._confusion_evidence(completed[-1])
        if evidence.get("status") != "completed":
            return evidence
        rows = evidence["per_true_class"]
        if set(class_names or []) - {row["true_class"] for row in rows}:
            raise ValueError("请使用混淆矩阵中的实际类别名")
        rows = [row for row in rows if not class_names or row["true_class"] in class_names]
        selected = rows[offset:offset + limit]
        names = {row["true_class"] for row in selected}
        return {**evidence, "per_true_class": selected, "total_true_classes": len(rows), "offset": offset,
            "has_more": offset + limit < len(rows),
            "top_confusions": [row for row in evidence["top_confusions"] if row["true_class"] in names],
            "unmatched_predictions_by_predicted_class": [row for row in evidence["unmatched_predictions_by_predicted_class"] if row["predicted_class"] in names]}

    def experiment_brief(self, item: AgentIteration, experiment: Experiment, *, full: bool) -> dict:
        """单个实验的档案条目，full=False 时只保留比较所需字段。

        :param item: 迭代记录。
        :param experiment: 实验快照。
        :param full: 是否包含曲线、尝试与产物明细。
        :returns: 可 JSON 序列化的摘要。
        """
        conclusion = item.conclusion or {}
        brief = {"iteration": item.number, "experiment_id": experiment.id, "status": experiment.status, "dataset_export_id": experiment.dataset_export_id, "changed_variables": experiment.changed_variables, "val_map50_95": conclusion.get("map50_95"), "delta_previous": conclusion.get("delta_previous")}
        if not full:
            return brief
        resolved = experiment.resolved_parameters or {}
        effective = resolved.get("optimizer_effective") or {}
        tasks = self._tasks(experiment.id)
        brief.update(
            parent_experiment_id=experiment.parent_experiment_id, parameters=experiment.parameters, reason=item.plan.get("reason"),
            comparison_mode=item.plan.get("comparison_mode"), target_parameter=item.plan.get("target_parameter"), hypothesis=item.plan.get("hypothesis"),
            value_assessment=item.plan.get("value_assessment"),
            resolved={"device": resolved.get("device"), "batch": resolved.get("batch"), "optimizer": effective.get("class"), "lr": [group.get("lr") for group in effective.get("groups", [])][:1], "trainer_parameters": resolved.get("trainer_parameters"), "augmentation_effective_at_start": resolved.get("augmentation_effective_at_start")},
            attempts=[{"task_id": task.id, "status": task.status, "epoch": f"{task.current_epoch}/{task.total_epochs}", "elapsed_seconds": round(task.elapsed_seconds), "seconds_per_epoch": round(task.elapsed_seconds / task.current_epoch, 1) if task.current_epoch else None, "resumable": bool(task.checkpoint_path) and task.status in {"failed", "stopped"}, "error": (task.error or "")[:300] or None} for task in tasks],
            curve=self.epoch_table(experiment.id), final_train_metrics={key: _round(value) for key, value in (experiment.final_metrics or {}).items()},
            conclusion=conclusion or None, models=[{"model_id": model.id, "kind": model.kind} for model in self._models(experiment.id)],
            exports=[{"format": record.format, "status": record.status, "model_id": record.model_id} for record in self.exports.list(model_ids=[model.id for model in self._models(experiment.id)])],
            charts=self.available_charts(experiment.id),
            comparison=self.comparison(experiment),
        )
        return brief

    def comparison(self, experiment: Experiment) -> dict:
        parent = self._own(experiment.parent_experiment_id)[1] if experiment.parent_experiment_id else None
        return comparison_design(parent, experiment.parameters, experiment.dataset_export_id, resolved=experiment.resolved_parameters)

    def stagnation(self) -> dict:
        """只在同一验证集内统计连续无明显提升，换数据集后重新计数。

        :returns: 停滞进度及所用数据工件。
        """
        rows = [(item, experiment) for item, experiment in self.iterations() if item.conclusion and item.conclusion.get("map50_95") is not None]
        if not rows:
            return {**stagnation_progress([], self.policy()), "dataset_export_id": None}
        export_id = rows[-1][1].dataset_export_id
        observations = [{"iteration": item.number, "map50_95": item.conclusion["map50_95"], "issues": item.conclusion.get("issues", [])} for item, experiment in rows if experiment.dataset_export_id == export_id]
        return {**stagnation_progress(observations, self.policy()), "dataset_export_id": export_id, "latest_issues": observations[-1]["issues"]}

    def decision_log(self, limit: int = 30) -> list[dict]:
        """近期工具调用及其理由，被裁剪的早期对话仍能从这里回忆。

        :param limit: 条数上限。
        :returns: 时间正序的精简记录。
        """
        with Session(self.engine) as session:
            rows = list(session.scalars(select(AgentEvent).where(AgentEvent.run_id == self.run_id, AgentEvent.event_type == "tool_call").order_by(AgentEvent.id.desc()).limit(limit)))
        return [{"at": row.created_at, "tool": row.payload.get("tool"), "arguments": _clip(row.payload.get("arguments"), 200), "ok": row.payload.get("ok"), "result": _clip(row.payload.get("result"), 200)} for row in reversed(rows)]

    def sync_iterations(self) -> None:
        """把已完成评估落成迭代结论并回报 Optuna；被放弃的失败实验标记为放弃。

        :returns: 无；重复调用只处理尚无结论的迭代。
        """
        pairs = self.iterations()
        if not pairs:
            return
        last = pairs[-1][0].number
        study = None
        for item, experiment in pairs:
            if item.conclusion is not None:
                continue
            completed = [task for task in self._evaluations(experiment.id) if task.status == "completed"]
            best = next((model for model in self._models(experiment.id) if model.kind == "best"), None)
            evaluation = next((task for task in reversed(completed) if best and task.model_id == best.id), None)
            if evaluation:
                report = self.report(evaluation.id)
                value = report["summary"]["map50_95"]
                previous = next((other.conclusion.get("map50_95") for other, parent in reversed(pairs) if other.number < item.number and other.conclusion and parent.dataset_export_id == experiment.dataset_export_id and other.conclusion.get("map50_95") is not None), None)
                design = self.comparison(experiment)
                parent_item = self._own(experiment.parent_experiment_id)[0] if experiment.parent_experiment_id else None
                parent_value = (parent_item.conclusion or {}).get("map50_95") if parent_item and design["same_validation_data"] else None
                conclusion = {"map50_95": value, "delta_previous": value - previous if value is not None and previous is not None else None, "delta_parent": value - parent_value if value is not None and parent_value is not None else None, "comparison": design, "changed_variable_count": design["variable_count"], "causal_limit": design["attribution"], "issues": report["summary"]["issues"], "recommendations": report["summary"]["recommendations"], "evaluation_report_sha256": evaluation.report_sha256, "dataset_export_id": experiment.dataset_export_id, "test_used": False}
                event = ("validation_observed", {"iteration": item.number, "evaluation_id": evaluation.id, "summary": report["summary"]})
            elif experiment.status in {"failed", "stopped"} and item.number < last:
                conclusion, evaluation = {"abandoned": True, "status": experiment.status, "map50_95": None, "test_used": False}, None
                value, event = None, ("experiment_abandoned", {"iteration": item.number, "experiment_id": experiment.id, "status": experiment.status})
            else:
                continue
            with Session(self.engine) as session, session.begin():
                row = session.get(AgentIteration, (self.run_id, item.number))
                row.conclusion = conclusion
                if evaluation:
                    row.evaluation_id = evaluation.id
            if item.optuna_trial_number is not None:
                study = study or self.study()
                study.finish(item.optuna_trial_number, value, failed=bool(conclusion.get("abandoned")) and experiment.status == "failed")
            self.service.event(self.run_id, "agent", *event)

    # ---- 暴露给模型的工具 ----

    def skill_knowledge(self) -> dict:
        """档案只列技能索引与已读来源，不把全部技能正文塞入每轮上下文。"""
        skills = training_skills(self.settings)
        if skills is None:
            return {"enabled": False, "available": [], "consulted": {}}
        run = self.service.get(self.run_id)
        return {"enabled": True, "available": skills.catalog(),
            "source": {"repository": skills.manifest["repository"], "revision": skills.manifest["revision"]},
            "consulted": run.state.get("skill_reads", {}),
            "rule": "使用 Skill 按需加载技能，使用 SkillRead 读取其中列出的参考文件；已读来源不代表正文永久保留，对话裁剪后可重新读取。技能是知识参考，参数、数据与预算仍以本次平台档案和工具校验为准。"}

    def _parameter_model(self):
        """按冻结策略、已准备模型和 Optuna 搜索空间动态收紧参数类型。"""
        policy, prepared = self.policy(), self.prepared_models()
        if not prepared:
            raise DatasetError(503, "本次运行没有已准备的允许模型")
        fields = {}
        for name, field in TrainingParameters.model_fields.items():
            if name == "pretrained":
                continue
            optional = deepcopy(field)
            optional.default = None
            fields[name] = (field.annotation | None, optional)
        fields.update({
            "model": (Literal[tuple(prepared)] | None, None),
            "epochs": (int | None, Field(default=None, ge=1, le=policy.max_epochs_per_experiment)),
            "imgsz": (int | None, Field(default=None, ge=64, le=policy.max_imgsz, multiple_of=32)),
        })
        for name, space in self.search_space().items():
            fields[name] = (Literal[tuple(space["choices"])] | None, None) if "choices" in space else (float | None, Field(default=None, ge=space["low"], le=space["high"]))
        return create_model("ParameterChanges", __base__=InputModel, **fields)

    def specs(self) -> list[dict]:
        """生成 OpenAI 兼容工具定义，参数 schema 与执行端校验同源。

        :returns: function 工具列表。
        """
        result = []
        for name, (model, description) in TOOLS.items():
            schema = model.model_json_schema()
            if name in {"start_experiment", "preview_experiment"}:
                schema["properties"]["parameters"] = {**self._parameter_model().model_json_schema(), "description": "只写要调整的训练参数。首轮未写项使用平台默认值及操作者明确填写值，只有 operator_first_round.pinned 是操作者首轮约束，其余可调项首轮即可覆盖；后续未写项继承父实验。所有轮次均遵守模型白名单与预算。"}
            schema.pop("title", None)
            result.append({"type": "function", "function": {"name": name, "description": description, "parameters": schema}})
        skills = training_skills(self.settings)
        if skills is not None:
            result.extend(skills.specs())
        return result

    def control_args(self, name: str, arguments: dict) -> dict:
        """校验休眠、人工复核和结束三类流程控制调用。

        :param name: 控制类工具名。
        :param arguments: 模型给出的参数。
        :returns: 校验后的参数。
        """
        model = TOOLS[name][0]
        result = model.model_validate(arguments).model_dump()
        if name == "sleep":
            interval = next((item for item in self.service.operator_instructions(self.run_id) if item["key"] == "sleep_interval"), None)
            if interval and result["minutes"] != interval["sleep_minutes"]:
                raise ValueError(f"操作者当前要求每次休眠 {interval['sleep_minutes']:g} 分钟，请使用 minutes={interval['sleep_minutes']:g}。任务完成仍会提前唤醒；变更节奏须按操作者新指令替换或撤销。")
        return result

    def dispatch(self, name: str, arguments: dict, *, key: str) -> dict:
        """执行一次非控制类工具调用。

        :param name: 白名单工具名。
        :param arguments: 模型给出的参数，先经 Pydantic 校验。
        :param key: 由轮次与调用位置推导的确定性键，重放时复用同一业务身份。
        :returns: 可 JSON 序列化的结果；以下划线开头的键只在系统内部使用。
        """
        if name in SKILL_TOOLS:
            self.guard("optimize")
            skills = training_skills(self.settings)
            if skills is None:
                raise ValueError("本次平台配置已关闭技能读取")
            output = skills.read(name, arguments)
            run = self.service.get(self.run_id)
            identity = f"{output['skill_name']}/{output['file_name']}"
            source = {key: output[key] for key in ("source", "sha256")}
            self.service.patch(self.run_id, state={"skill_reads": {**run.state.get("skill_reads", {}), identity: source}})
            return output
        if name not in TOOLS or name in CONTROL_TOOLS:
            raise ValueError(f"未知工具：{name}")
        args = TOOLS[name][0].model_validate(arguments)
        if name == "preview_experiment":
            return self.preview_experiment(args)
        readers = {"inspect_training_parameters": "parameters", "read_dataset_examples": "dataset_examples",
            "read_validation_errors": "validation_errors", "compare_experiments": "compare", "get_experiment": "experiment"}
        if name in readers:
            self.guard()
            return getattr(TrainingInspection(self), readers[name])(args)
        if name == "answer_operator":
            self.guard()
            return self.service.answer_operator(self.run_id, args.request_id, args.answer,
                instruction_action=args.instruction_action, instruction_key=args.instruction_key, sleep_minutes=args.sleep_minutes)
        if name == "read_chart":
            return self.read_chart(args)
        if name == "read_confusion_matrix":
            return self.read_confusion_matrix(args.experiment_id, class_names=args.class_names, offset=args.offset, limit=args.limit)
        if name == "read_training_log":
            return self.read_log(args)
        if name == "suggest_parameters":
            self.guard("optimize")
            pairs = self.iterations()
            baseline = TrainingParameters.model_validate(pairs[-1][1].parameters if pairs else self.service.get(self.run_id).initial_parameters)
            candidate = self.study().candidate(len(pairs), baseline)
            return {"iteration": len(pairs), "search_parameters": candidate["search_parameters"], "note": "TPE 联合建议可用于 joint_search；只比较整组配置。若有必要且有预算，可选择一个变量做定向消融，不要求逐参数穷举"}
        if name == "start_experiment":
            return self.start_experiment(args, key=key)
        if name == "stop_training":
            return self.stop_training(args.reason)
        if name == "resume_training":
            return self.resume_training(args.experiment_id, value_assessment=args.value_assessment, reason=args.reason)
        if name == "evaluate":
            return self.evaluate(args.experiment_id)
        return self.export_model(args)

    def read_chart(self, args: ChartArgs) -> dict:
        """返回图像引用，由上下文组装时转成图片输入，图像不进入检查点。

        :param args: 实验、来源与图名。
        :returns: 已附加与缺失的图名。
        """
        self.guard()
        self._own(args.experiment_id)
        directory = self._chart_dir(args.experiment_id, args.source)
        if not directory:
            raise DatasetError(404, f"该实验还没有 {args.source} 图")
        attached, missing, refs = [], [], []
        for chart in dict.fromkeys(args.charts):
            relative = f"{directory}/{CHARTS[chart]}"
            if owned_path(self.settings, relative).is_file():
                attached.append(chart)
                refs.append(relative)
            else:
                missing.append(chart)
        if not refs:
            raise DatasetError(404, f"请求的图都不存在，可用图见档案中的 charts：{missing}")
        result = {"attached": attached, "missing": missing, "note": "图像附在下一条消息里，只在下一轮可见", "_images": refs}
        if set(args.charts) & {"confusion_matrix", "confusion_matrix_normalized"}:
            result["interpretation_rule"] = "类别方向和比例只引用 confusion_evidence 或 read_confusion_matrix 的结构化事实；训练来源图片只作定性观察，等待固定 val 后再给数值诊断。"
            if args.source == "evaluation":
                result["confusion_evidence"] = self.read_confusion_matrix(args.experiment_id)
        return result

    def read_log(self, args: LogArgs) -> dict:
        """读取最近一次尝试的日志末尾，进度条回车只保留最后状态。

        :param args: 实验与行数。
        :returns: 日志文本。
        """
        self.guard()
        self._own(args.experiment_id)
        tasks = self._tasks(args.experiment_id)
        if not tasks:
            raise DatasetError(404, "该实验还没有训练尝试")
        path = owned_path(self.settings, tasks[-1].log_path)
        if not path.is_file():
            return {"task_id": tasks[-1].id, "log": ""}
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 256 * 1024))
            content = stream.read().decode("utf-8", errors="replace")
        lines = [line.split("\r")[-1] for line in content.splitlines() if line.strip()]
        return {"task_id": tasks[-1].id, "status": tasks[-1].status, "log": "\n".join(lines[-args.lines:])}

    def _resolve_candidate(self, args: StartArgs, pairs: list) -> tuple:
        """预检和开训共用参数合并、首轮约束、数据和实验设计校验。"""
        if pairs and args.value_assessment is None:
            raise ValueError("后续训练必须提供 value_assessment：evidence 引用已有证据，expected_benefit 说明尚值得验证的假设，cost_tradeoff 权衡新增成本；没有有据可行的方向时可直接 finish，不必耗尽停滞轮数或预算。")
        run, policy = self.service.get(self.run_id), self.policy()
        parent_pair = self._own(args.parent_experiment_id) if args.parent_experiment_id else (pairs[-1] if pairs else None)
        parent = parent_pair[1] if parent_pair else None
        baseline = parent.parameters if parent else run.initial_parameters
        change_model = self._parameter_model()
        changes = change_model.model_validate(args.parameters).model_dump(exclude_none=True)
        parameters = TrainingParameters.model_validate({**baseline, **changes})
        # 继承值也必须满足当前运行范围，预检不能漏掉 Optuna adopt 的限制。
        change_model.model_validate({name: value for name, value in parameters.model_dump().items() if name in change_model.model_fields})
        if parameters.model not in self.prepared_models() or parameters.epochs > policy.max_epochs_per_experiment or parameters.max_training_imgsz() > policy.max_imgsz:
            raise PermissionError("训练参数超出本运行的模型白名单或单实验上限")
        if not pairs:
            pinned = {field: run.initial_parameters[field] for field in run.state.get("operator_fields", [])}
            if any(parameters.model_dump()[field] != value for field, value in pinned.items()):
                raise ValueError(f"首轮必须使用操作者指定的参数：{pinned}，之后各轮才能调整")
        exports = {item["dataset_export_id"] for item in self.allowed_exports()}
        export_id = args.dataset_export_id or (parent.dataset_export_id if parent else run.dataset_export_id)
        if export_id not in exports:
            raise PermissionError("只能使用本项目内类别定义相同、已人审封存的数据工件")
        design = comparison_design(parent, parameters.model_dump(), export_id)
        validate_experiment_design(design, mode=args.comparison_mode, target=args.target_parameter, hypothesis=args.hypothesis)
        return parameters, export_id, parent_pair, baseline, design

    def _start_readiness(self, parent_pair) -> tuple[dict, list[str]]:
        budget = self.service.budget(self.run_id)
        blocked = []
        if "train" not in self.policy().allowed_operations:
            blocked.append("本次运行未授权训练")
        if budget["experiments"] >= budget["max_experiments"]:
            blocked.append("实验数预算已用完")
        if budget["remaining_seconds"] < 1:
            blocked.append("训练用时预算已耗尽")
        own_active = self._active_training()
        if own_active:
            blocked.append("本运行已有训练在进行，需等待完成或停止后才能开新实验")
        else:
            with Session(self.engine) as session:
                busy = session.scalar(select(TrainingTask.id).where(TrainingTask.resource_slot == "single-machine", TrainingTask.status.in_(ACTIVE)).limit(1))
            if busy:
                blocked.append("单机训练槽已有其它活跃任务，需等待槽位释放")
        if parent_pair:
            item, parent = parent_pair
            evaluation = next((task for task in self._evaluations(parent.id) if task.id == item.evaluation_id and task.status == "completed"), None)
            if parent.status != "completed" or evaluation is None or (item.conclusion or {}).get("map50_95") is None:
                blocked.append("父实验尚无有效固定 val 结论；停止或失败的训练可使用 resume_training")
            else:
                try:
                    self.report(evaluation.id)
                except (DatasetError, PermissionError, ValueError, OSError) as error:
                    blocked.append(f"父实验验证报告不可用：{error}")
        return budget, blocked

    def preview_experiment(self, args: StartArgs) -> dict:
        """只读预检，不创建实验、Optuna 试验、任务或 GPU 计算。"""
        self.guard("create_experiment")
        pairs = self.iterations()
        try:
            parameters, export_id, parent_pair, _, design = self._resolve_candidate(args, pairs)
        except (DatasetError, PermissionError, ValueError) as error:
            return {"valid": False, "can_start": False, "errors": [str(error)[:2500]], "preview_only": True}
        budget, blocked = self._start_readiness(parent_pair)
        warnings = list(design["coupled_effects"])
        if parameters.optimizer == "auto":
            warnings.append("auto 优化器会覆盖 lr0 与 warmup_bias_lr；实际优化器和学习率须等训练器记录。")
        if parameters.rect:
            warnings.append("rect 模式可能关闭不兼容的混合增强，以实际生效配置为准。")
        if parameters.close_mosaic >= parameters.epochs:
            warnings.append("close_mosaic 不小于总 epochs，关闭时点可能落在开始或开始之前，请核对是否符合预期。")
        if parameters.tile:
            warnings.append("切片会增加实际样本与计算量，不能直接套用未切片实验的耗时。")
        previous_matches = [{"experiment_id": experiment.id, "status": experiment.status, "val_map50_95": (item.conclusion or {}).get("map50_95")}
            for item, experiment in pairs if experiment.dataset_export_id == export_id
            and TrainingParameters.model_validate(experiment.parameters).model_dump() == parameters.model_dump()]
        if previous_matches:
            warnings.append("相同数据和完整参数曾被尝试；请先核对 previous_matches 的结果，只有明确的复验价值才继续，不能来回切换已试配置消耗预算。")
        return {"valid": True, "can_start": not blocked, "blocking_reasons": blocked,
            "parameters": parameters.model_dump(), "parent_experiment_id": parent_pair[1].id if parent_pair else None,
            "dataset_export_id": export_id, "comparison": design, "warnings": warnings,
            "previous_matches": previous_matches, "value_assessment": args.value_assessment.model_dump() if args.value_assessment else None,
            "max_training_imgsz": parameters.max_training_imgsz(), "budget": budget,
            "preview_only": True, "note": "预检不预留训练槽或预算；真正开训时重新校验。未估计显存或精度收益。"}

    def start_experiment(self, args: StartArgs, *, key: str) -> dict:
        """新建冻结参数的实验并启动训练；同一调用重放返回同一实验。

        :param args: 参数变化、数据工件、父实验与理由。
        :param key: 确定性调用键。
        :returns: 实验、任务与实际参数。
        """
        self.guard("create_experiment")
        experiment_id = self.identity(f"experiment:{key}")
        pairs = self.iterations()
        existing = next((item for item, experiment in pairs if experiment.id == experiment_id), None)
        if existing is None:
            run = self.service.get(self.run_id)
            parameters, export_id, parent_pair, baseline, design = self._resolve_candidate(args, pairs)
            _, blocked = self._start_readiness(parent_pair)
            if blocked:
                raise DatasetError(409, "；".join(blocked))
            parent = parent_pair[1] if parent_pair else None
            iteration = len(pairs)
            study = self.study()
            candidate = study.candidate(iteration, parameters if iteration == 0 else TrainingParameters.model_validate(baseline))
            number = study.adopt(iteration, candidate, parameters)
            experiment = self.training.create_experiment(ExperimentCreate(dataset_export_id=export_id, name=f"{run.name} · 第 {iteration + 1} 轮", parameters=parameters, parent_experiment_id=parent.id if parent else None, optimization_reason=args.reason), experiment_id=experiment_id)
            plan = {"decision": "train", "reason": args.reason, "comparison_mode": args.comparison_mode, "target_parameter": args.target_parameter, "hypothesis": args.hypothesis, "value_assessment": args.value_assessment.model_dump() if args.value_assessment else None, "comparison": design, "parameters": parameters.model_dump(), "dataset_export_id": export_id, "data_actions": [], "tool_call": key}
            with Session(self.engine) as session, session.begin():
                session.add(AgentIteration(run_id=self.run_id, number=iteration, experiment_id=experiment.id, plan=plan, optuna_trial_number=number))
            self.service.event(self.run_id, "agent", "experiment_created", {"iteration": iteration, "experiment_id": experiment.id, "parent_experiment_id": experiment.parent_experiment_id, "plan": plan, "optuna_trial_number": number})
            existing = self._own(experiment.id)[0]
        experiment = self._own(experiment_id)[1]
        tasks = self._tasks(experiment_id)
        task = tasks[-1] if tasks else self.train(experiment_id)
        return {"iteration": existing.number, "experiment_id": experiment_id, "task_id": task.id, "status": task.status, "parameters": experiment.parameters, "changed_variables": experiment.changed_variables, "comparison": existing.plan.get("comparison"), "optuna_trial_number": existing.optuna_trial_number}

    def stop_training(self, reason: str) -> dict:
        """协作停止本运行训练，批次边界保存检查点。

        :param reason: 停止理由。
        :returns: 各任务停止状态。
        """
        self.guard("train")
        tasks = self._active_training()
        if not tasks:
            raise DatasetError(409, "本运行没有正在进行的训练")
        result = []
        for task in tasks:
            current = self.training.get_task(task.id)
            if current.status in ACTIVE and not current.stop_requested:
                current = self.training.cancel(task.id)
            result.append({"task_id": current.id, "experiment_id": current.experiment_id, "status": current.status, "epoch": f"{current.current_epoch}/{current.total_epochs}"})
        self.service.event(self.run_id, "agent", "training_stop_requested", {"tasks": result, "reason": reason})
        return {"tasks": result, "note": "训练会在批次边界保存检查点后进入 stopped，可 resume_training 继续"}

    def resume_training(self, experiment_id: str, *, value_assessment: TrainingValueAssessment | None = None, reason: str = "") -> dict:
        """从最近一次中断检查点继续，已在执行时直接返回现状。

        :param experiment_id: 本运行实验。
        :returns: 新的或进行中的训练任务。
        """
        self.guard("train")
        self._own(experiment_id)
        tasks = self._tasks(experiment_id)
        if tasks and tasks[-1].status in ACTIVE:
            task = tasks[-1]
        else:
            if value_assessment is None:
                raise ValueError("恢复训练必须提供 value_assessment，先判断剩余训练是否仍有价值；不能仅因存在检查点就继续。")
            task = self.train(experiment_id, resume=True)
            self.service.event(self.run_id, "agent", "training_resume_assessed", {"experiment_id": experiment_id,
                "task_id": task.id, "reason": reason, "value_assessment": value_assessment.model_dump()})
        return {"experiment_id": experiment_id, "task_id": task.id, "status": task.status, "resumed_from": task.parent_task_id}

    def evaluate(self, experiment_id: str) -> dict:
        """为已完成实验的 best 权重提交默认 val 评估，已有有效评估时复用。

        :param experiment_id: 本运行实验。
        :returns: 评估任务状态。
        """
        self.guard("validate")
        _, experiment = self._own(experiment_id)
        if experiment.status != "completed":
            raise DatasetError(409, "只能评估已完成训练的实验，暂停的中间权重不能作为最终结果")
        model = next((model for model in self._models(experiment_id) if model.kind == "best"), None)
        if model is None:
            raise DatasetError(409, "实验没有 best 权重")
        evaluations = [task for task in self._evaluations(experiment_id) if task.model_id == model.id]
        if evaluations and evaluations[-1].status in {"queued", "running", "completed"}:
            task = evaluations[-1]
        else:
            task = self.evaluation.create(model.id, EvaluationCreate(), task_id=self.identity(f"evaluation:{model.id}:{len(evaluations)}"), execution_limits=ExecutionLimits(gpu_memory_ratio=self.policy().max_gpu_memory_ratio, timeout_seconds=min(172800, self.settings.evaluation_job_timeout)))
        return {"evaluation_id": task.id, "status": task.status, "model_id": model.id}

    def export_model(self, args: ExportArgs) -> dict:
        """导出本运行实验权重；同格式已完成或进行中时复用，失败后可重试。

        :param args: 实验、权重类型、格式与理由。
        :returns: 导出任务状态。
        """
        self.guard("export_model")
        self._own(args.experiment_id)
        model = next((model for model in self._models(args.experiment_id) if model.kind == args.kind), None)
        if model is None:
            raise DatasetError(409, f"实验没有 {args.kind} 权重")
        records = [record for record in self.exports.list(model_ids=[model.id]) if record.format == args.format]
        current = next((record for record in records if record.status != "failed"), None)
        if current is None:
            failed = sum(record.status == "failed" for record in records)
            current = self.exports.create(model.id, ModelExportCreate(format=args.format, reason=args.reason), export_id=self.identity(f"export:{model.id}:{args.format}:{failed}"), requested_by=f"agent:{self.run_id}")
        return {"export_id": current.id, "model_id": model.id, "format": current.format, "status": current.status, "note": "导出文件供人工下载；生产发布仍需人工确认"}

