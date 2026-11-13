"""Optuna 持久试验只使用真实验证值，并记录最终实际采用的参数。"""

import math

import optuna
import yaml
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState

from yolo_agent.schemas.agent import AgentPolicy
from yolo_agent.schemas.experiment import TrainingParameters
from yolo_agent.settings import Settings


def stagnation_progress(observations: list[dict], policy: AgentPolicy) -> dict:
    """按同一固定验证集的有效实验结论统计跨实验停滞，不替代训练器内部早停。

    :param observations: 按轮次排序且已校验报告的观测。
    :param policy: 操作者冻结的停滞轮数与绝对 mAP 提升阈值。
    :returns: 基线、实际最佳值、有效提升锚点及连续无有效提升轮数。
    """
    reference=best=None
    stalled=0
    count=0
    last_significant=latest=None
    for row in observations:
        value=row.get("map50_95")
        if value is None or not math.isfinite(value):
            continue
        count+=1
        latest=row["iteration"]
        best=value if best is None else max(best,value)
        gain=value-reference if reference is not None else None
        significant=reference is None or (gain>0 and (gain>=policy.minimum_improvement or math.isclose(gain,policy.minimum_improvement,rel_tol=0,abs_tol=1e-12)))
        if significant:
            reference=value
            last_significant=row["iteration"]
            stalled=0
        else:
            stalled+=1
    return {"metric":"val/mAP50-95","patience":policy.stagnation_rounds,"minimum_improvement":policy.minimum_improvement,"evaluated_rounds":count,"rounds_without_improvement":stalled,"significant_reference":reference,"best_map50_95":best,"last_significant_iteration":last_significant,"latest_iteration":latest,"triggered":stalled>=policy.stagnation_rounds}


def stagnation_recommendations(issues: list[dict]) -> list[str]:
    """根据最近有效评估的具体问题形成停止建议，不再调用模型或追加训练。

    :param issues: 最近一次有效验证报告的问题证据。
    :returns: 优先处理数据质量、困难样本和受控实验的中文建议。
    """
    codes={item.get("code") for item in issues}
    advice=[]
    if "insufficient_validation_support" in codes:
        advice.append("先补充缺少真值类别的独立人工审核验证样本；目前不能可靠判断这些类别的提升。")
    if "class_imbalance" in codes:
        advice.append("优先补充训练集中样本不足的类别，重新审核并封存新版本，保持采集分组隔离。")
    if "missed_objects" in codes:
        advice.append("回看最近验证报告的漏检图片，复查漏标、遮挡与训练样本覆盖，再决定增广策略。")
    if "false_positives" in codes:
        advice.append("检查最近的误检图片及困难负样本，确认类别边界与标注一致性。")
    if "small_object_miss" in codes:
        advice.append("针对已观察到的小目标漏检补充样本，再在显存预算内单独比较更高输入分辨率。")
    if not advice:
        advice.append("先核对验证集代表性、标注一致性和逐类支持量，再检查训练收敛与实际采用的优化器。")
    advice.append("保留当前最佳候选与报告；完成上述检查后新建有预算的运行，优先做单变量对比，保持测试集隔离，避免只增加训练轮数。")
    return advice


class OptimizationStudy:
    """每个 Agent 运行独立 study，复用成熟 TPE 和 ask/tell。"""

    def __init__(self, settings: Settings, run_id: str, policy: AgentPolicy):
        """创建隔离存储和受预算约束的搜索空间。

        :param settings: 操作者配置的搜索文件与存储位置。
        :param run_id: 运行 ID，不能来自模型输出。
        :param policy: 已冻结的预算。
        :returns: 无。
        """
        self.settings, self.run_id, self.policy = settings, run_id, policy
        root = settings.storage_dir.resolve() / "agents"
        root.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_load(settings.optimization_config_path.read_text())["search_space"]
        images = [int(value) for value in content["imgsz"] if 64<=int(value)<=policy.max_imgsz and int(value)%32==0]
        if not images:
            images = [policy.max_imgsz]
        self.distributions = {"imgsz":CategoricalDistribution(images),"optimizer":CategoricalDistribution(content["optimizer"]),"mosaic":CategoricalDistribution(content["mosaic"])}
        for key in ("lr0","weight_decay"):
            entry = content[key]
            self.distributions[key] = FloatDistribution(float(entry["low"]),float(entry["high"]),log=bool(entry.get("log",False)))
        storage = optuna.storages.RDBStorage(url=f"sqlite:///{root/'optuna.db'}",engine_kwargs={"connect_args":{"timeout":5}})
        self.study = optuna.create_study(storage=storage,study_name=f"agent-{run_id}",load_if_exists=True,direction="maximize",sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=1))

    def candidate(self, iteration: int, baseline: TrainingParameters) -> dict:
        """生成受限建议，重放同一轮时复用已有持久候选。

        :param iteration: 从零开始的实验轮次。
        :param baseline: 操作者或上一轮配置。
        :returns: Optuna 编号、搜索建议和完整参数。
        """
        existing = next((trial for trial in reversed(self.study.get_trials(deepcopy=False)) if trial.user_attrs.get("iteration")==iteration and trial.state==TrialState.RUNNING),None)
        if existing:
            params = {**baseline.model_dump(),**existing.params}
            return {"trial_number":existing.number,"search_parameters":existing.params,"parameters":params}
        if iteration==0:
            values = {key:baseline.model_dump()[key] for key in self.distributions}
            self._validate_search(values)
            self.study.enqueue_trial(values,user_attrs={"iteration":iteration,"stage":"candidate"})
        trial = self.study.ask(fixed_distributions=self.distributions)
        trial.set_user_attr("iteration",iteration)
        trial.set_user_attr("stage","candidate")
        return {"trial_number":trial.number,"search_parameters":trial.params,"parameters":{**baseline.model_dump(),**trial.params}}

    def adopt(self, iteration: int, candidate: dict, parameters: TrainingParameters) -> int:
        """LLM 修改候选时另立真实试验，不能把不同参数的结果冒记给原候选。

        :param iteration: 当前轮次。
        :param candidate: 持久 Optuna 建议。
        :param parameters: 最终校验通过的训练参数。
        :returns: 与真实执行参数对应的试验编号。
        """
        values = {key:parameters.model_dump()[key] for key in self.distributions}
        self._validate_search(values)
        if values == candidate["search_parameters"]:
            number = candidate["trial_number"]
        else:
            self.study.tell(candidate["trial_number"],state=TrialState.PRUNED,skip_if_finished=True)
            existing = next((trial for trial in reversed(self.study.get_trials(deepcopy=False)) if trial.user_attrs.get("iteration")==iteration and trial.user_attrs.get("stage")=="adopted" and trial.params==values and trial.state==TrialState.RUNNING),None)
            if existing:
                return existing.number
            self.study.enqueue_trial(values,user_attrs={"iteration":iteration,"stage":"adopted","training_parameters":parameters.model_dump()})
            number = self.study.ask(fixed_distributions=self.distributions).number
        # 完整配置在业务迭代记录中封存，Optuna 仅搜索声明的维度。
        return number

    def finish(self, trial_number: int, value: float | None, *, failed: bool=False) -> None:
        """只提交真实验证集 mAP50-95，缺失目标值不能伪装成零。

        :param trial_number: 实际执行试验。
        :param value: 已验证报告中的真实值。
        :param failed: 训练或评估是否失败。
        :returns: 无。
        """
        if failed:
            self.study.tell(trial_number,state=TrialState.FAIL,skip_if_finished=True)
        elif value is None or not math.isfinite(value):
            self.study.tell(trial_number,state=TrialState.PRUNED,skip_if_finished=True)
        else:
            self.study.tell(trial_number,float(value),skip_if_finished=True)

    def search_space(self) -> dict:
        """把受限分布提供给模型，模型不能修改分布或文件路径。

        :returns: 无可执行内容的搜索空间。
        """
        result = {}
        for name,distribution in self.distributions.items():
            if isinstance(distribution,CategoricalDistribution):
                result[name] = {"choices":list(distribution.choices)}
            else:
                result[name] = {"low":distribution.low,"high":distribution.high,"log":distribution.log}
        return result

    def _validate_search(self, values: dict) -> None:
        for name,value in values.items():
            distribution = self.distributions[name]
            if isinstance(distribution,CategoricalDistribution):
                if value not in distribution.choices:
                    raise ValueError(f"参数 {name} 不在操作者声明的搜索集合")
            elif not distribution.low <= value <= distribution.high:
                raise ValueError(f"参数 {name} 超出操作者声明的搜索范围")
