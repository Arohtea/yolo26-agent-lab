"""按同一固定验证协议选择候选，产出可追溯结论，保留发布确认。"""

import hashlib
import json
import math
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from yolo_agent.agents.evidence import confusion_evidence
from yolo_agent.agents.experiment_design import comparison_design
from yolo_agent.models.agent import AgentIteration
from yolo_agent.models.dataset import utc_now
from yolo_agent.models.evaluation import EvaluationInvalidation, EvaluationTask
from yolo_agent.models.experiment import Experiment, ModelArtifact, TrainingTask
from yolo_agent.schemas.evaluation import EvaluationCreate
from yolo_agent.services.agents import AgentService
from yolo_agent.services.datasets import owned_path


def finalize_run(service: AgentService, run_id: str, reason: str, *, status: str="completed") -> dict:
    """仅从本运行有效 val 报告选最佳候选，不更新生产模型。

    :param service: 当前业务资源。
    :param run_id: 固定运行身份。
    :param reason: 停止条件或操作者停止说明。
    :param status: completed 或 stopped。
    :returns: 完整机器报告，人工可读报告同时存于 docs/reports。
    """
    run=service.get(run_id)
    if run.status in {"completed","stopped"} and run.report_path:
        path=owned_path(service.settings,run.report_path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=run.report_sha256:
            raise RuntimeError("已封存 Agent 报告指纹变化")
        return json.loads(path.read_text())
    items,candidates=[],[]
    with Session(service.engine) as session:
        iterations=list(session.scalars(select(AgentIteration).where(AgentIteration.run_id==run_id).order_by(AgentIteration.number)))
        for iteration in iterations:
            experiment=session.get(Experiment,iteration.experiment_id)
            evaluation=session.get(EvaluationTask,iteration.evaluation_id) if iteration.evaluation_id else None
            usable=bool(evaluation and evaluation.status=="completed" and not session.get(EvaluationInvalidation,evaluation.id) and evaluation.parameters==EvaluationCreate().model_dump() and evaluation.dataset_export_id==experiment.dataset_export_id)
            if usable:
                report_path=owned_path(service.settings,evaluation.report_path)
                usable=report_path.is_file() and hashlib.sha256(report_path.read_bytes()).hexdigest()==evaluation.report_sha256
            model=session.scalar(select(ModelArtifact).where(ModelArtifact.experiment_id==experiment.id,ModelArtifact.kind=="best").order_by(ModelArtifact.created_at.desc()).limit(1))
            if usable and model and model.id!=evaluation.model_id:
                usable=False
            if usable and model:
                path=owned_path(service.settings,model.path)
                if not path.is_file():
                    usable=False
                else:
                    with path.open("rb") as stream:
                        usable=hashlib.file_digest(stream,"sha256").hexdigest()==model.sha256
            value=evaluation.summary.get("map50_95") if usable else None
            if value is not None and math.isfinite(value) and model:
                candidates.append({"model_id":model.id,"experiment_id":experiment.id,"evaluation_id":evaluation.id,"dataset_export_id":experiment.dataset_export_id,"map50_95":value,"sha256":model.sha256})
            attempts=list(session.scalars(select(TrainingTask).where(TrainingTask.experiment_id==experiment.id).order_by(TrainingTask.created_at)))
            items.append({"iteration":iteration.number,"experiment_id":experiment.id,"dataset_export_id":experiment.dataset_export_id,"parent_experiment_id":experiment.parent_experiment_id,"parameters":experiment.parameters,"changed_variables":experiment.changed_variables,"plan":iteration.plan,"optuna_trial_number":iteration.optuna_trial_number,"status":experiment.status,"evaluation_id":evaluation.id if evaluation else None,"evaluation_valid":usable,"metrics":evaluation.summary if usable else None,"conclusion":iteration.conclusion,"training_attempts":[{"id":task.id,"parent_task_id":task.parent_task_id,"status":task.status,"device":task.device,"epochs":task.current_epoch,"elapsed_seconds":task.elapsed_seconds,"mlflow_run_id":task.mlflow_run_id,"error":task.error} for task in attempts]})
            parent=session.get(Experiment,experiment.parent_experiment_id) if experiment.parent_experiment_id else None
            items[-1]["comparison"]=comparison_design(parent,experiment.parameters,experiment.dataset_export_id,resolved=experiment.resolved_parameters)
            if usable:
                try:
                    items[-1]["confusion_evidence"]={"status":"completed","report_sha256":evaluation.report_sha256,**confusion_evidence(json.loads(report_path.read_text()))}
                except ValueError as error:
                    items[-1]["confusion_evidence"]={"status":"unavailable","reason":str(error)}
    for item in items:
        parent=next((other for other in items if other["experiment_id"]==item["parent_experiment_id"]),None)
        left=(parent["metrics"] or {}).get("map50_95") if parent and item["comparison"]["same_validation_data"] else None
        right=(item["metrics"] or {}).get("map50_95")
        item["comparison"]["delta_parent"]=right-left if left is not None and right is not None else None
    # 优先采用 Agent 指定的实验，但它必须是通过默认验证协议的有效候选；否则在最后所用数据集内按 mAP 选择。
    finish=run.state.get("agent_finish") or {}
    chosen=next((item for item in candidates if item["experiment_id"]==finish.get("best_experiment_id")),None)
    latest_export=items[-1]["dataset_export_id"] if items else None
    pool=[item for item in candidates if item["dataset_export_id"]==latest_export] or candidates
    best=chosen or (max(pool,key=lambda item:item["map50_95"]) if pool else None)
    result={"agent_run_id":run_id,"name":run.name,"status":status,"finished_at":utc_now(),"stop_reason":reason,"dataset_version_id":run.dataset_version_id,"dataset_export_id":run.dataset_export_id,"policy":run.policy,"budget":service.budget(run_id),"test_used":False,"iterations":items,"best_model":best,"production_published":False,"selection_unavailable_reason":None if best else "没有通过默认验证协议且工件有效的完成模型","selection":"agent" if chosen else "highest_val_map","agent_summary":finish.get("summary") or None,"agent_choice_rejected":bool(finish.get("best_experiment_id")) and not chosen,"limitations":["仅按固定 val 的真实 mAP50-95 选择候选，不保证下一轮提升","样本与类别支持量限制必须与指标一起阅读","Mac CPU 结果不计作 NVIDIA GPU 验收","生产模型发布仍需人工确认"]}
    result["last_error"]=run.error
    result["agent_summary_status"]="unverified_model_text"
    result["agent_value_assessment"]=finish.get("value_assessment")
    result["stagnation"]=run.state.get("stagnation")
    result["stop_notice"]=run.state.get("stop_notice")
    result["recommendations"]=(result["stop_notice"] or {}).get("recommendations",[])
    unfinished=[item for item in iterations if item.conclusion is None and item.optuna_trial_number is not None]
    if unfinished:
        from yolo_agent.agents.optimization import OptimizationStudy
        from yolo_agent.schemas.agent import AgentPolicy
        study=OptimizationStudy(service.settings,run_id,AgentPolicy.model_validate(run.policy))
        for item in unfinished:
            study.finish(item.optuna_trial_number,None,failed=bool(run.error))
    output=owned_path(service.settings,f"agents/{run_id}/report.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(".pending.json")
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(output)
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    document=Path("docs/reports")/f"agent-{run_id}.md"
    document.parent.mkdir(parents=True,exist_ok=True)
    lines=[f"# {run.name}","",f"运行：`{run_id}`；状态：{status}；用途：`{run.policy['purpose']}`。",f"停止原因：{reason}","",f"不可变数据：`{run.dataset_version_id}`；审核后工件：`{run.dataset_export_id}`。测试集未参与优化。","","| 轮次 | 实验 | mAP50-95 | 设备 | 结论 |","| --- | --- | --- | --- | --- |"]
    for item in items:
        value=item["metrics"].get("map50_95") if item["metrics"] else None
        delta=item["comparison"]["delta_parent"]
        conclusion="首轮基线" if not item["parent_experiment_id"] else (f"相对父实验 {delta:+.6f}" if delta is not None else "缺少可比父实验指标")
        devices=",".join(dict.fromkeys(task["device"] or "未知" for task in item["training_attempts"]))
        lines.append(f"| {item['iteration']+1} | `{item['experiment_id']}` | {value if value is not None else '无法评估'} | {devices} | {conclusion} |")
    lines.extend(["",f"最佳候选：`{best['model_id']}`，mAP50-95={best['map50_95']}（{'Agent 指定' if chosen else '按验证集 mAP 选择'}）。" if best else "没有可选择的有效候选。",*(["Agent 指定的实验没有有效验证结果，已改按验证集 mAP 选择。"] if result["agent_choice_rejected"] else []),*(["","## Agent 总结原文","","以下为未逐句验证的模型原文；方向、数字与归因以每轮结构化证据为准。",finish["summary"]] if finish.get("summary") else []),"","生产发布未执行，保留人工确认。","",f"完整机器报告：`storage/{output.relative_to(service.settings.storage_dir.resolve())}`；SHA256：`{digest}`。","","## 每轮决策与证据"])
    if result["agent_value_assessment"]:
        assessment=result["agent_value_assessment"]
        lines.extend(["", "### 结束本次训练的价值判断（模型原文）", "", "以下为 Agent 的判断，预期价值不代表实测收益。",
            f"已有证据：{assessment['evidence']}", f"继续训练的预期价值：{assessment['expected_benefit']}", f"成本权衡：{assessment['cost_tradeoff']}"])
    for item in items:
        lines.extend(["",f"### 第 {item['iteration']+1} 轮","",item["plan"]["reason"],"",f"变量变化：`{json.dumps(item['changed_variables'],ensure_ascii=False)}`",f"父实验：`{item['parent_experiment_id']}`；评估：`{item['evaluation_id']}`；Optuna：`{item['optuna_trial_number']}`。",f"数据建议：{'; '.join(item['plan'].get('data_actions',[])) or '无新增建议'}"])
        if item["plan"].get("value_assessment"):
            assessment=item["plan"]["value_assessment"]
            lines.extend(["", "继续训练的价值判断（模型原文，预期收益未验证）：", f"已有证据：{assessment['evidence']}",
                f"预期价值：{assessment['expected_benefit']}", f"成本权衡：{assessment['cost_tradeoff']}"])
        if len(item["changed_variables"])>1:
            lines.extend(["",f"可比性说明：本轮实际同时改变 {len(item['changed_variables'])} 个变量，不能将指标差异归因于单一变量；上述模型理由保留为规划假设原文。"])
        comparison=item["comparison"]
        lines.extend(["",f"相对父实验的 mAP50–95 变化：{comparison['delta_parent']}；{comparison['attribution']}。",
                      f"实验模式：{item['plan'].get('comparison_mode') or '旧记录未声明'}；定向变量：{item['plan'].get('target_parameter') or '无'}。",
                      f"待验证假设：{item['plan'].get('hypothesis') or '旧记录未声明'}",
                      f"实际生效参数差异：`{json.dumps(comparison['effective_changes'],ensure_ascii=False)}`",*comparison["coupled_effects"]])
        evidence=item.get("confusion_evidence",{})
        if evidence.get("status")=="completed":
            lines.extend(["","### 程序核算的矩阵证据","",evidence["reading_rule"],evidence["limitation"],
                          f"协议：`{json.dumps(evidence['protocol'],ensure_ascii=False)}`；评估：`{evidence['evaluation_id']}`。","",
                          "| 真值类别 | 真值数 | 正确数 | 预测为背景数 | 漏检比例 |","| --- | --- | --- | --- | --- |"])
            for row in evidence["per_true_class"]:
                ratio=f"{row['missed_ratio_of_true']:.4f}" if row["missed_ratio_of_true"] is not None else "无真值"
                lines.append(f"| {row['true_class']} | {row['true_count']} | {row['correct_count']} | {row['predicted_background_count']} | {ratio} |")
            lines.extend(["","| 真值 → 预测 | 数量 / 该类真值数 | 比例 |","| --- | --- | --- |"])
            for row in evidence["top_confusions"]:
                lines.append(f"| {row['true_class']} → {row['predicted_class']} | {row['count']} / {row['true_count']} | {row['ratio_of_true']:.4f} |")
    if result["recommendations"]:
        progress=result["stagnation"]
        lines.extend(["","## 主动停止与后续建议","",f"连续无明显提升：{progress['rounds_without_improvement']} / {progress['patience']} 轮；最小绝对提升：{progress['minimum_improvement']}。",f"最近显著提升锚点：{progress['significant_reference']}；实际最佳 mAP50-95：{progress['best_map50_95']}。","",*[f"- {text}" for text in result["recommendations"]]])
    lines.extend(["","## 局限","",*result["limitations"],""])
    document.write_text("\n".join(lines))
    service.patch(run_id,status=status,phase="final_report",finished_at=result["finished_at"],best_model_id=best["model_id"] if best else None,report_path=output.relative_to(service.settings.storage_dir.resolve()).as_posix(),report_sha256=digest,state={"stop_reason":reason,"budget":result["budget"],"human_report":str(document.resolve())})
    service.event(run_id,"final_report","sealed",{"status":status,"report_sha256":digest,"best_model":best,"stop_reason":reason})
    return result
