"""对父子实验执行受控对比，不把配置变化数冒充因果证明。"""

from yolo_agent.schemas.experiment import TrainingParameters


def comparison_design(parent, parameters: dict, export_id: str, *, resolved: dict | None = None) -> dict:
    """比较完整校验参数，同时披露训练器实际生效配置的联动。"""
    if parent is None:
        return {"kind": "baseline", "parent_experiment_id": None, "changed_variables": {},
                "variable_count": 0, "same_validation_data": False,
                "attribution": "首轮基线，没有对比结论", "coupled_effects": [], "effective_changes": {}}
    before = TrainingParameters.model_validate(parent.parameters).model_dump()
    after = TrainingParameters.model_validate(parameters).model_dump()
    changed = {key: {"before": before[key], "after": value} for key, value in after.items() if before[key] != value}
    same_data = parent.dataset_export_id == export_id
    if not same_data:
        changed["dataset_export_id"] = {"before": parent.dataset_export_id, "after": export_id}
    ignored = [key for key in ("lr0", "warmup_bias_lr") if key in changed and before["optimizer"] == after["optimizer"] == "auto"]
    notes = []
    if before["optimizer"] == "auto" or after["optimizer"] == "auto":
        notes.append("auto 会联动优化器、有效学习率和 warmup_bias_lr；只改一个请求字段仍不保证只改一个实际因素。学习率实验请先建立显式优化器基线。")
    if set(changed) & {"batch", "nbs", "epochs", "optimizer", "rect", "model", "imgsz"}:
        notes.append("本次变量可能联动梯度累计、权重衰减、调度、增强或批量；必须核对实际解析配置，不能归因于其中的单一机制。")
    previous = parent.resolved_parameters or {}
    effective = {}
    if resolved:
        # 不比较路径、时间或动态末轮 lr；仅比较已由官方训练器记录的起始配置。
        for section in ("trainer_parameters", "augmentation_effective_at_start"):
            left, right = previous.get(section), resolved.get(section)
            if left is not None and right is not None:
                for key in left.keys() | right.keys():
                    if left.get(key) != right.get(key):
                        effective[f"{section}.{key}"] = {"before": left.get(key), "after": right.get(key)}
        for key in ("class", "groups"):
            def value(data):
                optimizer = data.get("optimizer_effective", {})
                return ([{name: group.get(name) for name in ("initial_lr", "weight_decay")} for group in optimizer.get("groups", [])]
                        if key == "groups" else optimizer.get(key))
            left, right = value(previous), value(resolved)
            if left != right:
                effective[f"optimizer_effective.{key}"] = {"before": left, "after": right}
    count = len(changed)
    kind = "data_change" if not same_data else "single_variable" if count == 1 else "joint_change" if count > 1 else "repeat"
    return {"kind": kind, "parent_experiment_id": parent.id, "changed_variables": changed,
            "variable_count": count, "ignored_parameter_changes": ignored, "same_validation_data": same_data,
            "coupled_effects": notes, "effective_changes": effective,
            "attribution": ("数据工件不同，验证指标不能直接比较" if not same_data else
                            "多变量同时变化，只能评价整组调整" if count > 1 else
                            "仅一个请求变量变化；单次固定验证集对比不能证明因果，需结合实际生效参数与重复实验")}


def validate_experiment_design(design: dict, *, mode: str, target: str | None, hypothesis: str | None) -> None:
    """联合搜索允许多变量；只有声称进行定向消融时才强制单变量。"""
    if design["kind"] == "baseline":
        return
    changes = design["changed_variables"]
    if not changes:
        raise ValueError("与父实验的参数和数据相同，请提出有依据的调整或使用 resume_training")
    if set(changes) <= set(design["ignored_parameter_changes"]):
        raise ValueError(f"auto 优化器会覆盖 {design['ignored_parameter_changes']}，本次调整没有预期效果；先单独建立显式优化器基线。")
    if not hypothesis or len(hypothesis.strip()) < 10:
        raise ValueError("请用 hypothesis 写出基于父实验证据的配置调整假设（至少10字），不能把假设写成已证明的单项贡献")
    if mode == "ablation":
        if len(changes) != 1 or not design["same_validation_data"]:
            raise ValueError(f"定向消融要求同一数据工件且只改一个变量；实际变化 {len(changes)} 项：{list(changes)}。可改用 joint_search 比较整组配置，不归因单项贡献。")
        if target not in changes:
            raise ValueError(f"定向消融必须声明 target_parameter={next(iter(changes))}")
    elif mode != "joint_search":
        raise ValueError("未知实验设计模式")
