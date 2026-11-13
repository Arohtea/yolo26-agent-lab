"""将验证器原始计数转换为带方向、分母和协议的 Agent 事实，不重新匹配目标。"""

import numpy as np


def confusion_evidence(report: dict) -> dict:
    """复用 NumPy 汇总已验证报告；拒绝未知轴，避免静默猜测矩阵方向。"""
    source = report["confusion_matrix"]
    if source.get("rows") != "predicted" or source.get("columns") != "true":
        raise ValueError("混淆矩阵轴协议未知，禁止根据图像猜测方向")
    labels = source["labels"]
    matrix = np.asarray(source["matrix"], dtype=float)
    if (not labels or labels[-1] != "background" or len(set(labels)) != len(labels)
            or matrix.shape != (len(labels), len(labels)) or not np.isfinite(matrix).all()
            or (matrix < 0).any() or not np.equal(matrix, np.floor(matrix)).all()):
        raise ValueError("混淆矩阵标签、维度或计数不合法")
    matrix = matrix.astype(np.int64)
    supports = matrix.sum(axis=0)
    background = len(labels) - 1
    classes, confusions = [], []
    for true_index, name in enumerate(labels[:-1]):
        support = int(supports[true_index])
        missed = int(matrix[background, true_index])
        correct = int(matrix[true_index, true_index])
        entries = []
        for predicted_index in range(background):
            count = int(matrix[predicted_index, true_index])
            if predicted_index != true_index and count:
                entries.append({"true_class": name, "predicted_class": labels[predicted_index],
                                "count": count, "true_count": support, "ratio_of_true": count / support})
        entries.sort(key=lambda row: row["count"], reverse=True)
        confusions.extend(entries)
        classes.append({"true_class": name, "true_count": support, "correct_count": correct,
                        "correct_ratio_of_true": correct / support if support else None,
                        "predicted_background_count": missed,
                        "missed_ratio_of_true": missed / support if support else None,
                        "most_confused_as": entries[:3],
                        "unavailable_reason": None if support else "没有此类真值，比例不可评价"})
    confusions.sort(key=lambda row: row["count"], reverse=True)
    return {
        "evaluation_id": report["evaluation_id"], "split": report["split"],
        "axes": {"rows": "predicted", "columns": "true", "normalization": "每列除以该真值列总数"},
        "protocol": {key: report.get("metric_protocol", {}).get(key) for key in ("confusion_confidence", "confusion_iou")},
        "reading_rule": "每条方向均为 true_class → predicted_class；漏检读 predicted=background 的行。true=background 的列是无真值匹配的预测，不能读成漏检。",
        "per_true_class": classes, "top_confusions": confusions[:10],
        "unmatched_predictions_by_predicted_class": [
            {"predicted_class": name, "count": int(matrix[index, background])}
            for index, name in enumerate(labels[:-1])],
        "limitation": "这些计数来自 Ultralytics 混淆矩阵协议，不等于 FiftyOne 固定阈值错误清单，也不等于 AP 最佳 F1 点的 P/R；数值分析以本结构为准，图片用于观察场景。",
    }
