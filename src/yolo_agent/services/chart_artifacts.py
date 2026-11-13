"""工作台与 Agent 共用训练库产物白名单，不接受任意文件路径。"""

CHARTS = {
    "results": "results.png", "labels": "labels.jpg", "train_batch": "train_batch0.jpg",
    "confusion_matrix": "confusion_matrix.png", "confusion_matrix_normalized": "confusion_matrix_normalized.png",
    "pr_curve": "BoxPR_curve.png", "f1_curve": "BoxF1_curve.png", "precision_curve": "BoxP_curve.png",
    "recall_curve": "BoxR_curve.png", "val_predictions": "val_batch0_pred.jpg", "val_labels": "val_batch0_labels.jpg",
}
