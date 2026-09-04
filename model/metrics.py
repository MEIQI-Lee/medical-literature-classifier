"""不依赖 sklearn 的文本分类指标与混淆矩阵工具。"""

from __future__ import annotations

from typing import Iterable


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], num_labels: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < num_labels and 0 <= pred < num_labels:
            matrix[true][pred] += 1
    return matrix


def classification_metrics(y_true: list[int], y_pred: list[int], num_labels: int) -> dict[str, float]:
    """返回 accuracy、macro precision/recall/F1 和 micro F1。"""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
        }

    matrix = confusion_matrix(y_true, y_pred, num_labels)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in range(num_labels):
        tp = matrix[label][label]
        fp = sum(matrix[row][label] for row in range(num_labels)) - tp
        fn = sum(matrix[label]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    correct = sum(matrix[i][i] for i in range(num_labels))
    total_fp = sum(sum(matrix[row]) for row in range(num_labels)) - correct
    total_fn = total_fp
    micro_f1 = (2 * correct / (2 * correct + total_fp + total_fn)
                if correct or total_fp or total_fn else 0.0)
    return {
        "accuracy": correct / len(y_true),
        "precision_macro": sum(precisions) / num_labels,
        "recall_macro": sum(recalls) / num_labels,
        "f1_macro": sum(f1s) / num_labels,
        "f1_micro": micro_f1,
    }
