"""Shared classification metric helpers."""

import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_multiclass_auc(labels, scores):
    """
    Args:
        labels: One-dimensional true class label array.
        scores: Two-dimensional class probability array, one column per class.

    Return:
        Macro OvR AUC value. Returns None when AUC is not defined.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if labels.size == 0 or scores.size == 0:
        return None
    if labels.ndim != 1 or scores.ndim != 2 or labels.shape[0] != scores.shape[0]:
        return None

    num_classes = scores.shape[1]
    if num_classes < 2:
        return None

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return None

    try:
        if num_classes == 2:
            return float(roc_auc_score(labels, scores[:, 1]))

        # 1. Multi-class macro OvR AUC needs every class to appear in the epoch.
        # 2. If a class is missing, returning None avoids reporting an incomplete metric.
        if unique_labels.size != num_classes or not np.array_equal(unique_labels, np.arange(num_classes)):
            return None
        return float(roc_auc_score(labels, scores, multi_class="ovr", average="macro"))
    except ValueError:
        return None
