from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def accuracy(output: np.ndarray, targets: np.ndarray) -> Tuple[float]:
    acc = accuracy_score(y_true=targets, y_pred=np.where(output >= 0.5, 1, 0))
    return acc


def auc_score(output: np.ndarray, targets: np.ndarray) -> Tuple[float]:
    auc = roc_auc_score(y_true=targets, y_score=output)
    return auc
