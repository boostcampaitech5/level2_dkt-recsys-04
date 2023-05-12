import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


class CatBoost:
    def __init__(self, args: argparse.Namespace, cat_features: list) -> None:
        if args.use_cuda_if_available:
            self.model = CatBoostClassifier(
                iterations=args.num_iterations,
                learning_rate=args.lr,
                depth=6,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=args.seed,
                task_type="GPU",
                devices="0",
            )
        else:
            self.model = CatBoostClassifier(
                iterations=args.num_iterations,
                learning_rate=args.lr,
                depth=6,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=args.seed,
            )
        self.cat_features = cat_features

    def train(
        self,
        train: pd.DataFrame,
        y_train: pd.Series,
        valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> None:
        """_summary_
        CatBoost Train
        Args:
            train (pd.DataFrame): train dataset
            y_train (pd.Series): train label
            valid (pd.DataFrame): valid dataset
            y_valid (pd.Series): valid label
        """
        self.model.fit(
            train,
            y_train,
            cat_features=self.cat_features,
            eval_set=(valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose_eval=100,
        )

    def score(self, y: pd.Series, pred: np.ndarray) -> Tuple[float, float]:
        """_summary_
        AUC, ACC Score
        Args:
            y (pd.Series): 정답 Label
            pred (np.ndarray): Inference 결과

        Returns:
            Tuple[float, float]: AUC, ACC Score
        """
        auc = roc_auc_score(y, pred)
        acc = accuracy_score(y, np.where(pred >= 0.5, 1, 0))
        return (auc, acc)

    def pred(self, data: pd.Series) -> np.ndarray:
        """_summary_
        Inference
        Args:
            data (pd.Series): Inference Data

        Returns:
            np.ndarray: Model Predict
        """
        return self.model.predict_proba(data)[:, 1]
