import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


class CatBoost:
    def __init__(
        self,
        args: argparse.Namespace,
        cat_features: list,
        model_path: str = None,
    ) -> None:
        self.model = self.load_model(args=args, model_path=model_path)
        self.cat_features = cat_features

    def train(
        self,
        train: pd.DataFrame,
        y_train: pd.Series,
        valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
    ) -> None:
        """_summary_
        CatBoost Train 함수
        Args:
            train (pd.DataFrame): train dataset
            y_train (pd.Series): train label
            valid (pd.DataFrame): valid dataset
            y_valid (pd.Series): valid label
        """
        eval_set = None
        if valid is not None and y_valid is not None:
            eval_set = (valid, y_valid)

        self.model.fit(
            train,
            y_train,
            cat_features=self.cat_features,
            eval_set=eval_set,
            use_best_model=True,
            early_stopping_rounds=100,
            verbose_eval=100,
        )

    def load_model(
        self,
        args: argparse.Namespace,
        model_path: str,
    ) -> CatBoostClassifier:
        """_summary_
        Train시에는 Config에 맞는 초기 모델을 반환하며
        Inference시에는 모델의 가중치(cbm) 파일 경로를 읽어 기존 모델을 Load하여 반환한다.

        Args:
            args (argparse.Namespace): Config
            model_path (str): Model을 Load하여 사용할 때 사용하는 변수로 기존 모델의 경로를 나타낸다.

        Returns:
            CatBoostClassifier: Model을 반환한다.
        """
        if model_path:
            self.model = CatBoostClassifier()
            self.model.load_model(model_path, format="cbm")
        else:
            if args.use_cuda_if_available:
                self.model = CatBoostClassifier(
                    iterations=args.num_iterations,
                    learning_rate=args.lr,
                    depth=args.depth,
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
                    depth=args.depth,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=args.seed,
                )
        return self.model

    def save_model(
        self,
        filename: str,
    ) -> bool:
        """_summary_
        CatBoost 모델 cbm(catboost model binary 형태) 저장
        Args:
            filename (str): File 경로

        Returns:
            bool: Save 성공 여부 반영
        """
        try:
            self.model.save_model(fname=filename, format="cbm")
        except Exception:
            return False
        return True

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
