import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

import sys

sys.path.append(r"/opt/ml/level2_dkt-recsys-04/code/")  # 절대경로 추가 (추후에 수정해주어야 함)

from lgbm.lgbm_util.utils import get_logger, Setting, logging_conf
from lgbm.lgbm_util.datasets import (
    feature_engineering,
    custom_train_test_split,
)
from lgbm.lgbm_util.args import train_parse_args

from tqdm import tqdm
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.utils import Bunch
import argparse

import warnings

warnings.filterwarnings("ignore")

setting = Setting()
logger = get_logger(logging_conf)


def custom_label_split(df: pd.DataFrame) -> Tuple[list, pd.DataFrame]:
    y = df["answerCode"]
    X = df.drop(["answerCode"], axis=1)
    return y, X


class LGBM:
    def __init__(
        self,
        args: argparse.Namespace,
        model_path: str = None,
    ) -> None:
        self.model = self.load_model(args=args, model_path=model_path)

    def train(
        self,
        lgb_train: lgb.Dataset,
        lgb_valid: lgb.Dataset,
    ) -> None:
        self.model = lgb.train(
            self.param,
            lgb_train,
            valid_sets=(lgb_train, lgb_valid),
            callbacks=[lgb.early_stopping(100, verbose=0)],
            verbose_eval=False,
        )

    def load_model(
        self,
        args: argparse.Namespace,
        model_path: str,
    ) -> lgb.Booster:
        if model_path:
            self.model = lgb.Booster(model_file=model_path)
        else:
            self.param = {
                "boosting_type": "gbdt",
                "objective": "binary",
                "metric": ["auc", "binary_logloss"],
                "num_leaves": args.num_leaves,
                "learning_rate": args.lr,
                "num_boost_round": args.num_boost_round,
                "verbose": -1,
            }
            self.model = None
        return self.model

    def score(self, y: pd.Series, pred: np.ndarray) -> Tuple[float, float]:
        auc = roc_auc_score(y, pred)
        acc = accuracy_score(y, np.where(pred >= 0.5, 1, 0))
        return (auc, acc)

    def predict(self, data: pd.Series) -> np.ndarray:
        return self.model.predict(data)


def main(args: argparse.Namespace):
    ########################   Set Random Seed
    print("--------------- Set Random Seed ---------------")
    setting.set_seeds(args.seed)

    ######################## DATA LOAD
    print("--------------- Load Data ---------------")
    data_dir = args.data_dir
    csv_file_path = os.path.join(data_dir, "train_data.csv")
    dataframe = pd.read_csv(csv_file_path)

    ######################## Feature Engineering
    dataframe = feature_engineering(dataframe)

    ########################   Data Split
    print("--------------- Data Split   ---------------")
    # 유저별 분리
    train, valid = custom_train_test_split(dataframe)

    ########################   Data Loader
    print("--------------- Data Loader   ---------------")

    # X, y 값 분리
    y_train, train = custom_label_split(train)
    y_valid, valid = custom_label_split(valid)

    ########################   Feature Selecting (Forward selection)
    print("--------------- Select features   ---------------")
    features = train.columns.tolist()
    feature_list = []
    feature_selection_result = pd.DataFrame(
        {
            "features": ["0"] * len(features),
            "len_features": ["0"] * len(features),
            "valid_auc": np.zeros(len(features)),
            "valid_acc": np.zeros(len(features)),
        }
    )
    temp_features = features.copy()

    for i in tqdm(range(len(features))):
        train_temp = train[temp_features]
        valid_temp = valid[temp_features]
        cat_features = list(np.where(train_temp.dtypes == np.object_)[0])

        lgb_train = lgb.Dataset(train_temp, y_train, categorical_feature=cat_features)
        lgb_valid = lgb.Dataset(
            valid_temp, y_valid, reference=lgb_train, categorical_feature=cat_features
        )

        model = LGBM(args)
        model.train(lgb_train, lgb_valid)

        if args.importance_type == "permutation":
            importances = permutation_importance(
                model,
                train_temp,
                y_train,
                scoring=make_scorer(roc_auc_score, greater_is_better=True),
                n_repeats=args.n_repeats,
                random_state=args.seed,
            )
        elif args.importance_type == "model":
            importances = model.model.feature_importance()
            importances = Bunch(**{"importances_mean": importances})

        sort_index = importances.importances_mean.argsort()
        temp_result = pd.DataFrame(
            importances.importances_mean[sort_index],
            index=train_temp.columns[sort_index],
        ).sort_values(0, ascending=False)
        temp_result = temp_result.rename(columns={0: "importances"})

        feature = list(temp_result.index)[0]
        temp_features.remove(feature)
        feature_list.append(feature)

        train_temp_ = train[feature_list]
        valid_temp_ = valid[feature_list]
        cat_features_temp = list(np.where(train_temp_.dtypes == np.object_)[0])

        lgb_train_temp = lgb.Dataset(
            train_temp_, y_train, categorical_feature=cat_features_temp
        )
        lgb_valid_temp = lgb.Dataset(
            valid_temp_,
            y_valid,
            reference=lgb_train_temp,
            categorical_feature=cat_features_temp,
        )

        model_temp = LGBM(args)
        model_temp.train(lgb_train_temp, lgb_valid_temp)

        pred = model_temp.predict(valid_temp_)
        auc, acc = model_temp.score(y_valid, pred)

        feature_selection_result.loc[i, "len_features"] = len(feature_list)
        feature_selection_result.loc[i, "features"] = str(feature_list)
        feature_selection_result.loc[i, "valid_auc"] = auc
        feature_selection_result.loc[i, "valid_acc"] = acc

        feature_selection_result = feature_selection_result.sort_values(
            "valid_auc", ascending=False
        ).reset_index(drop=True)
        feature_selection_result.to_csv("feature_selection_result.csv", index=False)

    print("--------------- Saved Result ---------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument(
        "--use_cuda_if_available", default=True, type=bool, help="Use GPU"
    )

    parser.add_argument("--data_dir", default="/opt/ml/input/data", type=str, help="")
    parser.add_argument("--model_dir", default="./models/", type=str, help="model dir")
    parser.add_argument(
        "--output_dir", default="./outputs/", type=str, help="output dir"
    )
    parser.add_argument(
        "--num_boost_round", default=500, type=int, help="num_boost_round"
    )
    parser.add_argument("--num_leaves", default=31, type=int, help="num_leaves")

    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")

    parser.add_argument(
        "--n_repeats", default=10, type=int, help="number of permutation"
    )
    parser.add_argument(
        "--importance_type",
        choices=["permutation", "model"],
        type=str,
        help="type of importance",
    )

    args = parser.parse_args()

    main(args=args)
