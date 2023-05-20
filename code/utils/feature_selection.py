import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

import sys

sys.path.append(r"/opt/ml/level2_dkt-recsys-04/code/")  # 절대경로 추가 (추후에 수정해주어야 함)
sys.path.append(
    r"/opt/ml/level2_dkt-recsys-04/code/catboost/"
)  # 절대경로 추가 (추후에 수정해주어야 함)

from lgbm.lgbm_util.utils import get_logger, Setting, logging_conf
from lgbm.lgbm_util.datasets import custom_train_test_split
from catboost_util.datasets import feature_engineering

from lgbm.lgbm_util.args import train_parse_args

from tqdm import tqdm
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch

import time
from datetime import datetime

import argparse

import warnings

warnings.filterwarnings("ignore")

setting = Setting()
logger = get_logger(logging_conf)

FEATS = [
    "calculate_cumulative_stats_by_time",
    "calculate_overall_accuracy_by_testID",
    "calculate_overall_accuracy_by_KnowledgeTag",
    # # 시간 칼럼을 사용하는 FE
    "calculate_solve_time_column",  # Time 관련 Feature Engineering할 때 필수!
    "check_answer_at_time",
    "calculate_total_time_per_user",
    "calculate_past_correct_answers_per_user",
    "calculate_future_correct_answers_per_user",
    "calculate_past_correct_attempts_per_user",
    "calculate_past_solved_problems_per_user",
    "calculate_past_average_accuracy_per_user",
    "calculate_past_average_accuracy_current_problem_per_user",
    "calculate_rolling_mean_time_last_3_problems_per_user",
    # "calculate_mean_and_stddev_per_user", # 오류가 많아서 스킵
    "calculate_median_time_per_user",
    "calculate_problem_solving_time_per_user",
    "calculate_accuracy_by_time_of_day",
    "calculate_user_activity_time_preference",
    "calculate_normalized_time_per_user",
    "calculate_relative_time_spent_per_user",
    "calculate_time_cut_column",
    # "calculate_items_svd_latent",
    # "calculate_times_nmf_latent",
    # "calculate_users_pca_latent",
    # "calculate_items_pca_latent",
    # "calculate_times_pca_latent",
    # "calculate_times_lda_latent",
    # "caculate_item_latent_dirichlet_allocation",  # 50초 걸림
    # "caculate_user_latent_dirichlet_allocation",  # 50초 걸림
]


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


class FeatureSelector:
    def __init__(self, args: argparse.Namespace):
        ########################   Set Random Seed
        print("--------------- Set Random Seed ---------------")
        setting.set_seeds(args.seed)

        ######################## DATA LOAD
        print("--------------- Load Data ---------------")
        data_dir = args.data_dir
        csv_file_path = os.path.join(data_dir, "train_data.csv")
        dataframe = pd.read_csv(csv_file_path)

        ######################## Feature Engineering
        dataframe = feature_engineering(args.feats, dataframe)

        ######################## ValueError 해결을 위한 전처리
        print("--------------- Data Preprocessing ---------------")
        cate_cols = ["assessmentItemID", "testId"]
        dataframe = self.preprocessing(dataframe, cate_cols)

        ########################   Data Split
        print("--------------- Data Split   ---------------")
        # 유저별 분리
        train, valid = custom_train_test_split(dataframe)

        ########################   Data Loader
        print("--------------- Data Loader   ---------------")

        # X, y 값 분리
        self.y_train, self.train = custom_label_split(train)
        self.y_valid, self.valid = custom_label_split(valid)

        ########################   Feature Selecting (Forward selection)
        print("--------------- Select Features   ---------------")
        self.features = self.train.columns.tolist()
        self.feature_list = []
        self.feature_selection_result = pd.DataFrame(
            {
                "features": ["0"] * len(self.features),
                "len_features": ["0"] * len(self.features),
                "valid_auc": np.zeros(len(self.features)),
                "valid_acc": np.zeros(len(self.features)),
            }
        )
        self.temp_features = self.features.copy()
        self.args = args

        if self.args.importance_type == "permutation":
            self.model = self.model_train(
                self.train, self.valid, self.y_train, self.y_valid, self.features
            )

    def select_features(self):
        if self.args.importance_type == "permutation":
            print("--------------- Start Permutation   ---------------")
            importances = self.perm_importance(self.model, self.train, self.y_train)
            sort_index = list(importances.importances_mean.argsort())
            print("--------------- Permutation Done!   ---------------")

        for i in tqdm(range(len(self.features))):
            if self.args.importance_type == "model":
                self.model = self.model_train(
                    self.train,
                    self.valid,
                    self.y_train,
                    self.y_valid,
                    self.temp_features,
                )
                importances = self.model_importance(self.model)

            if self.args.importance_type == "permutation":
                if i != 0:
                    sort_index.remove(
                        max(sort_index)
                    )  # 반복문이 돌때마다 permutation importances index 제거
            else:
                sort_index = importances.importances_mean.argsort()

            temp_result = pd.DataFrame(
                importances.importances_mean[sort_index],
                index=self.train[self.temp_features].columns[sort_index],
            ).sort_values(0, ascending=False)
            temp_result = temp_result.rename(columns={0: "importances"})

            feature = list(temp_result.index)[0]
            self.temp_features.remove(feature)
            self.feature_list.append(feature)

            model_temp = self.model_train(
                self.train, self.valid, self.y_train, self.y_valid, self.feature_list
            )

            pred = model_temp.predict(self.valid[self.feature_list])
            auc, acc = model_temp.score(self.y_valid, pred)

            self.feature_selection_result.loc[i, "len_features"] = len(
                self.feature_list
            )
            self.feature_selection_result.loc[i, "features"] = str(self.feature_list)
            self.feature_selection_result.loc[i, "valid_auc"] = auc
            self.feature_selection_result.loc[i, "valid_acc"] = acc

            self.feature_selection_result = self.feature_selection_result.sort_values(
                "valid_auc", ascending=False
            ).reset_index(drop=True)
            self.feature_selection_result.to_csv(
                "feature_selection_result.csv", index=False
            )

        print("--------------- Saved Result ---------------")

    def preprocessing(self, dataframe, cate_cols):
        # LabelEncoding
        for col in cate_cols:
            le = LabelEncoder()
            # For UNKNOWN class
            a = dataframe[col].unique().tolist() + ["unknown"]
            le.fit(a)

            # cate_cols 는 범주형이라고 가정
            dataframe[col] = dataframe[col].astype(str)
            encoded_values = le.transform(dataframe[col])
            dataframe[col] = encoded_values

        def convert_time(s: str):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        dataframe["Timestamp"] = dataframe["Timestamp"].map(lambda x: str(x))
        dataframe["Timestamp"] = dataframe["Timestamp"].apply(convert_time)

        try:
            dataframe["time_cut_enc"] = dataframe["time_cut_enc"].astype(int)
        except:
            print("'time_cut_enc' column not in dataframe.")
            pass

        return dataframe

    def perm_importance(self, model, train_x, train_y):
        importances = permutation_importance(
            model,
            train_x,
            train_y,
            scoring=make_scorer(roc_auc_score, greater_is_better=True),
            n_repeats=self.args.n_repeats,
            random_state=self.args.seed,
        )

        return importances

    def model_importance(self, model):
        importances = model.model.feature_importance()
        importances = Bunch(**{"importances_mean": importances})

        return importances

    def shap_importance(self):
        print("this has not been implemented yet.")

    def model_train(self, train, valid, y_train, y_valid, features_):
        train_temp = train[features_]
        valid_temp = valid[features_]
        cat_features = list(np.where(train_temp.dtypes == np.object_)[0])

        lgb_train = lgb.Dataset(train_temp, y_train, categorical_feature=cat_features)
        lgb_valid = lgb.Dataset(
            valid_temp, y_valid, reference=lgb_train, categorical_feature=cat_features
        )

        model = LGBM(self.args)
        model.train(lgb_train, lgb_valid)

        return model


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
    parser.add_argument(
        "--feats",
        default=FEATS,
        type=list,
        help="feats",
    )

    args = parser.parse_args()

    selector = FeatureSelector(args)
    selector.select_features()
