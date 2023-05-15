import os
import argparse
from pprint import pprint

import wandb
import numpy as np
import pandas as pd

from catboost_util.models import CatBoost
from catboost_util.utils import get_logger, Setting, logging_conf
from catboost_util.datasets import (
    feature_engineering,
    custom_train_test_split,
    custom_label_split,
    FEATS,
)
from catboost_util.args import train_parse_args

setting = Setting()
logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    ########################   Set Random Seed
    print("--------------- CatBoost Set Random Seed ---------------")
    setting.set_seeds(args.seed)

    ########################   Set WandB
    print("--------------- CatBoost Set WandB ---------------")
    # Wandb
    # wandb.login()
    # wandb.init(project="dkt", config=vars(args))

    ######################## DATA LOAD
    print("--------------- CatBoost Load Data ---------------")
    data_dir = args.data_dir
    csv_file_path = os.path.join(data_dir, "train_data.csv")
    dataframe = pd.read_csv(csv_file_path)

    # 테스트 세트 예측
    test_dataframe = pd.read_csv(os.path.join(data_dir, "test_data.csv"))

    ######################## Feature Engineering
    dataframe = feature_engineering(dataframe)
    test_dataframe = feature_engineering(test_dataframe)
    test_dataframe = test_dataframe[
        test_dataframe["userID"] != test_dataframe["userID"].shift(-1)
    ]

    ########################   Data Split
    print("--------------- CatBoost Data Split   ---------------")
    # 유저별 분리
    train, valid = custom_train_test_split(dataframe)

    ########################   Data Loader
    print("--------------- CatBoost Data Loader   ---------------")

    # X, y 값 분리
    y_train, train = custom_label_split(train)
    y_valid, valid = custom_label_split(valid)

    ########################   TRAIN
    print("--------------- CatBoost Train   ---------------")

    cat_features = list(np.where(train.dtypes == np.object_)[0])
    model = CatBoost(args, cat_features)
    model.train(train, y_train, valid, y_valid)

    ########################   VALID
    print("--------------- CatBoost Valid   ---------------")
    valid_preds = model.pred(valid)
    auc, acc = model.score(y_valid, valid_preds)
    print(f"VALID AUC : {auc} ACC : {acc}\n")

    print(f"BEST VALIDATION : {model.model.best_score_['validation']}\n")
    print("Feature Importance : ")
    feature_importance = sorted(
        dict(zip(FEATS, model.model.feature_importances_)).items(),
        key=lambda item: item[1],
        reverse=True,
    )
    pprint(
        feature_importance,
        width=50,
    )

    ########################   INFERENCE
    print("--------------- CatBoost Predict   ---------------")
    total_preds = model.pred(test_dataframe[FEATS])

    ######################## SAVE PREDICT
    print("\n--------------- Save Output Predict   ---------------")
    filename = setting.get_submit_filename(
        args.output_dir,
        model.model.best_score_["validation"]["AUC"],
    )
    setting.save_predict(filename=filename, predict=total_preds)

    ######################## SAVE MODEL
    print("\n--------------- Save Model   ---------------")
    model_path = setting.get_submit_filename(
        output_dir=args.model_dir,
        auc_score=model.model.best_score_["validation"]["AUC"],
        format_name="cbm",
    )
    print(f"saving model : {model_path}")
    model.save_model(filename=model_path)

    ######################## SAVE CONFIG
    print("\n--------------- Save Config   ---------------")
    setting.save_config(args, model.model.best_score_["validation"]["AUC"])


if __name__ == "__main__":
    args = train_parse_args()
    main(args=args)
