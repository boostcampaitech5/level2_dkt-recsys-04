import warnings
import os
import argparse
from pprint import pprint

import wandb
import numpy as np
import pandas as pd
import lightgbm as lgb

from boosting_util.models import Model
from boosting_util.utils import get_logger, Setting, logging_conf
from boosting_util.datasets import (
    preprocessing,
    feature_engineering,
    custom_train_test_split,
    custom_label_split,
)
from boosting_util.args import train_parse_args

# ignore warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    setting = Setting(args)

    ########################   Set Random Seed
    print(f"--------------- {args.model} Set Random Seed ---------------")
    setting.set_seeds(args.seed)

    ########################   Set WandB
    print(f"--------------- {args.model} Set WandB ---------------")
    # Wandb
    # wandb.login()
    # wandb.init(project="dkt", config=vars(args))

    ######################## DATA LOAD
    print(f"--------------- {args.model} Load Data ---------------")
    data_dir = args.data_dir
    csv_file_path = os.path.join(data_dir, "train_data.csv")
    dataframe = pd.read_csv(csv_file_path)

    # 테스트 세트 예측
    test_dataframe = pd.read_csv(os.path.join(data_dir, "test_data.csv"))

    ######################## Feature Engineering
    dataframe = feature_engineering(args.feats, dataframe)

    # Category Feature 선택
    cat_features, dataframe = preprocessing(dataframe)
    print(
        f"After Train/Valid DataSet Feature Engineering Columns : {dataframe.columns.values.tolist()}"
    )
    test_dataframe = feature_engineering(args.feats, test_dataframe)
    print(
        f"After Test DataSet Feature Engineering Columns : {test_dataframe.columns.values.tolist()}"
    )
    test_dataframe = test_dataframe[
        test_dataframe["userID"] != test_dataframe["userID"].shift(-1)
    ]
    # Category Feature 선택
    cat_features, test_dataframe = preprocessing(test_dataframe)

    ########################   Data Split
    print(f"--------------- {args.model} Data Split   ---------------")
    # 유저별 분리
    train, valid = custom_train_test_split(dataframe)

    train.to_csv(
        "/opt/ml/level2_dkt-recsys-04/code/boosting/train_7.csv", sep=","
    )
    valid.to_csv(
        "/opt/ml/level2_dkt-recsys-04/code/boosting/valid_3.csv", sep=","
    )

    ########################   Data Loader
    print(f"--------------- {args.model} Data Loader   ---------------")

    # X, y 값 분리
    y_train, train = custom_label_split(train)
    y_valid, valid = custom_label_split(valid)

    if args.model == "LGBM":
        lgb_train = lgb.Dataset(
            train,
            y_train,
            categorical_feature=cat_features,
        )
        lgb_valid = lgb.Dataset(
            valid,
            y_valid,
            reference=lgb_train,
            categorical_feature=cat_features,
        )

    ########################   TRAIN
    print(f"--------------- {args.model} Train   ---------------")
    print(
        f"Train Feature Engineering Columns : {train.columns.values.tolist()}"
    )

    print(f"Categoy : {cat_features}")
    model = Model(args.model, args, cat_features=cat_features).load_model()

    if args.model == "CatBoost":
        model.train(train, y_train, valid, y_valid)
    elif args.model == "LGBM":
        model.train(lgb_train, lgb_valid)

    ########################   VALID
    print(f"--------------- {args.model} Valid   ---------------")
    valid_preds = model.pred(valid)
    auc, acc = model.score(y_valid, valid_preds)
    print(f"VALID AUC : {auc} ACC : {acc}\n")
    print(f"BEST VALIDATION : {model.best_validation_score}\n")
    print("Feature Importance : ")
    feature_importance = sorted(
        dict(
            zip(
                train.columns.values.tolist(),
                model.feature_importance,
            )
        ).items(),
        key=lambda item: item[1],
        reverse=True,
    )
    pprint(
        feature_importance,
        width=50,
    )

    ########################   INFERENCE
    print(f"--------------- {args.model} Predict   ---------------")
    total_preds = model.pred(test_dataframe.drop(["answerCode"], axis=1))

    ######################## SAVE PREDICT
    print("\n--------------- Save Output Predict   ---------------")
    filename = setting.get_submit_filename(
        output_dir=args.output_dir,
        auc_score=model.best_validation_score,
    )
    setting.save_predict(filename=filename, predict=total_preds)

    ######################## SAVE MODEL
    print("\n--------------- Save Model   ---------------")
    format_name = "cbm" if args.model == "CatBoost" else "txt"
    model_path = setting.get_submit_filename(
        output_dir=args.model_dir,
        auc_score=model.best_validation_score,
        format_name=format_name,
    )
    print(f"saving model : {model_path}")
    model.save_model(filename=model_path)

    ######################## SAVE CONFIG
    print("\n--------------- Save Config   ---------------")
    setting.save_config(args, model.best_validation_score)


if __name__ == "__main__":
    args = train_parse_args()
    main(args=args)
