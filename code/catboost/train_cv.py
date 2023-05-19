import os
import argparse
from pprint import pprint

import wandb
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from catboost_util.models import CatBoost
from catboost_util.utils import get_logger, Setting, logging_conf
from catboost_util.datasets import (
    feature_engineering,
    custom_train_test_split,
    custom_label_split,
    BlockingTimeSeriesSplit,
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

    ######################## Cross-Validation
    selected_cv = sum([args.use_kfold, args.use_skfold, args.use_tscv, args.use_btscv])
    if selected_cv != 1:
        print("Please select one CV option.")
        return

    if args.use_kfold:
        CV = KFold
        cv_info = "kfold" + str(args.n_splits)
    elif args.use_skfold:
        CV = StratifiedKFold
        cv_info = "skfold" + str(args.n_splits)
    elif args.use_tscv:
        CV = TimeSeriesSplit
        cv_info = "tscv" + str(args.n_splits)
    elif args.use_btscv:
        CV = BlockingTimeSeriesSplit
        cv_info = "btscv" + str(args.n_splits)

    # X, y 값 분리
    label, features = custom_label_split(dataframe)

    cat_features = list(np.where(features.dtypes == np.object_)[0])
    model = CatBoost(args, cat_features)
    cv_scores = {"AUC": [], "ACC": []}

    cv = CV(n_splits=args.n_splits)

    for i, (train_idx, valid_idx) in enumerate(cv.split(features)):
        print(
            f"------------------------------ Fold {i + 1} ------------------------------"
        )
        X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
        y_train, y_valid = label.iloc[train_idx], label.iloc[valid_idx]

        ########################   TRAIN
        print("--------------- CatBoost Train   ---------------")
        model.train(X_train, y_train, X_valid, y_valid)

        ########################   VALID
        print("--------------- CatBoost Valid   ---------------")
        valid_preds = model.pred(X_valid)
        auc, acc = model.score(y_valid, valid_preds)

        cv_scores["AUC"].append(auc)
        cv_scores["ACC"].append(acc)
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

    ########################   PRINT THE CV SCORES
    print("--------------- Summarize Cross-Validation Scores   ---------------")
    print("CV Scores:")
    for fold in range(args.n_splits):
        print(
            f"Fold {fold + 1} - {', '.join([f'{metric_name}: {scores[fold]:.4f}' for metric_name, scores in cv_scores.items()])}"
        )

    print("\nMean CV Scores:")
    for metric_name, scores in cv_scores.items():
        print(f"{metric_name}: {np.mean(scores):.4f}")

    print("\nStd CV Scores:")
    for metric_name, scores in cv_scores.items():
        print(f"{metric_name}: {np.std(scores):.4f}")

    ########################   TRAIN ON THE ENTIRE DATASET
    print("--------------- CatBoost Train   ---------------")
    model.train(features, label)

    ########################   INFERENCE
    print("--------------- CatBoost Predict   ---------------")
    total_preds = model.pred(test_dataframe[FEATS])

    ######################## SAVE PREDICT
    print("\n--------------- Save Output Predict   ---------------")
    filename = setting.get_submit_filename(
        args.output_dir, np.mean(cv_scores["AUC"]), cv_info=cv_info
    )
    setting.save_predict(filename=filename, predict=total_preds)

    ######################## SAVE MODEL
    print("\n--------------- Save Model   ---------------")
    model_path = setting.get_submit_filename(
        output_dir=args.model_dir,
        auc_score=np.mean(cv_scores["AUC"]),
        cv_info=cv_info,
        format_name="cbm",
    )
    print(f"saving model : {model_path}")
    model.save_model(filename=model_path)

    ######################## SAVE CONFIG
    print("\n--------------- Save Config   ---------------")
    setting.save_config(args, np.mean(cv_scores["AUC"]), cv_info)


if __name__ == "__main__":
    args = train_parse_args()
    main(args=args)
