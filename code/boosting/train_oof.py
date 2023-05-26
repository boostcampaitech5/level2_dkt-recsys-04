import warnings
import os
import argparse
from pprint import pprint

import wandb
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from boosting_util.models import Model
from boosting_util.utils import get_logger, Setting, logging_conf
from boosting_util.datasets import (
    preprocessing,
    feature_engineering,
    custom_train_test_split,
    custom_label_split,
    BlockingTimeSeriesSplit,
)
from boosting_util.args import cv_parse_args
from boosting_util.custom_cv import UserBasedKFold

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

    ######################## Cross-Validation
    selected_cv = sum(
        [
            args.use_kfold,
            args.use_skfold,
            args.use_tscv,
            args.use_btscv,
            args.user_based_kfold,
        ]
    )
    if selected_cv != 1:
        print("Please select one CV option.")
        return

    if args.use_kfold:
        CV = KFold
        cv_info = "kfold"
    elif args.use_skfold:
        CV = StratifiedKFold
        cv_info = "skfold"
    elif args.use_tscv:
        CV = TimeSeriesSplit
        cv_info = "tscv"
    elif args.use_btscv:
        CV = BlockingTimeSeriesSplit
        cv_info = "btscv"
    elif args.user_based_kfold:
        CV = UserBasedKFold
        cv_info = "user_based_kfold"

    ########################   Data Split
    print(f"--------------- {args.model} Data Split   ---------------")
    # 유저별 분리
    train, test = custom_train_test_split(
        dataframe, user_id_dir=args.user_id_dir
    )
    oof_train = train[train["userID"] != train["userID"].shift(-1)]
    oof_test = test[test["userID"] != test["userID"].shift(-1)]

    ########################   Data Loader
    print(f"--------------- {args.model} Data Loader   ---------------")
    # X, y 값 분리
    label, features = custom_label_split(train)

    oof_train_label, oof_train = custom_label_split(oof_train)
    oof_test_label, oof_test = custom_label_split(oof_test)

    ####################### Cross Validation
    print(f"--------------- {args.model} Cross Validation   ---------------")
    cv_scores = {"AUC": [], "ACC": []}
    cv = CV(n_splits=args.n_splits)
    oof = np.zeros(oof_train.shape[0])
    print(f"Features Shape : {oof_train.shape[0]}")
    for i, (train_idx, valid_idx) in enumerate(cv.split(features)):
        print(
            f"------------------------------ {cv_info} {i + 1} ------------------------------"
        )
        X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
        y_train, y_valid = label.iloc[train_idx], label.iloc[valid_idx]

        oof_X_valid = oof_train[oof_train["userID"].isin(X_valid["userID"])]
        oof_Y_valid = oof_train_label[
            oof_train["userID"].isin(X_valid["userID"])
        ]

        ########################   TRAIN
        print(f"--------------- {args.model} Train   ---------------")
        model = Model(args.model, args, cat_features).load_model()
        if args.model == "LGBM":
            lgb_train = lgb.Dataset(
                X_train,
                y_train,
                categorical_feature=cat_features,
            )
            lgb_valid = lgb.Dataset(
                X_valid,
                y_valid,
                reference=lgb_train,
                categorical_feature=cat_features,
            )
            model.train(lgb_train=lgb_train, lgb_valid=lgb_valid)
        else:
            model.train(X_train, y_train, X_valid, y_valid)

        ########################   VALID
        print(f"--------------- {args.model} Valid   ---------------")
        valid_preds = model.pred(oof_X_valid)
        oof[oof_train["userID"].isin(X_valid["userID"])] = valid_preds

        auc, acc = model.score(oof_Y_valid, valid_preds)
        cv_scores["AUC"].append(auc)
        cv_scores["ACC"].append(acc)
        print(f"VALID AUC : {auc} ACC : {acc}\n")
        print(f"BEST VALIDATION : {model.best_validation_score}\n")
        print("Feature Importance : ")
        feature_importance = sorted(
            dict(
                zip(
                    X_train.columns.values.tolist(),
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

        ######################## SAVE MODEL
        print("\n--------------- Save Model   ---------------")
        format_name = "cbm" if args.model == "CatBoost" else "txt"
        model_path = setting.get_submit_filename(
            output_dir=args.model_dir,
            auc_score=auc,
            cv_info=cv_info + f"_{str(i+1)}",
            format_name=format_name,
        )
        print(f"saving model : {model_path}")
        model.save_model(filename=model_path)

    ######################## SAVE S_Train
    oof = oof.reshape(-1, 1)
    oof = pd.DataFrame(oof, columns=["prediction"])
    oof.to_csv(
        "/opt/ml/level2_dkt-recsys-04/code/boosting/oof_data/S_train.csv",
        index=False,
    )

    ########################   Second VALID
    print(f"--------------- {args.model} Valid   ---------------")
    valid_preds = model.pred(oof_test)

    ######################## SAVE S_test
    s_test = pd.DataFrame(valid_preds, columns=["prediction"])
    s_test.to_csv(
        "/opt/ml/level2_dkt-recsys-04/code/boosting/oof_data/S_test.csv",
        index=False,
    )

    ######################## Model Score
    auc, acc = model.score(oof_test_label, valid_preds)
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

    # ########################   INFERENCE
    # print(f"--------------- {args.model} Predict   ---------------")
    # total_preds = model.pred(test_dataframe.drop(["answerCode"], axis=1))

    # ######################## SAVE PREDICT
    # print("\n--------------- Save Output Predict   ---------------")
    # filename = setting.get_submit_filename(
    #     output_dir=args.output_dir,
    #     auc_score=model.best_validation_score,
    #     cv_info=cv_info,
    # )
    # setting.save_predict(filename=filename, predict=total_preds)

    # ######################## SAVE MODEL
    # print("\n--------------- Save Model   ---------------")
    # format_name = "cbm" if args.model == "CatBoost" else "txt"
    # model_path = setting.get_submit_filename(
    #     output_dir=args.model_dir,
    #     auc_score=model.best_validation_score,
    #     format_name=format_name,
    #     cv_info=cv_info,
    # )
    # print(f"saving model : {model_path}")
    # model.save_model(filename=model_path)

    ######################## SAVE CONFIG
    print("\n--------------- Save Config   ---------------")
    print(
        f"Inferencing CV : python inference_cv.py --model={args.model} --model_name={setting.save_time}"
    )
    setting.save_config(args, model.best_validation_score, cv_info=cv_info)


if __name__ == "__main__":
    args = cv_parse_args()
    main(args=args)
