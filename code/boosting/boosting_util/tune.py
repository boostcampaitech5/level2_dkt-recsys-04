import os
import numpy as np
import pandas as pd
import torch
import gc
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
import lightgbm as lgb

import easydict
from hyperopt import hp, fmin, tpe, Trials

from args import cv_parse_args
from utils import trials_to_df, Setting
from datasets import (
    preprocessing,
    feature_engineering,
    custom_label_split,
)

from models import Model


# 목적 함수
def objective_function(space):
    # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
    torch.cuda.empty_cache()
    gc.collect()

    space = easydict.EasyDict(space)
    args = space["args"]

    # 하이퍼파라미터 값 변경
    args.lr = space["lr"]
    args.depth = space["depth"]
    label = space["label"]
    features = space["features"]
    cat_features = space["cat_features"]
    cv = space["cv"]

    cv_scores = {"AUC": [], "ACC": []}
    for i, (train_idx, valid_idx) in enumerate(cv.split(features)):
        print(
            f"------------------------------ kfold {i + 1} ------------------------------"
        )
        X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
        y_train, y_valid = label.iloc[train_idx], label.iloc[valid_idx]

        ########################   TRAIN
        # print(f"--------------- {args.model} Train   ---------------")
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
        # print(f"--------------- {args.model} Valid   ---------------")
        valid_preds = model.pred(X_valid)
        auc, acc = model.score(y_valid, valid_preds)

        cv_scores["AUC"].append(auc)
        cv_scores["ACC"].append(acc)

    total_auc = np.mean(cv_scores["AUC"])
    print(f"--------------- AUC: {total_auc} ---------------")

    # setting.save_config(args, total_auc, "kfold")

    return -1 * total_auc  # 목적 함수 값을 -auc로 설정 => 목적 함수 최소화 => auc 최대화


def main(args, label, features, cat_features, cv):
    # 탐색 공간
    space = {
        "lr": hp.uniform("lr", 0.01, 0.2),  # default=0.1
        "depth": hp.choice("depth", [6, 7, 8, 9, 10, 11, 12]),  # default=6
        "args": args,
        "seed": args.seed,
        "label": label,
        "features": features,
        "cat_features": cat_features,
        "cv": cv,
    }

    # 최적화
    trials = Trials()
    best = fmin(
        fn=objective_function,  # 최적화 할 함수 (목적 함수)
        space=space,  # Hyperparameter 탐색 공간
        algo=tpe.suggest,  # 베이지안 최적화 적용 알고리즘 : Tree-structured Parzen Estimator (TPE)
        max_evals=30,  # 입력 시도 횟수
        trials=trials,  # 시도한 입력 값 및 입력 결과 저장
        rstate=np.random.default_rng(
            seed=args.seed
        ),  ## fmin()을 시도할 때마다 동일한 결과를 가질 수 있도록 설정하는 랜덤 시드
    )

    print("best:", best)

    # 출력 & 저장
    df = trials_to_df(trials, space, best)
    df.sort_values(by="metric", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("./tune_result.csv", index=False)


if __name__ == "__main__":
    args = cv_parse_args()

    setting = Setting(args)
    setting.set_seeds(args.seed)

    ######################## DATA LOAD
    print(f"--------------- {args.model} Load Data ---------------")
    data_dir = args.data_dir
    csv_file_path = os.path.join(data_dir, "train_data.csv")
    dataframe = pd.read_csv(csv_file_path)

    ######################## Feature Engineering
    dataframe = feature_engineering(args.feats, dataframe)

    ######################## Preprocessing
    cat_features, dataframe = preprocessing(dataframe)  # Category Feature 선택

    ########################   Data Loader
    # X, y 값 분리
    label, features = custom_label_split(dataframe)

    ######################## Cross Validation
    print(f"--------------- {args.model} Cross Validation   ---------------")
    cv = KFold(n_splits=args.n_splits)

    main(args, label, features, cat_features, cv)
