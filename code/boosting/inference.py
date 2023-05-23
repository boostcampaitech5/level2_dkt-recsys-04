import os
import argparse

import numpy as np
import pandas as pd

from boosting_util.models import Model
from boosting_util.utils import get_logger, Setting, logging_conf
from boosting_util.datasets import feature_engineering, preprocessing
from boosting_util.args import inference_parse_args

logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    setting = Setting(args)

    ########################   Set Random Seed
    print(f"--------------- {args.model} Set Random Seed ---------------")
    setting.set_seeds(args.seed)

    ######################## DATA LOAD
    print(f"--------------- {args.model} Load Data ---------------")
    data_dir = args.data_dir
    # 테스트 세트 예측
    test_dataframe = pd.read_csv(os.path.join(data_dir, "test_data.csv"))

    ######################## Feature Engineering
    test_dataframe = feature_engineering(args.feats, test_dataframe)
    test_dataframe = test_dataframe[
        test_dataframe["userID"] != test_dataframe["userID"].shift(-1)
    ]
    # Category Feature 선택
    cat_features, test_dataframe = preprocessing(test_dataframe)

    ########################   LOAD Model
    print(f"--------------- {args.model} Load Model ---------------")
    print(f"Categoy : {cat_features}")
    model_path = os.path.join(args.model_dir, args.model, args.model_name)
    model = Model(
        args.model,
        args,
        cat_features=cat_features,
        model_path=model_path,
    ).load_model()

    ########################   INFERENCE
    print(f"--------------- {args.model} Predict   ---------------")
    total_preds = model.pred(test_dataframe.drop(["answerCode"], axis=1))

    ######################## SAVE PREDICT
    print("\n--------------- Save Output Predict   ---------------")
    filename = os.path.join(
        args.output_dir,
        args.model,
        args.model_name.replace("catboost", "catboost_inference")
        .replace("lgbm", "lgbm_inference")
        .replace("cbm", "csv"),
    )
    setting.save_predict(filename=filename, predict=total_preds)


if __name__ == "__main__":
    args = inference_parse_args()
    main(args=args)
