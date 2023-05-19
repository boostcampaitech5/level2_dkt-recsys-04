import os
import argparse

import numpy as np
import pandas as pd

from catboost_util.models import CatBoost
from catboost_util.utils import get_logger, Setting, logging_conf
from catboost_util.datasets import (
    feature_engineering,
)
from catboost_util.args import inference_parse_args

setting = Setting()
logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    ########################   Set Random Seed
    print("--------------- CatBoost Set Random Seed ---------------")
    setting.set_seeds(args.seed)

    ######################## DATA LOAD
    print("--------------- CatBoost Load Data ---------------")
    data_dir = args.data_dir
    # 테스트 세트 예측
    test_dataframe = pd.read_csv(os.path.join(data_dir, "test_data.csv"))

    ######################## Feature Engineering
    test_dataframe = feature_engineering(args.feats, test_dataframe)
    test_dataframe = test_dataframe[
        test_dataframe["userID"] != test_dataframe["userID"].shift(-1)
    ]
    ########################   LOAD Model
    print("--------------- CatBoost Load Model ---------------")
    cat_features = list(np.where(test_dataframe.dtypes == np.object_)[0])
    print(f"Categoy : {cat_features}")
    model_path = os.path.join(args.model_dir, args.model_name)
    model = CatBoost(
        args,
        cat_features=cat_features,
        model_path=model_path,
    )

    ########################   INFERENCE
    print("--------------- CatBoost Predict   ---------------")
    total_preds = model.pred(test_dataframe)

    ######################## SAVE PREDICT
    print("\n--------------- Save Output Predict   ---------------")
    filename = os.path.join(
        args.output_dir,
        args.model_name.replace("catboost", "catboost_inference").replace(
            "cbm", "csv"
        ),
    )
    setting.save_predict(filename=filename, predict=total_preds)


if __name__ == "__main__":
    args = inference_parse_args()
    main(args=args)
