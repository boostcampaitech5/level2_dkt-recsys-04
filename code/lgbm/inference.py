import os
import argparse

import pandas as pd

from lgbm_util.models import LGBM
from lgbm_util.utils import get_logger, Setting, logging_conf
from lgbm_util.datasets import (
    feature_engineering,
    FEATS,
)
from lgbm_util.args import inference_parse_args

setting = Setting()
logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    ########################   Set Random Seed
    print("--------------- LGBM Set Random Seed ---------------")
    setting.set_seeds(args.seed)

    ######################## DATA LOAD
    print("--------------- LGBM Load Data ---------------")
    data_dir = args.data_dir

    # 테스트 세트 예측
    test_dataframe = pd.read_csv(os.path.join(data_dir, "test_data.csv"))

    ######################## Feature Engineering
    test_dataframe = feature_engineering(test_dataframe)
    test_dataframe = test_dataframe[
        test_dataframe["userID"] != test_dataframe["userID"].shift(-1)
    ]
    ########################   LOAD Model
    print("--------------- LGBM Load Model ---------------")
    model_path = os.path.join(args.model_dir, args.model_name)
    model = LGBM(
        args,
        model_path=model_path,
    )

    ########################   INFERENCE
    print("--------------- LGBM Predict   ---------------")
    total_preds = model.pred(test_dataframe[FEATS])

    ######################## SAVE PREDICT
    print("\n--------------- Save Output Predict   ---------------")
    filename = os.path.join(
        args.output_dir,
        args.model_name.replace("lgbm", "lgbm_inference").replace("txt", "csv"),
    )
    setting.save_predict(filename=filename, predict=total_preds)


if __name__ == "__main__":
    args = inference_parse_args()
    main(args=args)
