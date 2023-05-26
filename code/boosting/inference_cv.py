import os
import argparse

import numpy as np
import pandas as pd

from boosting_util.models import Model
from boosting_util.utils import (
    get_logger,
    Setting,
    logging_conf,
    find_files_with_string,
    calculate_average_from_list,
    calculate_average_score_from_extract_numbers_from_strings,
)
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
    # test_dataframe = test_dataframe[
    #     test_dataframe["userID"] != test_dataframe["userID"].shift(-1)
    # ]
    # Category Feature 선택
    cat_features, test_dataframe = preprocessing(test_dataframe)

    ########################   Cross Validation Inference
    model_dir_path = os.path.join(args.model_dir, args.model)
    models_path = find_files_with_string(
        path=model_dir_path, search_string=args.model_name
    )
    predicts = []
    for idx, model_path in enumerate(models_path):
        ########################   LOAD Model
        print(
            f"--------------- {args.model} Load Model #{idx + 1}---------------"
        )
        model = Model(
            args.model,
            args,
            cat_features=cat_features,
            model_path=os.path.join(model_dir_path, model_path),
        ).load_model()

        ########################   INFERENCE
        print(
            f"--------------- {args.model} Predict #{idx + 1} ---------------"
        )
        predicts.append(model.pred(test_dataframe.drop(["answerCode"], axis=1)))

    total_auc = calculate_average_score_from_extract_numbers_from_strings(
        models_path
    )
    total_preds = calculate_average_from_list(predicts=predicts)

    ######################## SAVE PREDICT
    print("\n--------------- Save Output Predict   ---------------")
    setting.make_dir(args.output_dir)
    setting.make_dir(os.path.join(args.output_dir, args.model))
    filename = os.path.join(
        args.output_dir,
        args.model,
        args.model_name + f"_{total_auc}_{args.model.lower()}_cv_inference.csv",
    )

    setting.save_predict(
        filename=filename,
        predict=total_preds[
            test_dataframe["userID"] != test_dataframe["userID"].shift(-1)
        ],
    )


if __name__ == "__main__":
    args = inference_parse_args()
    main(args=args)
