import argparse


def train_parse_args() -> argparse.Namespace:
    """ """
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
        "--num_iterations", default=1000, type=int, help="num_iterations"
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--depth", default=6, type=int, help="depth")
    parser.add_argument("--use_kfold", default=True, type=bool, help="Use K-Fold CV")
    parser.add_argument(
        "--use_skfold", default=False, type=bool, help="Use Stratified K-Fold CV"
    )
    parser.add_argument(
        "--use_tscv", default=False, type=bool, help="Use Time Series CV"
    )
    parser.add_argument(
        "--use_btscv", default=False, type=bool, help="Use Blocking Time Series CV"
    )
    parser.add_argument(
        "--n_splits", default=5, type=int, help="n_splits in Cross-Validation"
    )
    args = parser.parse_args()

    return args


def inference_parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--data_dir", default="/opt/ml/input/data", type=str, help="")
    parser.add_argument(
        "--output_dir", default="./outputs/", type=str, help="output dir"
    )
    parser.add_argument("--model_dir", default="./models/", type=str, help="model dir")
    parser.add_argument(
        "--model_name", default="catboost.cbm", type=str, help="output dir"
    )

    args = parser.parse_args()

    return args
