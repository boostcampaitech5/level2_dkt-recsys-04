import argparse


def train_parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument(
        "--use_cuda_if_available", default=True, type=bool, help="Use GPU"
    )

    parser.add_argument(
        "--data_dir", default="/opt/ml/input/data", type=str, help=""
    )
    parser.add_argument(
        "--model_dir", default="./models/", type=str, help="model dir"
    )
    parser.add_argument(
        "--output_dir", default="./outputs/", type=str, help="output dir"
    )
    parser.add_argument(
        "--num_boost_round", default=500, type=int, help="num_boost_round"
    )
    parser.add_argument("--num_leaves", default=31, type=int, help="num_leaves")

    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")

    args = parser.parse_args()

    return args


def inference_parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument(
        "--data_dir", default="/opt/ml/input/data", type=str, help=""
    )
    parser.add_argument(
        "--output_dir", default="./outputs/", type=str, help="output dir"
    )
    parser.add_argument(
        "--model_dir", default="./models/", type=str, help="model dir"
    )
    parser.add_argument(
        "--model_name", default="lgbm.txt", type=str, help="output filename"
    )

    args = parser.parse_args()

    return args
