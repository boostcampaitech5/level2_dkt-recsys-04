import argparse


def parse_args():
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
        "--output_dir", default="./outputs/", type=str, help="output dir"
    )

    parser.add_argument("--num_iterations", default=1000, type=int, help="")
    parser.add_argument("--lr", default=0.1, type=float, help="")

    args = parser.parse_args()

    return args
