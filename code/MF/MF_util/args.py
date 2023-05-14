import argparse


def parse_args():
    """ """
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="랜덤 시드를 지정합니다.")

    parser.add_argument(
        "--data_dir", default="/opt/ml/input/data", type=str, help="데이터 경로를 지정합니다."
    )

    parser.add_argument(
        "--output_dir",
        default="./outputs/",
        type=str,
        help="predict 결과를 저장할 경로를 지정합니다.",
    )

    parser.add_argument("--k", default=12, type=int, help="잠재행렬의 요소 수(k)를 지정합니다.")

    parser.add_argument(
        "--model",
        default="NMF",
        choices=["SVD", "NMF"],
        help="SVD, NMF 모델 중 하나를 선택합니다.",
    )

    args = parser.parse_args()

    return args
