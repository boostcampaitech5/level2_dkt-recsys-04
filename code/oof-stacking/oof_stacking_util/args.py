import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument(
        "--use_cuda_if_available", default=True, type=bool, help="Use GPU"
    )

    parser.add_argument(
        "--train_data_dir",
        default="/opt/ml/input/data/train_data.csv",
        type=str,
        help="",
    )
    parser.add_argument(
        "--test_data_dir",
        default="/opt/ml/input/data/test_data.csv",
        type=str,
        help="",
    )
    parser.add_argument(
        "--data_dir", default="/opt/ml/input/data/", type=str, help=""
    )

    # 데이터
    parser.add_argument("--max_seq_len", default=300, type=int, help="")

    # 데이터 증강
    parser.add_argument("--window", default=False, type=bool, help="")
    parser.add_argument("--stride", default=300, type=bool, help="")
    parser.add_argument("--shuffle", default=False, type=bool, help="")
    parser.add_argument("--shuffle_n", default=2, type=int, help="")

    parser.add_argument("--output_dir", default="./outputs/", type=str, help="")

    # 모델
    parser.add_argument("--hidden_dim", default=128, type=int, help="")
    parser.add_argument("--n_layers", default=1, type=int, help="")
    parser.add_argument("--dropout", default=0.0, type=float, help="")
    parser.add_argument("--n_heads", default=4, type=float, help="")

    # T Fixup
    parser.add_argument("--Tfixup", default=False, type=bool, help="")
    parser.add_argument("--layer_norm", default=True, type=bool, help="")

    # 훈련
    parser.add_argument("--n_epochs", default=5, type=int, help="")
    parser.add_argument("--lr", default=0.001, type=float, help="")
    parser.add_argument("--batch_size", default=64, type=int, help="")
    parser.add_argument("--clip_grad", default=0, type=int, help="")

    ### 중요 ###
    parser.add_argument("--model", default="bert", type=str, help="")
    parser.add_argument("--optimizer", default="adam", type=str, help="")
    parser.add_argument("--scheduler", default="plateau", type=str, help="")

    parser.add_argument("--model_dir", default="./models/", type=str, help="")
    # parser.add_argument("--model_name", default="best_model.pt", type=str, help="")

    args = parser.parse_args()

    return args
