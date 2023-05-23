import numpy as np
import pandas as pd
import hyperopt
from hyperopt import pyll, hp, fmin, tpe, STATUS_OK, Trials
from dkt.args import parse_args
from dkt.utils import set_seeds
from dkt.dataloader import Preprocess
from dkt.trainer import run
from dkt.model import LSTM, LSTMATTN


# 목적 함수
def objective_function(space):
    """
    space 예시 {'batch_size': 64, 'lr': 0.00010810929882981193, 'n_layers': 1}
    """
    # args가 dict으로 건네지기 때문에 easydict으로 바꿔준다
    args = space["args"]
    # args = easydict.EasyDict(args)

    # 하이퍼파라미터 값 변경
    args.max_seq_len = space["max_seq_len"]
    args.hidden_dim = space["hidden_dim"]
    args.n_layers = space["n_layers"]
    args.n_heads = space["n_heads"]
    args.drop_out = space["drop_out"]
    args.lr = space["lr"]
    args.train_data = space["train_data"]
    args.valid_data = space["valid_data"]

    # seed 설정
    set_seeds(args.seed)

    if args.model == "lstm":
        model = LSTM(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_tests=1538,
            n_questions=9455,
            n_tags=913,
        )
    best_auc = run(args, args.train_data, args.valid_data, model)

    return -best_auc  # 목적 함수 값을 -auc로 설정 => 목적 함수 최소화 => auc 최대화


# 결과 저장 함수
def trials_to_df(trials, space, best):
    # 전체 결과
    rows = []
    keys = list(trials.trials[0]["misc"]["vals"].keys())

    # 전체 실험결과 저장
    for trial in trials:
        row = {}

        # tid
        tid = trial["tid"]
        row["experiment"] = str(tid)

        # hyperparameter 값 저장
        vals = trial["misc"]["vals"]
        hparam = {key: value[0] for key, value in vals.items()}

        # space가 1개 - 값을 바로 반환
        # space가 다수 - dict에 값을 반환
        hparam = hyperopt.space_eval(space, hparam)

        if len(keys) == 1:
            row[keys[0]] = hparam
        else:
            for key in keys:
                row[key] = hparam[key]

        # metric
        row["metric"] = abs(trial["result"]["loss"])

        # 소요 시간
        row["time"] = (trial["refresh_time"] - trial["book_time"]).total_seconds()

        rows.append(row)

    experiment_df = pd.DataFrame(rows)

    # best 실험
    row = {}
    best_hparam = hyperopt.space_eval(space, best)

    if len(keys) == 1:
        row[keys[0]] = best_hparam
    else:
        for key in keys:
            row[key] = best_hparam[key]
    row["experiment"] = "best"

    best_df = pd.DataFrame([row])

    # best 결과의 auc / time searching 하여 찾기
    search_df = pd.merge(best_df, experiment_df, on=keys)

    # column명 변경
    search_df = search_df.drop(columns=["experiment_y"])
    search_df = search_df.rename(columns={"experiment_x": "experiment"})

    # 가장 좋은 metric 결과 중 가장 짧은 시간을 가진 결과를 가져옴
    best_time = search_df.time.min()
    search_df = search_df.query("time == @best_time")

    df = pd.concat([experiment_df, search_df], axis=0)

    return df


def main(args, train_data, valid_data):
    # sequential model parameters
    # max_seq_len

    # 모델 파라미터
    # hidden_dim *
    # n_layers *
    # n_heads *
    # drop_out *

    # 훈련 파라미터
    # n_epochs - 50 ~ 100 정도면 됨 굳이 X
    # batch_size
    # lr *
    # clip_grad
    # patience
    # log_steps

    # 탐색 공간
    space = {  # 범위는 넓게 잡을 수록 좋다
        "max_seq_len": hp.choice(
            "max_seq_len", [10, 20, 50, 100, 500, 1000]
        ),  # default=20
        "hidden_dim": hp.choice("hidden_dim", [32, 64, 128, 256, 512]),  # default=64
        "n_layers": hp.choice("n_layers", [1, 2, 3]),  # default=2
        "n_heads": hp.choice(
            "n_heads", [1, 2, 3, 4]
        ),  # default=2  # multi-head attention
        "drop_out": hp.uniform("drop_out", 0.1, 0.9),  # default=0.2
        "lr": hp.uniform("lr", 0.00001, 0.005),  # default=0.0001
        "args": args,
        "train_data": train_data,
        "valid_data": valid_data,
    }

    # 최적화
    # 하나당 3 epoch를 돌리기 때문에 최대한 숫자를 줄이기위해 5번만 시도

    trials = Trials()
    best = fmin(
        fn=objective_function,  # 최적화 할 함수 (목적 함수)
        space=space,  # Hyperparameter 탐색 공간
        algo=tpe.suggest,  # 베이지안 최적화 적용 알고리즘 : Tree-structured Parzen Estimator (TPE)
        max_evals=2,  # 입력 시도 횟수
        trials=trials,  # 시도한 입력 값 및 입력 결과 저장
        rstate=np.random.default_rng(
            seed=42
        ),  ## fmin()을 시도할 때마다 동일한 결과를 가질 수 있도록 설정하는 랜덤 시드
    )

    print("best:", best)
    print(trials.results)
    print(trials.vals)

    # 하이퍼파라메타 원상복구
    args.n_epochs = 10
    args.lr = 0.0001
    args.batch_size = 64
    args.n_layers = 1

    # 출력
    df = trials_to_df(trials, space, best)
    df.sort_values(by="metric", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("./tune_result.csv", index=False)
    print(df)


if __name__ == "__main__":
    args = parse_args()
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data=train_data)
    print("dataload done")

    main(args, train_data, valid_data)
