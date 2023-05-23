import os

import torch
import gc

import numpy as np
import pandas as pd

import random

import warnings

warnings.filterwarnings("ignore")

from collections import OrderedDict

import easydict

from sequence_utils.args import load_args
from sequence_utils.datasets import Preprocess, data_augmentation, get_loaders
from sequence_utils.trainer import get_model, get_optimizer, get_scheduler
from sequence_utils.trainer import train, validate
from sequence_utils.utils import seed_everything, get_logger, logging_conf, trials_to_df

import hyperopt
from hyperopt import pyll, hp, fmin, tpe, STATUS_OK, Trials

args = load_args()
logger = get_logger(logger_conf=logging_conf)


# 목적 함수
def objective_function(space):
    """
    space 예시 {'batch_size': 64, 'lr': 0.00010810929882981193, 'n_layers': 1}
    """
    # space 가 dict으로 건네지기 때문에 easydict으로 바꿔준다

    # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
    torch.cuda.empty_cache()
    gc.collect()

    space = easydict.EasyDict(space)
    args = space["args"]

    # 하이퍼파라미터 값 변경
    args.max_seq_len = space["max_seq_len"]
    args.hidden_dim = space["hidden_dim"]
    args.n_layers = space["n_layers"]
    args.n_heads = space["n_heads"]
    args.dropout = space["dropout"]
    args.lr = space["lr"]
    args.window = space["window"]
    args.stride = space["stride"]

    seed = space["seed"]

    logger.info("Loading Data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=os.path.join(args["data_dir"], "train_data.csv"))
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data=train_data)

    ## augmentation
    augmented_train_data = data_augmentation(train_data, args)
    if len(augmented_train_data) != len(train_data):
        print(f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n")

    train_loader, valid_loader = get_loaders(args, augmented_train_data, valid_data)

    logger.info("Building Model ...")
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    gradient = False
    # gradient step 분석에 사용할 변수
    if gradient:
        args.n_iteration = 0
        args.gradient = OrderedDict()

        # 모델의 gradient값을 가리키는 모델 명 저장
        args.gradient["name"] = [name for name, _ in model.named_parameters()]

    # seed 설정
    seed_everything(seed)

    best_auc = -1
    best_auc_epoch = -1

    logger.info(f"Training Model ...")
    for epoch in range(args.n_epochs):
        ### TRAIN
        train_loss, train_auc, train_acc = train(train_loader, model, optimizer, scheduler, args, gradient)

        ### VALID
        valid_auc, valid_acc, preds, targets = validate(valid_loader, model, args)

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_auc_epoch = epoch + 1

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)

    logger.info(f"Best Weight Confirmed : {best_auc_epoch}'th epoch & Best score : {best_auc}")

    return -1 * best_auc  # 목적 함수 값을 -auc로 설정 => 목적 함수 최소화 => auc 최대화


def main(args):
    # seed 설정
    seed_everything(args.seed)

    args_origin = args.copy()

    # 탐색 공간
    space = {
        "max_seq_len": hp.choice("max_seq_len", [10, 20, 50, 100, 500, 1000]),
        "hidden_dim": hp.choice("hidden_dim", [64, 128, 256, 512]),
        "n_layers": hp.choice("n_layers", [1, 2, 3]),
        "n_heads": hp.choice("n_heads", [1, 2, 4]),
        "dropout": hp.uniform("dropout", 0.1, 0.9),
        "lr": hp.uniform("lr", 0.00005, 0.001),
        "window": hp.choice("window", [True, False]),
        "stride": hp.choice("stride", [10, 20, 50, 100, 500]),
        "seed": args.seed,
        "args": args,
    }

    # 최적화
    trials = Trials()
    best = fmin(
        fn=objective_function,  # 최적화 할 함수 (목적 함수)
        space=space,  # Hyperparameter 탐색 공간
        algo=tpe.suggest,  # 베이지안 최적화 적용 알고리즘 : Tree-structured Parzen Estimator (TPE)
        max_evals=2,  # 입력 시도 횟수
        trials=trials,  # 시도한 입력 값 및 입력 결과 저장
        rstate=np.random.default_rng(seed=args.seed),  ## fmin()을 시도할 때마다 동일한 결과를 가질 수 있도록 설정하는 랜덤 시드
    )

    print("best:", best)

    # 하이퍼파라미터 원상복구
    # args.max_seq_len = args_origin["max_seq_len"]
    # args.hidden_dim = args_origin["hidden_dim"]
    # args.n_layers = args_origin["n_layers"]
    # args.n_heads = args_origin["n_heads"]
    # args.dropout = args_origin["dropout"]
    # args.lr = args_origin["lr"]
    # args.window = args_origin["window"]
    # args.stride = args_origin["stride"]

    # Save
    df = trials_to_df(trials, space, best)
    df.sort_values(by="metric", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("./tune_result.csv", index=False)


if __name__ == "__main__":
    args = load_args()

    main(args)
