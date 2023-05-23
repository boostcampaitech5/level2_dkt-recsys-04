import os
import json
import argparse
import warnings
from easydict import EasyDict


import torch

from oof_stacking_util.trainer import Trainer, Stacking
from oof_stacking_util.args import parse_args
from oof_stacking_util.utils import set_seeds, get_logger, logging_conf, get_metric
from oof_stacking_util.datasets import Preprocess

from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    ########################   Set Random Seed
    print("--------------- Set Random Seed ---------------")
    logger.info("Set Seed ...")
    set_seeds(args.seed)

    ########################   Set device
    use_cuda: bool = torch.cuda.is_available() and args.use_cuda_if_available
    args.device = "cuda" if use_cuda else "cpu"
    print(args.device)

    ########################   Data Loader(load, preprocessing)
    print("---------------  Data Loader   ---------------")
    logger.info("Preparing data ...")

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.train_data_dir)

    train_data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(train_data)

    data = train_data
    # 기타 모델 성능 개선 전용으로 사용할 데이터
    size = int(len(train_data) * 0.8)

    temp_train_data = train_data[:size]
    temp_valid_data = train_data[size:]

    # 실제로 test data에는 target값이 들어가지만 임의의 target값이 배정된다
    temp_test_data = valid_data

    ########################   Model Build (load, preprocessing)
    print("---------------  Model Build   ---------------")
    logger.info("Building Model ...")

    trainer = Trainer()

    ########################   Single Model TRAIN
    # print("--------------- Model Train   ---------------")
    # logger.info("Start Training ...")
    # model = trainer.train(args, temp_train_data, temp_valid_data)  # 훈련 마친 모델 반환

    # valid_target = trainer.get_target(temp_valid_data)  # valid_data에 대한 target값
    # valid_predict = trainer.evaluate(
    #     args, model, temp_valid_data
    # )  # valid_data에 대한 predict값

    # # 검증셋 성능
    # valid_auc, valid_acc = get_metric(valid_target, valid_predict)
    # print(valid_auc, valid_acc)

    # # 여기 test데이터는 진짜 target값을 지니고 있다. 그렇기 때문에 여기서는
    # # test데이터에서 직접 target값을 추출하는 것이며, 실제 대회에서는
    # # test 데이터에는 임의의 fake target값이 들어가있다
    test_target = trainer.get_target(temp_test_data)  # test_data에 대한 target값
    # test_predict = trainer.test(args, model, temp_test_data)  # test_data에 대한 predict값

    # # 테스트셋 성능
    # test_auc, test_acc = get_metric(test_target, test_predict)
    # print(test_auc, test_acc)

    ########################   OOF Stacking
    print("--------------- Meta Model Train   ---------------")
    logger.info("Start Training ...")

    # oof stacking ensemble
    # 1) 각 모델별 args_list 불러오기
    # JSON 파일 불러오기
    with open("args_list.json", "r") as f:
        models_args = json.load(f)

    # 각 dict 원소를 EasyDict로 변환
    args_list = [EasyDict(args) for args in models_args]

    # 2) Meta 모델 학습
    # Train
    stacking = Stacking(Trainer())
    meta_model = LinearRegression()  # meta model

    meta_model, models_list, S_train, target = stacking.train(
        meta_model, args_list, data
    )

    print("--------------- Meta Model Test   ---------------")
    logger.info("Start Testing ...")
    # Test
    stacking = Stacking(Trainer())
    test_predict, S_test = stacking.test(
        meta_model, args_list, models_list, temp_test_data
    )
    test_target = trainer.get_target(temp_test_data)

    # 테스트셋 성능
    print("--------------- Meta Model Performance   ---------------")
    stack_test_auc, stack_test_acc = get_metric(test_target, test_predict)
    print(f"test_auc: {stack_test_auc}, test_acc:{stack_test_acc}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
