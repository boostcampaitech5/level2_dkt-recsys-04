import os

import torch
import gc

import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

from collections import OrderedDict

from sequence_utils.args import load_args
from sequence_utils.datasets import Preprocess, data_augmentation, get_loaders
from sequence_utils.trainer import get_model, get_optimizer, get_scheduler
from sequence_utils.trainer import train, validate
from sequence_utils.utils import seed_everything, get_logger, logging_conf

import time
import shutil
import wandb


args = load_args()
logger = get_logger(logger_conf=logging_conf)


def main(args, gradient=False):
    ########################   Set Random Seed
    logger.info("Set Seed ...")
    seed_everything(args.seed)

    # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
    torch.cuda.empty_cache()
    gc.collect()

    ########################   Set WandB
    logger.info("Set WandB ...")
    wandb.login()
    wandb.init(project="dkt", config=vars(args))


    ########################   Data Loader(load, preprocessing)
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(os.path.join(args["data_dir"], "train_data.csv"))

    train_data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(train_data)

    ## augmentation
    augmented_train_data = data_augmentation(train_data, args)
    if len(augmented_train_data) != len(train_data):
        print(f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n")

    train_loader, valid_loader = get_loaders(args, augmented_train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10

    ########################   Model Build
    logger.info("Building Model ...")
    model = get_model(args)

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    # 🌟 분석에 사용할 값 저장 🌟
    report = OrderedDict()

    # gradient step 분석에 사용할 변수
    if gradient:
        args.n_iteration = 0
        args.gradient = OrderedDict()

        # 모델의 gradient값을 가리키는 모델 명 저장
        args.gradient["name"] = [name for name, _ in model.named_parameters()]

    best_auc = -1
    best_auc_epoch = -1
    best_acc = -1
    best_acc_epoch = -1
    
    ########################   TRAIN
    logger.info(f"Training Started : n_epochs={args.n_epochs}")
    os.makedirs(name=args.model_dir, exist_ok=True)
    shutil.copy(f'{os.getcwd()}/sequence_utils/config.py', args.model_dir)

    for epoch in range(args.n_epochs):
        epoch_report = {}

        ### TRAIN
        train_start_time = time.time()
        train_loss, train_auc, train_acc = train(train_loader, model, optimizer, scheduler, args, gradient)
        train_time = time.time() - train_start_time

        epoch_report["train_auc"] = train_auc
        epoch_report["train_acc"] = train_acc
        epoch_report["train_time"] = train_time

        ### VALID
        valid_start_time = time.time()
        valid_auc, valid_acc, preds, targets = validate(valid_loader, model, args)
        valid_time = time.time() - valid_start_time

        epoch_report["valid_auc"] = valid_auc
        epoch_report["valid_acc"] = valid_acc
        epoch_report["valid_time"] = valid_time

        logger.info("Epoch: %s / %s, train_loss: %.4f, train_auc: %.4f, valid_auc: %.4f", 
                    epoch+1, args.n_epochs, train_loss, train_auc, valid_auc)
        # save lr
        epoch_report["lr"] = optimizer.param_groups[0]["lr"]

        # 🌟 save it to report 🌟
        report[f"{epoch + 1}"] = epoch_report

        wandb.log(
            dict(
                train_loss_epoch=train_loss,
                train_acc_epoch=train_acc,
                train_auc_epoch=train_auc,
                valid_acc_epoch=valid_acc,
                valid_auc_epoch=valid_auc,
            )
        )

        if valid_auc > best_auc:
            logger.info(
                "Best model updated AUC from %.4f to %.4f at %s", best_auc, valid_auc, epoch
            )
            best_auc = valid_auc
            best_auc_epoch = epoch + 1
            torch.save(
                obj={"model": model.state_dict(), "epoch": best_auc_epoch},
                f=os.path.join(args.model_dir, f"updated_model_{valid_auc}.pt"),
            )
            torch.save(
                obj={"model": model.state_dict(), "epoch": best_auc_epoch},
                f=os.path.join(args.model_dir, f"best_model.pt"),
            )

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_acc_epoch = epoch + 1

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)

    logger.info(f"Best Weight Confirmed : {best_auc_epoch}'th epoch & Best score : {best_auc}")

    # save best records
    report["best_auc"] = best_auc
    report["best_auc_epoch"] = best_auc_epoch
    report["best_acc"] = best_acc
    report["best_acc_epoch"] = best_acc_epoch

    # save gradient informations
    if gradient:
        report["gradient"] = args.gradient
        del args.gradient
        del args["gradient"]

    # return report

if __name__ == "__main__":
    args = load_args()
    main(args)
