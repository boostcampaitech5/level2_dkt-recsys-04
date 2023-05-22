import os

import torch
import gc

import numpy as np
import random
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from collections import OrderedDict

import argparse

from sequence_utils.args import load_args
from sequence_utils.datasets import Preprocess, data_augmentation, get_loaders
from sequence_utils.trainer import get_model, get_optimizer, get_scheduler
from sequence_utils.trainer import train, validate, process_batch
from sequence_utils.utils import seed_everything, get_logger, logging_conf

import time
import shutil
import wandb


args = load_args()
logger = get_logger(logger_conf=logging_conf)


def main(args, load_path, gradient=False):
    ########################   Set Random Seed
    logger.info("Set Seed ...")
    seed_everything(args.seed)

    # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
    torch.cuda.empty_cache()
    gc.collect()

    ########################   Data Loader(load, preprocessing)
    logger.info("Preparing data ...")
    preprocess = Preprocess(args, is_train=False)
    preprocess.load_train_data(os.path.join(args["data_dir"], "test_data.csv"))

    test_data = preprocess.get_train_data()

    test_loader, _ = get_loaders(args, test_data, None, inference=True)

    # only when using warmup scheduler
    args.total_steps = int(len(test_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10

    ########################   Model Build
    logger.info("Loading Model ...")
    model = get_model(args)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model"], strict=False)
    
    model.eval()
    
    total_preds = []
    ########################   TRAIN
    logger.info(f"Inference Started")
    os.makedirs(name=os.path.join(args.model_dir, 'outputs'), exist_ok=True)
    shutil.copy(f'{os.getcwd()}/sequence_utils/config.py', os.path.join(args.model_dir, 'outputs'))

    total_preds = []
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            ### INFERENCE
            input = process_batch(batch, args)

            preds = model(input)
            index = input[-1]

            preds = preds.gather(1, index).view(-1)
            
            if args.device == "cuda":
                preds = preds.to("cpu").detach().numpy()
            else:  # cpu
                preds = preds.detach().numpy()

            total_preds.append(preds)
        
        total_preds = np.concatenate(total_preds)
    
    write_path = os.path.join(os.path.join(args.model_dir, 'outputs'), "submission.csv")
    pd.DataFrame({"prediction": total_preds}).to_csv(
        path_or_buf=write_path, index_label="id"
    )    
    logger.info(f"Successfully saved submission")


if __name__ == "__main__":
    args = load_args()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, help="model load path")
    parse = parser.parse_args()

    main(args, parse.load_path)
