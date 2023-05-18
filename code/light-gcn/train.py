import os
import argparse

import torch
import wandb

from lightgcn_utils.args import parse_args
from lightgcn_utils.datasets import prepare_dataset
from lightgcn_utils import models
from lightgcn_utils.utils import get_logger, set_seeds, logging_conf


logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    ########################   Set Random Seed
    print("--------------- LightGCN Set Random Seed ---------------")
    set_seeds(args.seed)

    ########################   Set WandB
    print("--------------- LightGCN Set WandB ---------------")
    wandb.login()
    wandb.init(project="dkt", config=vars(args))

    ########################   Set device
    use_cuda: bool = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    ########################   Data Loader(load, preprocessing)
    print("--------------- LightGCN Data Loader   ---------------")
    logger.info("Preparing data ...")
    train_data, test_data, n_node = prepare_dataset(
        device=device, data_dir=args.data_dir
    )  # n_node: id2index의 길이 <- 고유한 item과 user의 수

    ########################   LightGCN Build
    print("--------------- LightGCN Model Build   ---------------")
    logger.info("Building Model ...")
    model = models.build(
        n_node=n_node,
        embedding_dim=args.hidden_dim,
        num_layers=args.n_layers,
        alpha=args.alpha,
    )
    model = model.to(device)

    ########################   TRAIN
    print("--------------- LightGCN Train   ---------------")
    logger.info("Start Training ...")
    models.run(
        model=model,
        train_data=train_data,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
