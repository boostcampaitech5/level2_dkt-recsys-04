import os

import torch

from lightgcn_utils.args import parse_args
from lightgcn_utils.datasets import prepare_dataset
from lightgcn_utils import models, trainer
from lightgcn_utils.utils import get_logger, logging_conf, set_seeds


logger = get_logger(logging_conf)


def main(args):
    ########################   Set Random Seed
    print("--------------- LightGCN Set Random Seed ---------------")
    set_seeds(args.seed)

    ########################   Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################   Data Loader(load, preprocessing)
    print("--------------- LightGCN Data Loader   ---------------")
    logger.info("Preparing data ...")
    train_data, test_data, n_node = prepare_dataset(
        device=device, data_dir=args.data_dir
    )

    ########################   LOAD Model
    print("--------------- LightGCN Load Model ---------------")
    logger.info("Loading Model ...")
    weight: str = os.path.join(args.model_dir, args.model_name)  # best_model.pt 파일 load
    model: torch.nn.Module = models.build(
        n_node=n_node,
        embedding_dim=args.hidden_dim,
        num_layers=args.n_layers,
        alpha=args.alpha,
        weight=weight,
    )
    model = model.to(device)

    ########################   INFERENCE
    print("--------------- LightGCN Predict   ---------------")
    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(model=model, data=test_data, output_dir=args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
