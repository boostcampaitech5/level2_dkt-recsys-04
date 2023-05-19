import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch_geometric.nn.models import LightGCN
import wandb

from .utils import get_logger, logging_conf


logger = get_logger(logger_conf=logging_conf)


########################   Model Build
def build(n_node: int, weight: str = None, **kwargs):
    """
    Args:
        num_nodes (int): The number of nodes in the graph.
        **kwargs (optional)
        - embedding_dim (int): The dimensionality of node embeddings.
        - num_layers (int): The number of
            :class:`~torch_geometric.nn.conv.LGConv` layers.
        - alpha (float or torch.Tensor, optional): The scalar or vector
            specifying the re-weighting coefficients for aggregating the final
            embedding. If set to :obj:`None`, the uniform initialization of
            :obj:`1 / (num_layers + 1)` is used. (default: :obj:`None`)
    Returns:
        model
    """

    model = LightGCN(num_nodes=n_node, **kwargs)
    if weight:  # 사전에 학습된 모델 가중치의 파일 경로가 있으면
        if not os.path.isfile(path=weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(f=weight)["model"]  # weight 경로에 저장된 모델 가중치 파일 로드
        model.load_state_dict(state)  # 저장된 모델 가중치를 model에 로드. 모델의 가중치 업데이트
        return model  # 가중치가 적용된 모델
    else:  # 가중치 파일이 지정되지 않은 경우에는 모델을 그대로 반환하고, 로그 메시지를 출력
        logger.info("No load model")
        return model


########################   Train & Valid
def run(
    model: nn.Module,
    train_data: dict,
    valid_data: dict = None,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay=1e-5,
    lr_decay: int = 10,
    gamma: float = 0.9,
    model_dir: str = None,
):
    model.train()

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)

    os.makedirs(name=model_dir, exist_ok=True)

    if valid_data is None:  # valid data 없으면 train data에서 일부를 랜덤하게 선택하여 사용
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids])

    logger.info(f"Training Started : n_epochs={n_epochs}")
    best_auc, best_epoch = 0, -1

    for e in range(n_epochs):
        logger.info("Epoch: %s", e)
        # TRAIN
        train_auc, train_acc, train_loss = train(
            train_data=train_data, model=model, optimizer=optimizer
        )

        # VALID
        auc, acc = validate(valid_data=valid_data, model=model)
        wandb.log(
            dict(
                train_loss_epoch=train_loss,
                train_acc_epoch=train_acc,
                train_auc_epoch=train_auc,
                valid_acc_epoch=acc,
                valid_auc_epoch=auc,
            )
        )

        if (
            auc > best_auc
        ):  # 현재 에폭에서의 AUC가 이전까지의 최고 AUC보다 큰 경우, 최적의 모델 가중치로 업데이트
            logger.info(
                "Best model updated AUC from %.4f to %.4f", best_auc, auc
            )
            best_auc, best_epoch = auc, e
            torch.save(
                obj={"model": model.state_dict(), "epoch": e + 1},
                f=os.path.join(model_dir, f"best_model.pt"),
            )
    torch.save(
        obj={"model": model.state_dict(), "epoch": e + 1},
        f=os.path.join(model_dir, f"last_model.pt"),
    )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def train(model: nn.Module, train_data: dict, optimizer: torch.optim.Optimizer):
    pred = model(train_data["edge"])
    loss = model.link_pred_loss(pred=pred, edge_label=train_data["label"])

    prob = model.predict_link(edge_index=train_data["edge"], prob=True)
    prob = prob.detach().cpu().numpy()

    label = train_data["label"].cpu().numpy()
    acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
    auc = roc_auc_score(y_true=label, y_score=prob)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.info(
        "TRAIN AUC : %.4f ACC : %.4f LOSS : %.4f", auc, acc, loss.item()
    )
    return auc, acc, loss


def validate(valid_data: dict, model: nn.Module):
    with torch.no_grad():
        prob = model.predict_link(edge_index=valid_data["edge"], prob=True)
        prob = prob.detach().cpu().numpy()

        label = valid_data["label"]
        acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
        auc = roc_auc_score(y_true=label, y_score=prob)
    logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc


def inference(model: nn.Module, data: dict, output_dir: str):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(edge_index=data["edge"], prob=True)

    logger.info("Saving Result ...")
    ######################## SAVE PREDICT
    print("\n--------------- Save Output Predict   ---------------")
    pred = pred.detach().cpu().numpy()
    os.makedirs(name=output_dir, exist_ok=True)
    write_path = os.path.join(output_dir, "lightgcn.csv")
    pd.DataFrame({"prediction": pred}).to_csv(
        path_or_buf=write_path, index_label="id"
    )
    logger.info("Successfully saved submission as %s", write_path)
