import os
import time
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


############# seed
def set_seeds(seed: int = 42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 로거 객체 생성
def get_logger(logger_conf: dict):
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}


def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = args.stride

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1

            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(
                        col[window_i * stride : window_i * stride + window_size]
                    )

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))

    return augmented_datas


def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas


def data_augmentation(data, args):
    if args.window:
        data = slidding_window(data, args)

    return data


def get_metric(targets, preds):
    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))

    return auc, acc


def get_optimizer(model, args):
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    if args.optimizer == "adamW":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()

    return optimizer


def get_scheduler(optimizer, args):
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, mode="max", verbose=True
        )
    elif args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )
    return scheduler


def get_gradient(model):
    gradient = []

    for name, param in model.named_parameters():
        grad = param.grad
        if grad != None:
            gradient.append(grad.cpu().numpy().astype(np.float16))
            # gradient.append(grad.clone().detach())
        else:
            gradient.append(None)

    return gradient


def get_criterion(pred, target):
    loss = nn.BCELoss(reduction="none")
    return loss(pred, target)


# 배치 전처리
def process_batch(batch, args):
    test, question, tag, correct, mask = batch

    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다
    interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction[:, 0] = 0  # set padding index to the first sequence
    interaction = (interaction * mask).to(torch.int64)

    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1

    # device memory로 이동
    test = test.to(args.device)
    question = question.to(args.device)

    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    return (test, question, tag, correct, mask, interaction, gather_index)


# loss계산하고 parameter update!
def compute_loss(preds, targets, index):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)
        index    : (batch_size, max_seq_len)

        만약 전체 sequence 길이가 max_seq_len보다 작다면 해당 길이로 진행
    """
    loss = get_criterion(preds, targets)
    loss = torch.gather(loss, 1, index)
    loss = torch.mean(loss)

    return loss


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)

        preds = model(input)
        targets = input[3]  # correct
        index = input[-1]  # gather index

        # predictions
        preds = preds.gather(1, index).view(-1)
        targets = targets.gather(1, index).view(-1)

        if args.device == "cuda":
            preds = preds.to("cpu").detach().numpy()
            targets = targets.to("cpu").detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    return auc, acc, total_preds, total_targets


def get_pred(test_loader, model, args):
    model.eval()
    total_preds = []

    for step, batch in enumerate(test_loader):
        input = process_batch(batch, args)
        preds = model(input)
        index = input[-1]

        # predictions
        preds = preds.gather(1, index).view(-1)
        if args.device == "cuda":
            preds = preds.to("cpu").detach().numpy()
        else:
            preds = preds.detach().numpy()

        total_preds.append(preds)
    total_preds = np.concatenate(total_preds)

    return total_preds


def make_dir(path: str) -> str:
    """
    [description]
    경로가 존재하지 않을 경우 해당 경로를 생성하며, 존재할 경우 pass를 하는 함수입니다.

    [arguments]
    path : 경로

    [return]
    path : 경로
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass
    return path


def save_predict(dir_path: str, filename: str, predict: pd.DataFrame) -> bool:
    """_summary_
    예측값을 파일에 작성하기
    Args:
        filename (str): FileName
        predict (pd.DataFrame): Predic 결과

    Returns:
        bool: Save 결과
    """
    dir_path = make_dir(dir_path)
    filename = os.path.join(dir_path, filename)
    with open(filename, "w", encoding="utf8") as w:
        print("writing prediction : {}".format(filename))
        w.write("id,prediction\n")
        for id, p in enumerate(predict):
            w.write("{},{}\n".format(id, p))
    return True


def get_submit_filename(
    auc_score: float,
    format_name: str = "csv",
) -> str:
    """
    [description]
    submit file을 저장할 경로를 반환하는 함수입니다.

    [arguments]
    args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

    [return]
    filename : submit file을 저장할 경로를 반환합니다.
    이 때, 파일명은 submit/날짜_시간_모델명.csv 입니다.
    """
    now = time.localtime()
    now_date = time.strftime("%Y%m%d", now)
    now_hour = time.strftime("%X", now)
    save_time = now_date + "_" + now_hour.replace(":", "")

    filename = f"{save_time}_{auc_score:.5f}_oof_stacking.{format_name}"

    return filename
