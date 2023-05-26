import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from sequence_utils.models import LSTM
from sequence_utils.models import Bert
from sequence_utils.models import Saint

# from mission_utils.models import FixupEncoder
from sequence_utils.models import LastQuery
from sequence_utils.models import LSTMATNN
from sequence_utils.models import SaintPlus

import numpy as np


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
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode="max", verbose=True)
    elif args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps
        )
    return scheduler


def get_criterion(pred, target):
    loss = nn.BCELoss(reduction="none")
    return loss(pred, target)


def get_metric(targets, preds):
    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))

    return auc, acc


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == "lstm":
        model = LSTM(args)
    if args.model == "bert":
        model = Bert(args)
    if args.model == "last_query":
        model = LastQuery(args)
    if args.model == "saint":
        model = Saint(args)
    if args.model == "lstm_attn":
        model = LSTMATNN(args)
    if args.model == "saint_plus":
        model = SaintPlus(args)
    # if args.model == "tfixup":
    #     model = FixupEncoder(args)

    model.to(args.device)

    return model


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


# 배치 전처리
def process_batch(batch, args):
    # test, question, tag, correct, elapsed_question, elapsed_test, mask = batch
    test, question, tag, correct, elapsed_question, mask = batch

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
    elapsed_question = ((elapsed_question + 1) * mask).to(torch.int64)
    # elapsed_test = ((elapsed_test + 1) * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1

    # device memory로 이동
    test = test.to(args.device)
    question = question.to(args.device)

    tag = tag.to(args.device)
    correct = correct.to(args.device)

    elapsed_question = elapsed_question.to(args.device)
    # elapsed_test = elapsed_test.to(args.device)

    correct = correct.to(args.device)

    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    # return (test, question, tag, correct, elapsed_question, elapsed_test, mask, interaction, gather_index)
    return (test, question, tag, correct, elapsed_question, mask, interaction, gather_index)


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


def train(train_loader, model, optimizer, scheduler, args, gradient=False):
    model.train()

    total_preds = []
    total_targets = []
    total_losses = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)

        preds = model(input)
        targets = input[3]  # correct
        index = input[-1]  # gather index

        loss = compute_loss(preds, targets, index)
        loss.backward()

        # save gradient distribution
        if gradient:
            args.n_iteration += 1
            args.gradient[f"iteration_{args.n_iteration}"] = get_gradient(model)

        # grad clip
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()
        optimizer.zero_grad()

        # warmup scheduler
        if args.scheduler == "linear_warmup":
            scheduler.step()

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
        total_losses.append(loss.cpu().detach().numpy())

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)
    total_losses = np.mean(total_losses)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    return total_losses, auc, acc


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
