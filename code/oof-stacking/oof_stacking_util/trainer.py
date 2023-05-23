import gc
import os

import copy
import numpy as np
import pandas as pd

import torch
from tqdm import notebook

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import BaseCrossValidator

from oof_stacking_util.utils import (
    data_augmentation,
    get_optimizer,
    get_scheduler,
    get_metric,
    get_gradient,
    process_batch,
    compute_loss,
    validate,
)

from oof_stacking_util.datasets import DKTDataset
from oof_stacking_util.datloader import collate, get_loaders
from oof_stacking_util.models import LSTM, Bert, Saint, LastQuery, FixupEncoder


class UserBasedKFold(BaseCrossValidator):
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def _iter_test_indices(self, X, y=None, groups=None):
        n_users = X.userID.nunique()
        user_cnt = X.groupby(["userID"]).count()["assessmentItemID"].values
        user_cnt_tup_list = np.array(
            sorted(
                [(user, cnt) for user, cnt in enumerate(user_cnt)],
                key=lambda x: x[1],
            )
        )

        for i in range(self.n_splits):
            indices = (
                np.arange(n_users, step=self.n_splits) + i
            )  # sequence 수로 정렬된 리스트에서 가져올 index
            indices = indices[indices <= n_users - 1]  # 범위를 초과하는 부분 예외 처리
            userID_list = np.array(
                [user for user, _ in user_cnt_tup_list[indices]]
            )
            fold_indices = X.loc[X["userID"].isin(userID_list)].index.values

            yield fold_indices


############# Trainer
class Trainer:
    def __init__(self):
        pass

    def train(self, args, train_data, valid_data):
        """훈련을 마친 모델을 반환한다"""

        # args update
        self.args = args

        # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
        torch.cuda.empty_cache()
        gc.collect()

        # augmentation
        augmented_train_data = data_augmentation(train_data, args)
        if len(augmented_train_data) != len(train_data):
            print(
                f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n"
            )

        train_loader, valid_loader = get_loaders(
            args, augmented_train_data, valid_data
        )

        # only when using warmup scheduler
        args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (
            args.n_epochs
        )
        args.warmup_steps = args.total_steps // 10

        model = get_model(args)
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        best_auc = -1
        best_model = -1

        for epoch in notebook.tqdm(range(args.n_epochs)):
            ### TRAIN
            train_auc, train_acc = model_train(
                train_loader, model, optimizer, scheduler, args
            )

            ### VALID
            valid_auc, valid_acc, preds, targets = validate(
                valid_loader, model, args
            )

            ### TODO: model save or early stopping
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_model = copy.deepcopy(model)

            # scheduler
            if args.scheduler == "plateau":
                scheduler.step(best_auc)
            else:
                scheduler.step()

        return best_model

    def evaluate(self, args, model, valid_data):
        """훈련된 모델과 validation 데이터셋을 제공하면 predict 반환"""
        pin_memory = False

        valset = DKTDataset(valid_data, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

        auc, acc, preds, _ = validate(valid_loader, model, args)
        print(f"AUC : {auc}, ACC : {acc}")

        return preds

    def test(self, args, model, test_data):
        return self.evaluate(args, model, test_data)

    def get_target(self, datas):
        targets = []
        for data in datas:
            targets.append(data[-1][-1])

        return np.array(targets)


############# Stacking
class Stacking:
    def __init__(self, trainer):
        self.trainer = trainer

    def get_train_oof(self, args, data, fold_n=5, cv_info="ubfold"):
        """
        Args:
            - args: 모델의 config
            - data: train_data
        Returns:
            oof, fold_models
        """
        oof = np.zeros(data.shape[0])  # 고유한 user_id 수

        fold_models = []

        if cv_info == "skfold":
            kfold = StratifiedKFold(n_splits=fold_n)
        elif cv_info == "ubfold":
            kfold = UserBasedKFold(n_splits=fold_n)
        else:
            kfold = KFold(n_splits=fold_n)

        # 클래스 비율 고려하여 Fold별로 데이터 나눔
        target = self.trainer.get_target(data)
        for i, (train_index, valid_index) in enumerate(
            kfold.split(data, target)
        ):
            train_data, valid_data = data[train_index], data[valid_index]

            # 모델 생성 및 훈련
            print(f"Calculating train oof {i + 1}")
            trained_model = self.trainer.train(args, train_data, valid_data)

            # 모델 검증
            predict = self.trainer.evaluate(args, trained_model, valid_data)

            # fold별 oof 값 모으기
            oof[valid_index] = predict
            fold_models.append(trained_model)

        return oof, fold_models

    def get_test_avg(self, args, models, test_data):
        predicts = np.zeros(test_data.shape[0])

        # 클래스 비율 고려하여 Fold별로 데이터 나눔
        for i, model in enumerate(models):
            print(f"Calculating test avg {i + 1}")
            predict = self.trainer.test(args, model, test_data)

            # fold별 prediction 값 모으기
            predicts += predict

        # prediction들의 average 계산
        predict_avg = predicts / len(models)

        return predict_avg

    def train_oof_stacking(self, args_list, data, fold_n=5, cv_info="ubfold"):
        S_train = None  # OOF 예측 결과들을 모아놓은 DataFrame
        models_list = []
        for i, args in enumerate(args_list):
            print(f"training oof stacking model [ {i + 1}번: {args.model} ]")
            train_oof, models = self.get_train_oof(
                args, data, fold_n=fold_n, cv_info=cv_info
            )
            train_oof = train_oof.reshape(-1, 1)

            # oof stack!
            if not isinstance(S_train, np.ndarray):
                S_train = train_oof
            else:
                S_train = np.concatenate([S_train, train_oof], axis=1)

            # store fold models
            models_list.append(models)

        return models_list, S_train

    def test_avg_stacking(self, args_list, models_list, test_data):
        S_test = None
        for i, models in enumerate(models_list):
            print(f"test average stacking model [ {i + 1} ]")
            test_avg = self.get_test_avg(args_list[i], models, test_data)
            test_avg = test_avg.reshape(-1, 1)

            # avg stack!
            if not isinstance(S_test, np.ndarray):
                S_test = test_avg
            else:
                S_test = np.concatenate([S_test, test_avg], axis=1)

        return S_test

    def train(self, meta_model, args_list, data):
        models_list, S_train = self.train_oof_stacking(args_list, data)
        # 디렉토리가 존재하지 않을 경우 생성
        output_dir = "./stacked_output"  # train_oof_stacked 파일 저장 디렉토리
        output_file = "S_train.csv"
        output_path = os.path.join(output_dir, output_file)
        os.makedirs(output_dir, exist_ok=True)

        # S_train을 DataFrame으로 변환 후 저장
        S_train_df = pd.DataFrame(S_train)
        S_train_df.to_csv(output_path, index=False)

        # meta model train
        target = self.trainer.get_target(data)
        meta_model.fit(S_train, target)

        return meta_model, models_list, S_train, target

    def test(self, meta_model, args_list, models_list, test_data):
        S_test = self.test_avg_stacking(args_list, models_list, test_data)
        predict = meta_model.predict(S_test)

        return predict, S_test


def model_train(
    train_loader, model, optimizer, scheduler, args, gradient=False
):
    model.train()

    total_preds = []
    total_targets = []
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

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

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
    if args.model == "tfixup":
        model = FixupEncoder(args)

    model.to(args.device)

    return model
