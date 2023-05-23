import os
from typing import Tuple

import pandas as pd
import torch

from lightgcn_utils.utils import get_logger, logging_conf


logger = get_logger(logging_conf)


def prepare_dataset(
    device: str, data_dir: str, verbose=True
) -> Tuple[dict, dict, int]:
    data = load_data(data_dir=data_dir)
    train_data, test_data = split_data(data=data)
    id2index: dict = indexing_data(data=data)
    train_data_proc = process_data(
        data=train_data, id2index=id2index, device=device
    )
    test_data_proc = process_data(
        data=test_data, id2index=id2index, device=device
    )

    if verbose:
        print_data_stat(train_data, "Train")
        print_data_stat(test_data, "Test")

    return train_data_proc, test_data_proc, len(id2index)


########################   Data load
print("--------------- LightGCN Load Data ---------------")


def load_data(data_dir: str) -> pd.DataFrame:
    path1 = os.path.join(data_dir, "train_data.csv")
    path2 = os.path.join(data_dir, "test_data.csv")
    data1 = pd.read_csv(path1)  # train data
    data2 = pd.read_csv(path2)  # test data

    data = pd.concat(
        [data1, data2]
    )  # train 데이터 셋에 test 데이터 셋의 유저가 포함돼있지 않기 때문에 모든 데이터를 통합하여 사용(GCN 특성상 처음 본 node는 예측할 수 없기 때문)
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )  # 중복 레코드 제거(Node간의 link는 하나만 생성될 수 있으므로 최종 경우를 제외하고 제거)
    return data


########################   Data split
print("--------------- LightGCN Data Split ---------------")


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame]:
    train_data = data[data.answerCode >= 0]  # loss는 answerCode가 -1이 아닌 값만
    test_data = data[
        data.answerCode == -1
    ]  # answerCode가 -1인 항목은 최종 평가시 사용되는 항목
    return train_data, test_data


########################   Data indexing
def indexing_data(data: pd.DataFrame) -> dict:  # user/item ID를 node index와 매빙
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user = len(userid)

    userid2index = {v: i for i, v in enumerate(userid)}
    itemid2index = {v: i + n_user for i, v in enumerate(itemid)}
    id2index = dict(userid2index, **itemid2index)
    return id2index


########################   Data processing
def process_data(
    data: pd.DataFrame, id2index: dict, device: str
) -> dict:  # user/item을 node로 바꾸어 edge와 label을 구함
    edge, label = [], []
    for user, item, acode in zip(
        data.userID, data.assessmentItemID, data.answerCode
    ):
        uid, iid = id2index[user], id2index[item]
        edge.append([uid, iid])
        label.append(acode)  # 사용자가 해당 문항을 맞췄는지 여부를 label에 저장

    edge = torch.LongTensor(edge).T  # 문제를 푼 경우 user-item간 edge가 존재
    label = torch.LongTensor(label)  # 문제를 맞췄을 때는 1, 틀렸을 때는 0
    return dict(edge=edge.to(device), label=label.to(device))


########################   Preprocessed data info
def print_data_stat(data: pd.DataFrame, name: str) -> None:
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
