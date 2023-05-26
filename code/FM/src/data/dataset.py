import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset


def process_context_data(train, test):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """
    train_count_df = pd.DataFrame(
        train[["userID", "assessmentItemID"]]
        .groupby(["userID", "assessmentItemID"])
        .value_counts()
    )
    test_count_df = pd.DataFrame(
        test[["userID", "assessmentItemID"]]
        .groupby(["userID", "assessmentItemID"])
        .value_counts()
    )

    train = pd.merge(
        left=train, right=train_count_df, on=["userID", "assessmentItemID"]
    )
    test = pd.merge(left=test, right=test_count_df, on=["userID", "assessmentItemID"])

    train = train.drop_duplicates(subset=["userID", "assessmentItemID"], keep="last")
    test = test.drop_duplicates(subset=["userID", "assessmentItemID"], keep="last")
    tmp = test[test["answerCode"] == -1].copy()
    test = test[test["answerCode"] >= 0].copy()

    train.drop(["Timestamp"], axis=1, inplace=True)
    test.drop(["Timestamp"], axis=1, inplace=True)
    tmp.drop(["Timestamp"], axis=1, inplace=True)

    train_df = pd.concat([train, test])
    test_df = tmp.copy()

    context_df = pd.concat([train_df, test_df])

    itemid2idx = {v: i for i, v in enumerate(context_df.assessmentItemID.unique())}
    testid2idx = {v: i for i, v in enumerate(context_df.testId.unique())}
    tag2idx = {v: i for i, v in enumerate(context_df.KnowledgeTag.unique())}

    train_df["assessmentItemID"] = train_df["assessmentItemID"].map(itemid2idx)
    train_df["testId"] = train_df["testId"].map(testid2idx)
    train_df["KnowledgeTag"] = train_df["KnowledgeTag"].map(tag2idx)

    test_df["assessmentItemID"] = test_df["assessmentItemID"].map(itemid2idx)
    test_df["testId"] = test_df["testId"].map(testid2idx)
    test_df["KnowledgeTag"] = test_df["KnowledgeTag"].map(tag2idx)

    field_dims = (
        context_df.drop(["answerCode"], axis=1).agg(lambda x: x.nunique()).values
    )
    # field_dims = []
    # for col in context_df.drop(['answerCode'], axis=1).columns:
    #     field_dims.append(context_df[col].nunique())

    field_dims = np.array(field_dims)

    return train_df, test_df, field_dims


def context_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    train = pd.read_csv(args.data_path + "train_data.csv")
    test = pd.read_csv(args.data_path + "test_data.csv")

    train_df, test_df, field_dims = process_context_data(train, test)

    data = {
        "train": train_df,
        "test": test_df,
        "field_dims": field_dims,
    }

    return data


def context_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
        data["train"].drop(["answerCode"], axis=1),
        data["train"]["answerCode"],
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )
    data["X_train"], data["X_valid"], data["y_train"], data["y_valid"] = (
        X_train,
        X_valid,
        y_train,
        y_valid,
    )

    return data


def context_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = TensorDataset(
        torch.LongTensor(data["X_train"].values),
        torch.LongTensor(data["y_train"].values),
    )
    valid_dataset = TensorDataset(
        torch.LongTensor(data["X_valid"].values),
        torch.LongTensor(data["y_valid"].values),
    )
    test_dataset = TensorDataset(torch.LongTensor(data["test"].values))

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    data["train_dataloader"], data["valid_dataloader"], data["test_dataloader"] = (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
    )

    return data
