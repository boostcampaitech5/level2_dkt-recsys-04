import torch

from torch.nn.utils.rnn import pad_sequence
from oof_stacking_util.datasets import DKTDataset


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            col_list[i].append(col)

    # 각 column의 값들을 대상으로 padding 진행
    # pad_sequence([[1, 2, 3], [3, 4]]) -> [[1, 2, 3],
    #                                       [3, 4, 0]]
    for i, col_batch in enumerate(col_list):
        col_list[i] = pad_sequence(col_batch, batch_first=True)

    # mask의 경우 max_seq_len을 기준으로 길이가 설정되어있다.
    # 만약 다른 column들의 seq_len이 max_seq_len보다 작다면
    # 이 길이에 맞추어 mask의 길이도 조절해준다
    col_seq_len = col_list[0].size(1)
    mask_seq_len = col_list[-1].size(1)
    if col_seq_len < mask_seq_len:
        col_list[-1] = col_list[-1][:, :col_seq_len]

    return tuple(col_list)


def get_loaders(args, train, valid):
    pin_memory = False

    trainset = DKTDataset(train, args)
    valset = DKTDataset(valid, args)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=pin_memory,
        collate_fn=collate,
    )

    valid_loader = torch.utils.data.DataLoader(
        valset,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=pin_memory,
        collate_fn=collate,
    )

    return train_loader, valid_loader
