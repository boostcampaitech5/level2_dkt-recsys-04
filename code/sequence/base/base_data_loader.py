import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from torch.nn.utils.rnn import pad_sequence

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


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        batch_size,
        shuffle: bool,
        validation_split_size: float,
        num_workers: int,
        collate_fn=collate,
    ):
        self.validation_split_size = validation_split_size
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split_size)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }

        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def get_loader(self):
        return DataLoader(sampler=self.sampler, **self.init_kwargs)
