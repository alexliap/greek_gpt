import pickle

import torch
from torch.utils.data import DataLoader, SequentialSampler, StackDataset, TensorDataset


class CustomDataLoader:
    def __init__(self, data_path: str, batch_size: int, is_test: bool = False):
        self.data_path = data_path

        self.is_test = is_test

        self.dataset = self._make_stacked_dataset()

        self.batch_size = batch_size

    def get_dataloder(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(self.dataset),
            num_workers=10,
        )

    def _load_dataset_from_pkl(self):
        with open(self.data_path, "rb") as data:
            dataset = pickle.load(data)

        if self.is_test:
            dataset = dataset[:10_000]

        return dataset

    def _make_x_y(self):
        dataset = self._load_dataset_from_pkl()

        x = dataset[:-1]
        y = dataset[1:]

        return x, y

    def _make_stacked_dataset(self):
        x, y = self._make_x_y()

        tdt_x = TensorDataset(torch.tensor(x))
        tdt_y = TensorDataset(torch.tensor(y))

        stack = StackDataset(tdt_x, tdt_y)

        return stack
