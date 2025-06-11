from torch.utils.data import Dataset, TensorDataset


class MyCustomDataset(Dataset):
    def __init__(self, data: TensorDataset, targets: TensorDataset, block_size: int):
        self.data = data
        self.targets = targets
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        return self.data[idx : idx + self.block_size], self.targets[
            idx : idx + self.block_size
        ]
