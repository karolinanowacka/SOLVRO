import torch
from torch.utils.data import Dataset
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, data, labels, scaler=None):
        self.data = data
        self.labels = labels
        self.scaler = scaler

        if self.scaler is not None:
            data_reshaped = self.data.reshape(self.data.shape[0], -1)
            self.data = self.scaler.transform(data_reshaped).reshape(self.data.shape)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)