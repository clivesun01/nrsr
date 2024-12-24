import torch
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(self, data):
        X_train_norm = torch.tensor(data['X_train_norm'], dtype=torch.float32)
        self.features = X_train_norm
        y = (data['y_train'] - min(data['y_train'])) / (max(data['y_train']) - min(data['y_train']))
        self.labels = torch.tensor(y, dtype=torch.float32)
        self.var_mask = list(data['var_mask'])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_input_dim(self):
        return self.features.shape[-1]

    def get_var_mask(self):
        return self.var_mask
