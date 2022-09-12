import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class EGGDataset(Dataset):
    def __init__(self, mat_path):
        self.mat_path = mat_path
        self.mat = loadmat(mat_path)

        self.label_value = self.mat["label"]
        self.channel_value = self.mat["channel_labels"]
        self.time_value = self.mat["time_points"]
        self.data_value = self.mat["data"]

        assert self.label_value.shape[0] == self.data_value.shape[2]

    def __len__(self):
        return self.label_value.shape[0]

    def __getitem__(self, idx):
        data = self.data_value[:, :, idx]
        data = torch.from_numpy(data).float()

        label = self.label_value[idx, 0]
        label = torch.tensor(label)
        
        return data, label
