import torch
import numpy as np
from torch.utils.data import Dataset

class TemperatureDataset(Dataset):
    """
    Loads images from a .npy file shaped [N, 1, D, D].
    """
    def __init__(self, month):
        super().__init__()
        npy_path = "/home/tea1rng/workspace_gen/baselines/dsbm-pytorch/bridge/data/months_temperature/month_" + month + ".npy"
        self.data = np.load(npy_path, mmap_mode="r")  # lazy loading

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]   # shape: (1, D, D)
        tensor_img = torch.tensor(img, dtype=torch.float32)
        return tensor_img, torch.zeros((1,))





