import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MultivariateGaussianDataset(Dataset):

    def __init__(self, num_samples, means=[0, 0],
                 cov=[[1, 0], [0, 1]], seed=None, fisher_set=False):
        self.num_samples = num_samples
        self.dimensions = len(means)
        self.means = means
        self.cov = cov
        np.random.seed(seed)
        rng = np.random.default_rng()
        self.seed = seed
        self.fisher_set = fisher_set

        self.data = rng.multivariate_normal(means, cov, size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # NNGeometry expects data in this form
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        if self.fisher_set:
            # This unpacks the tensor to a tuple which NNGeometry likes
            return (data_tensor, )
        else:
            return data_tensor
