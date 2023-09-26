import pandas as pd
import torch
import numpy as np 
from torch.utils.data import Dataset
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import os


class CSVDataset(Dataset): #src: https://github.com/BiggyBing/CausalTGAN/blob/main/dataset.py
    """CSV Dataset"""

    def __init__(self, csv_file, reorder=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            reorder: a dictionary {idx_in_graph: feature_name} which key is the idx in the causal graph and the feature name is the value
        """
        self.data = pd.read_csv(csv_file)
        # self.data = np.expand_dims(pd.read_csv(csv_file),2) #!!! added np.expand_dims to resolve a mat_mul dimensionality mismatch later in the code
        # keep_col = self.data.keys().tolist()
        # keep_col = ['Unnamed' not in item for item in keep_col]
        # self.data = self.data.iloc[:, keep_col]

        # if reorder is not None:
        #     cols = [reorder[i] for i in range(len(reorder))]
        #     self.data = self.data[cols]
        # # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return self.data[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data.iloc[idx, :]
        data = np.array(data)
        data = torch.Tensor(data)
        print(data.shape)
        return data

# ds = CSVDataset(csv_file='datasets/sachs_data.csv')

# def load_data(file_path, batch_size=1000):

#     # load dataset as numpy array

#     X = np.expand_dims(
#         np.loadtxt(
#             os.path.expanduser(file_path),
#             skiprows=1,
#             delimiter=",",
#             dtype=np.int32,
#         ),
#         2,
#     )
#     print(X.shape)
#     feat = torch.FloatTensor(X)

#     # reconstruct itself
#     data = TensorDataset(feat)

#     return data


