import numpy as np
import torch
import os
import sys
from scipy.stats import norm
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from config import *


class SubjectDependentDataset(InMemoryDataset):
    def __init__(self, task, subject_name, fold, phase="train", transform=None, pre_transform=None):
        self.task = task
        self.subject_name = subject_name
        self.fold = fold
        self.phase = phase
        super(SubjectDependentDataset, self).__init__(
            root_template.format(task=task, subject=subject_name, fold=fold, phase=phase),
            transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data = np.load(DE_data_template.format(task=self.task, subject=self.subject_name,
                                               fold=self.fold, phase=self.phase))
        label = np.load(DE_label_template.format(task=self.task, subject=self.subject_name,
                                                 fold=self.fold, phase=self.phase))
        data_list = []
        m, n = data.shape
        assert m == label.shape[0]
        for i in tqdm(range(m)):
            x = np.reshape(data[i, :], [n_channels, n_bands])
            y = label[i]
            edge_index = [[i, j] for i in range(n_channels) for j in range(n_channels) if i != j]
            edge_index = torch.tensor(edge_index, dtype=torch.long).reshape((2, -1))
            assert edge_index.shape[1] == n_channels * (n_channels - 1)
            graph_data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(graph_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    dataset = SubjectDependentDataset(task="animal", subject_name="diaoyuqi", fold=0, phase="train")
    print(dataset)
