from dataset import SubjectDependentDataset
from torch_geometric.data import DataLoader
from model import SymSimGCNNet
from config import *
from scipy.stats import norm
import numpy as np
import torch


def distribution_label(y, std=1):
    x = np.linspace(0, 4, 5)
    p = norm.pdf(x, y, std)
    p = np.exp(p) / np.sum(np.exp(p))
    return p


def train_RGNN(tr_dataset, te_dataset, n_epochs, batch_size, lr, z_dim, K, dropout, adj_type, learn_edge, lambda1,
               domain_adaptation, lambda2, label_type, model_ckpt=None):
    if label_type not in label_types:
        raise Exception("undefined label_type")
    if adj_type not in adj_types:
        raise Exception("undefined adj_type")

    edge_weight = initial_adjacency_matrix(adj_type)
    model = SymSimGCNNet(n_channels, learn_edge, edge_weight, n_bands, [z_dim], n_classes[label_type],
                         K, dropout, domain_adaptation)
    epoch = 0
    if model_ckpt is not None:
        ckpt = torch.load(model_ckpt)
        epoch = model_ckpt["epoch"]
        if epoch >= n_epochs:
            raise Exception("loaded model have trained >= n_epochs")
        state_dict = model_ckpt["state_dict"]
        model.load_state_dict(state_dict)
    model.to(device)

    logger.info("tr_dataset: {}".format(tr_dataset))
    logger.info("te_dataset: {}".format(te_dataset))
    logger.info("training start from epoch {}".format(epoch))
    tr_loader = DataLoader(tr_dataset, batch_size, True)
    te_loader = DataLoader(te_dataset, batch_size, True)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    for ep in range(epoch + 1, n_epochs + 1):
        model.train()
        for data in tr_loader:
            pass




def initial_adjacency_matrix(adj_type):
    if adj_type == "uniform":
        adj = np.zeros([n_channels, n_channels])
        xs, ys = np.tril_indices(n_channels, -1)
        adj[xs, ys] = np.random.uniform(0, 1, xs.shape[0])
        adj = adj + adj.T
        return adj


def evaluate():
    pass


if __name__ == '__main__':
    torch.manual_seed(seed_num)
    pass
