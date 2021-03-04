from dataset import SubjectDependentDataset
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


def train_one_dataset(model, optimizer, tr_dataset, te_dataset, n_epochs, batch_size,
                      domain_adap=None,label_type=None, adj_type=None):
    pass


def initial_adjacency_matrix():



def train_one_epoch():
    pass


def evaluate():
    pass


if __name__ == '__main__':
    torch.manual_seed(seed_num)
    pass
