from dataset import SubjectDependentDataset
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from model import SymSimGCNNet
from config import *
from scipy.stats import norm
import numpy as np
import torch

soft_label_table = None


def distribution_label(labels, std=1):
    global soft_label_table
    if soft_label_table is None:
        soft_label_table = np.zeros([5, 5])
        for y in range(5):
            x = np.linspace(0, 4, 5)
            p = norm.pdf(x, y, std)
            soft_label_table[y] = np.exp(p) / np.sum(np.exp(p))
        print("generate soft_label_table")

    t = torch.Tensor(soft_label_table)
    return t[labels.long(), :]


def train_RGNN(tr_dataset, te_dataset, n_epochs, batch_size, lr, z_dim, K, dropout, adj_type, learn_edge, lambda1,
               lambda2, domain_adaptation, lambda_dat, label_type, ckpt_save_name, ckpt_load=None):
    # parameter sanity check
    if label_type not in label_types:
        raise Exception("undefined label_type")
    if adj_type not in adj_types:
        raise Exception("undefined adj_type")

    # construct model
    edge_weight = initial_adjacency_matrix(adj_type)
    model = SymSimGCNNet(n_channels, learn_edge, edge_weight, n_bands, [z_dim], n_classes[label_type],
                         K, dropout, domain_adaptation)
    last_epoch = 0
    if ckpt_load is not None:
        ckpt = torch.load(ckpt_load)
        last_epoch = ckpt_load["epoch"]
        if last_epoch >= n_epochs:
            raise Exception("loaded model have trained >= n_epochs")
        state_dict = ckpt_load["state_dict"]
        model.load_state_dict(state_dict)
    # use multiple GPU
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    print(model)

    # prepare dataloader and optimizer
    logger.info("tr_dataset: {}".format(tr_dataset))
    logger.info("te_dataset: {}".format(te_dataset))
    logger.info("training start from epoch {}".format(last_epoch))
    tr_loader = DataLoader(tr_dataset, batch_size, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda2)
    for ep in range(last_epoch + 1, n_epochs + 1):
        model.train()
        loss_all = 0
        reverse_scale = 2 / (1 + np.exp(-10 * ep / n_epochs)) - 1
        for tr_data in tr_loader:
            tr_data = tr_data.to(device)
            output, domain_output = model(tr_data, reverse_scale)
            # print(label_type, "output shape", output.shape, "data.y shape", tr_data.y.shape)
            # classification loss
            if label_type == "hard":
                loss = F.cross_entropy(output, tr_data.y)
            elif label_type == "soft":
                loss = - distribution_label(tr_data.y) * F.log_softmax(output, dim=1)
                loss = torch.mean(torch.sum(loss, dim=1))
            else:
                loss = F.mse_loss(output, tr_data.y - 2)
            # l1 regularization loss
            if learn_edge:
                loss += lambda1 * torch.sum(torch.abs(model.edge_weight))
            # domain adaptation loss
            if domain_adaptation:
                # tr_data.x: [num_graph * n_channels, feature_dim]
                n_nodes = len(tr_data.x)
                loss += lambda_dat * F.cross_entropy(domain_output, torch.zeros(n_nodes))
                te_indices = torch.randint(0, len(te_dataset), tr_data.num_graphs)
                te_data = te_dataset[te_indices]
                _, te_domain_output = model(te_data, reverse_scale)
                loss += lambda_dat * F.cross_entropy(te_domain_output, torch.ones(n_nodes))

            loss_all += loss.item() * tr_data.num_graphs
            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # evaluate the model
        eval_acc = evaluate_RGNN(model, te_dataset, label_type)
        logger.info("epoch: {:>4}; loss: {:<10}; eval acc {:<10}".format(ep, loss_all/len(tr_dataset), eval_acc))

    # save model checkpoint
    logger.info(model.edge_weight)
    checkpoint = {"epoch": n_epochs, "state_dict": model.state_dict()}
    torch.save(checkpoint, ckpt_dir + '/' + ckpt_save_name)
    return eval_acc


def evaluate_RGNN(model, te_dataset, label_type):
    assert label_type in label_types
    model.eval()
    n_correct = 0
    te_loader = DataLoader(te_dataset)
    with torch.no_grad():
        for te_data in te_loader:
            te_data = te_data.to(device)
            output, _ = model(te_data)
            if label_type == "hard" or label_type == "soft":
                pred = torch.argmax(output, dim=1)
                n_correct += torch.sum(pred == te_data.y).item()
            else:
                sep = torch.Tensor([-2, -1, 0, 1, 2])
                sep = sep.repeat(len(te_data.y), 1)
                diff = torch.abs(sep.t() - te_data.y)
                pred = torch.argmin(diff, dim=0)
                n_correct += torch.sum(pred == te_data.y).item()

    return n_correct / len(te_dataset)


def initial_adjacency_matrix(adj_type):
    if adj_type == "uniform":
        adj = np.zeros([n_channels, n_channels])
        xs, ys = np.tril_indices(n_channels, -1)
        adj[xs, ys] = np.random.uniform(0, 1, xs.shape[0])
        adj = adj + adj.T + np.identity(len(adj))
        return torch.tensor(adj, dtype=torch.float)


def train_RGNN_for_all():
    torch.manual_seed(seed_num)
    n_epochs = 1000
    batch_size = 16
    K = 2
    dropout = 0.7
    lr = 0.001
    z_dim = 5
    adj_type = "uniform"
    learn_edge = True
    lambda1 = 0.001
    lambda2 = 0.001
    lambda_dat = 0.001
    label_type = "hard"
    domain_adaptation = None
    for task in tasks:
        eval_acc_all = []
        for subject_name in subject_name_list:
            for fold in range(n_folds):
                ckpt_save_name = "{}_{}_{}_{}.ckpt".format(task, subject_name, fold, n_epochs)
                ckpt_load = None
                tr_dataset = SubjectDependentDataset(task, subject_name, fold, "train")
                te_dataset = SubjectDependentDataset(task, subject_name, fold, "test")
                eval_acc = train_RGNN(tr_dataset, te_dataset, n_epochs, batch_size, lr, z_dim, K, dropout, adj_type, learn_edge,
                           lambda1, lambda2, domain_adaptation, lambda_dat, label_type, ckpt_save_name, ckpt_load)
                eval_acc_all.append(eval_acc_all)
        eval_acc_all = np.array(eval_acc_all)
        eval_acc_mean = np.mean(eval_acc_all)
        eval_acc_std = np.std(eval_acc_all)
        logger.critical("task: {:>10}; acc_mean: {:<10}; acc_std {:<10}".format(task, eval_acc_mean, eval_acc_std))


if __name__ == '__main__':
    train_RGNN_for_all()

