from dataset import SubjectDependentDataset
import torch.nn.functional as F
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from model import SymSimGCNNet
from config import *
from scipy.stats import norm
import sklearn
import numpy as np
import math
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
    # log hyper-parameter
    logger.critical('batch_size {}, lr {}, z_dim {}, K {}, dropout {}, adj_type {}, learn_edge {}, lambda1, {}'
                    'lambda2 {}, domain_adaptation {}, lambda_dat {}, label_type {}'
                    .format(batch_size, lr, z_dim, K, dropout, adj_type, learn_edge, lambda1,
                            lambda2, domain_adaptation, lambda_dat, label_type))

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
    model = DataParallel(model, device_ids=device_ids).to(device)
    logger.info(model)

    # prepare dataloader
    logger.info("tr_dataset: {}".format(tr_dataset))
    logger.info("te_dataset: {}".format(te_dataset))
    logger.info("training start from epoch {}".format(last_epoch))
    tr_loader = DataListLoader(tr_dataset, batch_size, True)

    # prepare optimizer
    param_list1 = []
    param_list2 = []
    for name, param in model.named_parameters():
        if name in ['module.edge_weight', 'module.conv1.lin.bias', 'module.fc.bias']:
            param_list1.append(param)
        else:
            param_list2.append(param)
    optimizer = torch.optim.Adam([
        {'params': param_list1, 'weight_decay': 0},
        {'params': param_list2, 'weight_decay': lambda2}
    ], lr=lr)
    # iterate over all epochs
    eval_acc_list = []
    macro_f1_list = []
    for ep in range(last_epoch + 1, n_epochs + 1):
        model.train()
        loss_all = 0
        reverse_scale = 2 / (1 + math.exp(-10 * ep / n_epochs)) - 1
        if domain_adaptation == 'RevGrad':
            model.module.alpha = reverse_scale

        # iterate over all graphs
        for tr_data_list in tr_loader:
            # output shape (len(tr_data_list), 5 or 1)
            output, domain_output = model(tr_data_list)
            # classification loss
            # y shape (len(tr_data_list), )
            y = torch.cat([data.y for data in tr_data_list]).to(output.device)
            if label_type == "hard":
                loss = F.cross_entropy(output, y)
            elif label_type == "soft":
                loss = - distribution_label(y) * F.log_softmax(output, dim=1)
                loss = torch.mean(torch.sum(loss, dim=1))
            else:
                loss = F.mse_loss(output, y - 2)
            # l1 regularization loss
            if learn_edge:
                loss += lambda1 * torch.sum(torch.abs(model.module.edge_weight))
            # domain adaptation loss
            if domain_adaptation:
                # tr_data.x: [num_graph * n_channels, feature_dim]
                n_nodes = domain_output.size(0)
                loss += lambda_dat * F.cross_entropy(domain_output, torch.zeros(n_nodes).cuda())
                te_indices = torch.randint(0, len(te_dataset), len(tr_data_list))
                te_data = te_dataset[te_indices]
                _, te_domain_output = model(te_data)
                loss += lambda_dat * F.cross_entropy(te_domain_output, torch.ones(n_nodes).cuda())

            loss_all += loss.item() * len(tr_data_list)
            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # evaluate the model
        accuracy, macro_f1_score, micro_f1_score = evaluate_RGNN(model, te_dataset, label_type)
        eval_acc_list.append(accuracy)
        macro_f1_list.append(macro_f1_score)
        train_acc, _, _ = evaluate_RGNN(model, tr_dataset, label_type)
        logger.info('epoch: {:4d}; loss: {:9.5f}; train acc: {:9.5f}; eval acc: {:9.5f}; '
                    'macro f1: {:9.5f}; micro f1: {:9.5f}'
                    .format(ep, loss_all/len(tr_dataset), train_acc, accuracy, macro_f1_score, micro_f1_score))

    # save model checkpoint
    logger.info(list(model.parameters()))
    logger.info(format_list(model.module.edge_weight.detach().cpu().numpy().flatten()))
    checkpoint = {"epoch": n_epochs, "state_dict": model.state_dict()}
    torch.save(checkpoint, ckpt_dir + '/' + ckpt_save_name)
    return eval_acc_list, macro_f1_list


def evaluate_RGNN(model, te_dataset, label_type):
    assert label_type in label_types
    model.eval()
    y_pred = []
    y_true = []
    te_loader = DataListLoader(te_dataset)
    with torch.no_grad():
        for te_data_list in te_loader:
            # output shape (len(te_data_list), 5 or 1)
            output, _ = model(te_data_list)
            y = torch.cat([data.y for data in te_data_list]).to(output.device)
            y_true.extend(y.detach().cpu().numpy())
            if label_type == "hard" or label_type == "soft":
                pred = torch.argmax(output, dim=1)
                y_pred.extend(pred.detach().cpu().numpy())
            else:
                sep = torch.Tensor([-2, -1, 0, 1, 2])
                sep = sep.repeat(len(te_data_list), 1)
                diff = torch.abs(sep.t() - y)
                pred = torch.argmin(diff, dim=0)
                y_pred.extend(pred.detach().cpu().numpy())

    macro_f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    micro_f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    return accuracy, macro_f1_score, micro_f1_score


def initial_adjacency_matrix(adj_type):
    if adj_type == "uniform":
        adj = np.zeros([n_channels, n_channels])
        xs, ys = np.tril_indices(n_channels, -1)
        adj[xs, ys] = np.random.uniform(0, 1, xs.shape[0])
        adj = adj + adj.T + np.identity(len(adj))
        return torch.tensor(adj, dtype=torch.float)


def format_list(ls):
    n_epochs = len(ls)
    result = "\n"
    for i in range(n_epochs):
        result += '({:4d},{:9.6f}) '.format(i+1, ls[i])
        if (i + 1) % 100 == 0:
            result += '\n'
    return result


def train_RGNN_for_all(dropout):
    torch.manual_seed(seed_num)
    n_epochs = 500
    batch_size = 32
    K = 2
    z_dim = 5
    #dropout = 0.7
    label_type = "hard"
    adj_type = "uniform"
    learn_edge = True
    domain_adaptation = None
    lr = 0.001
    lambda1 = 0.001
    lambda2 = 0.001
    lambda_dat = 0.001
    # Choose which tasks to run
    for task in tasks[0:1]:
        acc_all_sub = []
        f1_all_sub = []
        # Choose which subjects to run
        for subject_name in subject_name_list[0:1]:
            avg_acc_per_ep = np.zeros(n_epochs)
            avg_f1_per_ep = np.zeros(n_epochs)
            for fold in range(n_folds):
                ckpt_save_name = "{}_{}_{}_{}.ckpt".format(task, subject_name, fold, n_epochs)
                ckpt_load = None
                tr_dataset = SubjectDependentDataset(task, subject_name, fold, "train")
                te_dataset = SubjectDependentDataset(task, subject_name, fold, "test")
                acc_list, f1_list = train_RGNN(tr_dataset, te_dataset, n_epochs, batch_size, lr, z_dim, K, dropout,
                                               adj_type, learn_edge, lambda1, lambda2, domain_adaptation, lambda_dat,
                                               label_type, ckpt_save_name, ckpt_load)
                logger.critical('-' * 100)
                logger.critical("task: {:>10}; subject: {:>15}; fold: {}".format(task, subject_name, fold))
                logger.critical(format_list(acc_list))
                logger.critical(format_list(f1_list))
                avg_acc_per_ep += np.array(acc_list)
                avg_f1_per_ep += np.array(f1_list)

            # log average metrics over all 5 folds
            avg_acc_per_ep /= n_folds
            avg_f1_per_ep /= n_folds
            max_avg_acc_idx = np.argmax(avg_acc_per_ep)
            max_avg_f1_idx = np.argmax(avg_f1_per_ep)
            logger.critical('=' * 100)
            logger.critical("task: {:>10}; subject: {:>15}; max acc: ({:4d}, {:9.5f}); max f1: ({:4d}, {:9.5f})"
                            .format(task, subject_name, max_avg_acc_idx + 1, avg_acc_per_ep[max_avg_acc_idx],
                                    max_avg_f1_idx + 1, avg_f1_per_ep[max_avg_f1_idx]))
            logger.critical(format_list(avg_acc_per_ep))
            logger.critical(format_list(avg_f1_per_ep))
            # record max avg metric of a subject
            acc_all_sub.append(avg_acc_per_ep[max_avg_acc_idx])
            f1_all_sub.append(avg_f1_per_ep[max_avg_f1_idx])

        # compute metrics over all subjects
        acc_all_sub = np.array(acc_all_sub)
        acc_mean = np.mean(acc_all_sub)
        acc_std = np.std(acc_all_sub)
        f1_all_sub = np.array(f1_all_sub)
        f1_mean = np.mean(f1_all_sub)
        f1_std = np.std(f1_all_sub)
        # log average metrics over all subjects
        logger.critical('*' * 100)
        logger.critical("task: {:>10}; acc mean: {:9.5f}; acc_std {:9.5f}; f1 mean: {:9.5f}; f1 std: {:9.5f}"
                        .format(task, acc_mean, acc_std, f1_mean, f1_std))
        logger.critical(list(zip(acc_all_sub, f1_all_sub)))


if __name__ == '__main__':
    for dropout in [0.8, 0.9, 1.0]:
        train_RGNN_for_all(dropout)

