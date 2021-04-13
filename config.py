import logging
import sys
import argparse


def get_logger():
    l = logging.getLogger()
    l.setLevel(logging.NOTSET)
    fh = logging.FileHandler(log_file)
    fh_cri = logging.FileHandler(log_file_critical)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    fh_cri.setFormatter(formatter)
    sh.setFormatter(formatter)
    fh.setLevel(logging.NOTSET)
    sh.setLevel(logging.NOTSET)
    fh_cri.setLevel(logging.CRITICAL)
    l.addHandler(fh_cri)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


def get_args():
    parser = argparse.ArgumentParser(description='Confidence')
    parser.add_argument('--task', type=str, default='animal')
    parser.add_argument('--subject', type=str, default='wuxin')
    parser.add_argument('--proc', type=int, required=True)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--z_dim', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--label_type', type=str, default='hard')
    parser.add_argument('--adj_type', type=str, default='uniform')
    parser.add_argument('--domain_adaptation', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l1', type=float, default=0.0001)
    parser.add_argument('--l2', type=float, default=0.0001)
    parser.add_argument('--l_dat', type=float, default=0.0001)

    return parser.parse_args()


args = get_args()
process_id = args.proc
tasks = ['animal', 'planeship']
subject_name_list = ['wuxin', 'diaoyuqi', 'zhouxiaozhen', 'liqinchao', 'litian', 'fengjialu', 'luotian',
                     'liaoruosong', 'wangjian', 'lijie', 'wangyajie', 'songkunjie', 'huangzhongyu', 'wangzhiqi']
DE_dir_template = "/home/PublicDir/liuledian/data/DE/{task}/DE_feature/{subject}/{fold}"
DE_data_template = DE_dir_template + "/{phase}_data.npy"
DE_label_template = DE_dir_template + "/{phase}_label.npy"
root_template = "/home/PublicDir/liuledian/data/graph_sub_dep/{task}/{subject}/{fold}/{phase}"
n_channels = 62
n_bands = 5
n_folds = 5
n_classes = {'hard': 5, 'soft': 5, 'numeric': 1}
label_types = ['hard', 'soft', 'numeric']
adj_types = ['uniform', 'RGNN', 'corr']
seed_num = 0
device_ids = [process_id]
device = device_ids[0]
ckpt_dir = "/home/PublicDir/liuledian/ckpt_{}".format(process_id)
log_file = "/home/PublicDir/liuledian/log/confidence_{}.log".format(process_id)
log_file_critical = "/home/PublicDir/liuledian/log/critical_{}.log".format(process_id)
logger = get_logger()
