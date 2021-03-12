import logging
import torch


def get_logger():
    l = logging.getLogger()
    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    l.addHandler(fh)
    l.addHandler(sh)
    l.setLevel(logging.NOTSET)
    return l


subject_name_list = ['wuxin', 'diaoyuqi', 'zhouxiaozhen', 'liqinchao', 'litian', 'fengjialu', 'luotian',
                     'liaoruosong', 'wangjian', 'lijie', 'wangyajie', 'songkunjie', 'huangzhongyu', 'wangzhiqi']
DE_dir_template = "/home/PublicDir/liuledian/data/DE/{task}/DE_feature/{subject}/{fold}"
DE_data_template = DE_dir_template + "/{phase}_data.npy"
DE_label_template = DE_dir_template + "/{phase}_label.npy"
root_template = "/home/PublicDir/liuledian/data/graph_sub_dep/{task}/{subject}/{fold}/{phase}"
ckpt_dir = "/home/PublicDir/liuledian/ckpt"
n_channels = 62
n_bands = 5
n_folds = 5
n_classes = {'hard': 5, 'soft': 5, 'numeric': 1}
label_types = ['hard', 'soft', 'numeric']
adj_types = ['uniform', 'RGNN', 'corr']
tasks = ['animal', 'planeship']
seed_num = 0
device = torch.device('cuda')
log_file = "/home/PublicDir/liuledian/log/confidence.log"
logger = get_logger()
