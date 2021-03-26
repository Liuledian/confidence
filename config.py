import logging


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
device_ids = [0, 1]
device = device_ids[0]
log_file = "/home/PublicDir/liuledian/log/confidence.log"
log_file_critical = "/home/PublicDir/liuledian/log/critical.log"
logger = get_logger()
