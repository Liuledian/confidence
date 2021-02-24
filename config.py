import os
# m = os.listdir(r"D:\data_preprocessed_200hz_filter\data")
# print(m)

subject_names = ['diaoyuqi', 'fengjialu', 'guoqianyu', 'huangzhongyu', 'jiangting',
                 'liaoruosong', 'lijie', 'liqinchao', 'litian', 'liuwen', 'luotian',
                 'luyewangqing', 'quyuqi', 'songkunjie', 'wangjian', 'wangyajie', 'wangzhiqi',
                 'wuxin', 'yudalong', 'zhouxiaozhen']
data_file_template = "D:/results/{task}/fea_smoothed/feature/corr/{threshold:.2f}/{fea}/{subject}_{session}.mat"
label_file_template = "D:/data_preprocessed_200hz_filter/label/{subject}/{session}/{task}/label.mat"
n_sessions = 2
n_bands = 5