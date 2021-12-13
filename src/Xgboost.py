import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

for year in range(1,6):
    enroll  = f'../data_split/enroll_{year}year.csv'
    test = f'../data_split/test_{year}year.csv'

    X = []
    y = []

    with open(enroll) as lines:
        for id,data in enumerate(lines):
            if id == 0:
                continue

            temp = data.strip().split(",")
            feature = temp[:-1]
            length = len(feature)

            for i in range(length):
                feature[i] = float(feature[i])
            label = temp[-1]

            X.append(feature)
            y.append(label)

    dtrain = xgb.DMatrix(np.array(X), np.array(y))

    num_round = 2000
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',  # 多分类的问题
        # 'num_class': 2,               # 类别数，与 multisoftmax 并用
        'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 12,               # 构建树的深度，越大越容易过拟合
        'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,              # 随机采样训练样本
        'colsample_bytree': 0.7,       # 生成树时进行的列采样
        'min_child_weight': 3,
        'verbosity':1,
        'eta': 0.007,                  # 如同学习率
        'seed': 1000,
        'nthread': 4,                  # cpu 线程数
        'eval_metric':'auc'
    }
    plst = params.items()

    model = xgb.train(list(params.items()), dtrain, num_round)

    right_num = 0
    total = 0

    y_true = [] # 真实标签
    y_score = [] # 预测得分
    with open(test) as lines:
        for id,data in enumerate(lines):
            if id == 0:
                continue

            temp = data.strip().split(",")
            feature = temp[:-1]
            length = len(feature)

            for i in range(length):
                feature[i] = float(feature[i])
            label = temp[-1]
            y_true.append(int(label))
            
            dtest = xgb.DMatrix([feature])
            y_pred = model.predict(dtest)

            y_score.append(y_pred)

            if round(y_pred[0]) == int(label):
                right_num += 1
            total += 1

    print(f"{year}year:")        
    print("acc:",right_num / total)
    print("auc:",roc_auc_score(y_true = y_true, y_score=y_score))
    print()