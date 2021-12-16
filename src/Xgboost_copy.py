import numpy as np
from scipy.sparse import data
from torch.nn.functional import threshold
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pickle
import os
import torch
import pandas
import copy

from sklearn.metrics import roc_curve

def train_model(model_dir="../xgbmodels"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
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

        fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=1)
        fnr = 1 - tpr
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        print(f"eer:{eer}, threshold:{eer_threshold}")
        save_model = {"model":model, "threshold":eer_threshold}
        pickle.dump(save_model, open(f'{model_dir}/{year}.model', 'wb'))


def load_csv(csv_file):
    data_dict = {"feats":[], "labels":[]}
    with open(csv_file) as fp:
        fp.readline()
        for id, data in enumerate(fp):
            content = data.strip().split(",")
            feature = content[:-1]
            data_dict["feats"].append( [float(feat) for feat in feature] )
            data_dict["labels"].append( int(content[-1]) )
    means = np.mean(data_dict["feats"], axis=0)
    var = np.var(data_dict["feats"], axis=0)
    data_dict["feats_mean"] = means
    data_dict["feats_var"] = var
    return data_dict

# class dmDataset(torch.utils.data.Dataset):
class dmDataset:
    def __init__(self, csv_file):
        data_dict = load_csv(csv_file)
        label = data_dict["labels"]
        feats = data_dict["feats"]
        mean = data_dict["feats_mean"]
        var = data_dict["feats_var"]

        self.size = len(label)
        self.feats = torch.tensor(feats).cuda()
        self.label = torch.tensor(label).cuda()
        self.feats_mean = torch.tensor(mean).cuda()
        self.feats_var = torch.tensor(var).cuda()

        print(self.feats.shape)
        print(self.label.shape)

    def __getitem__(self, index):
        # one_hot = self.label[index].new_zeros(2)
        # one_hot[int(self.label[index])] = 1
        res = {
            "id":index,
            "feats":self.feats[index],
            "label":self.label[index],
            "mean":self.feats_mean,
            "var":self.feats_var,
        }
        return res
    def __len__(self):
        return self.size

def jitter_by_value(data, idx, ratio=0.5):
    data_new = copy.deepcopy(data)
    data_new["feats"][idx] = data["feats"][idx] * ratio
    return data_new

def jitter_by_var(data, idx, ratio=1):
    data_new = copy.deepcopy(data)
    data_new["feats"][idx] = data["feats"][idx] + ratio * data["var"][idx]
    return data_new

def valid_feats(model, data_item, threshold):
    res = []
    for idx in range(len(data_item["feats"])):
        jvalue_data = jitter_by_value(data_item, idx=idx)
        jvar_data = jitter_by_var(data_item, idx=idx)
        preds = {}
        preds['ori'] = model.predict(xgb.DMatrix([data_item["feats"].cpu().numpy().tolist()])) > threshold
        preds['value'] = model.predict(xgb.DMatrix([jvalue_data["feats"].cpu().numpy().tolist()])) > threshold
        preds['var'] = model.predict(xgb.DMatrix([jvar_data["feats"].cpu().numpy().tolist()])) > threshold
        tuple_res = ( 
            data_item['id'], 
            idx,
            data_item['label'], 
            preds['ori'],
            preds['value'],
            preds['var'],
            )
        res.append(tuple_res)
    # df = pandas.DataFrame(res, columns=['id', 'jitter_idx', 'label', 'ori', 'jitter_value', 'jitter_var'])
    # print(df)
    return res

def test_model(model_dir="../xgbmodels"):
    for year in range(1,6):
        res = []
        load_model = pickle.load(open(f'{model_dir}/{year}.model', 'rb'))
        threshold = load_model["threshold"]
        model = load_model["model"]
        dataset = dmDataset(f'../data_split/test_{year}year.csv')
        for data_item in dataset:
            for result in valid_feats(model, data_item, threshold):
                res.append(result) 
        df = pandas.DataFrame(res, columns=['id', 'jitter_idx', 'label', 'ori', 'jitter_value', 'jitter_var'])
        df.to_csv(f"{model_dir}/jitter_res_{year}year.csv")


# train_model()
test_model()
        
        
