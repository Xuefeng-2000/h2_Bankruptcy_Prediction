
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_iris
from sklearn import tree
import sklearn
import pydotplus
import os
from matplotlib import pyplot as plt
for year in range(1, 6):
    enroll = f'../data_split/enroll_{year}year.csv'
    test = f'../data_split/test_{year}year.csv'

    X = []
    y = []

    with open(enroll) as lines:
        for id, data in enumerate(lines):
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

    model = tree.DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=16)
    model.fit(X, y)

    ''' 决策树
    tree.plot_tree(model)
    fig = plt.figure(figsize=(100, 100))
    feature_names = [i for i in range(63)]
    target_names = ['Norupt','Rupt']
    _ = tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=target_names,
        filled=True
    )
    
    fig.savefig("../decTree-png/decistion_tree"+str(year)+".png")
    '''

    right_num = 0
    total = 0

    y_true = []  # 真实标签
    y_score = []  # 预测得分
    y_pred = []
    with open(test) as lines:
        for id, data in enumerate(lines):
            if id == 0:
                continue

            temp = data.strip().split(",")
            feature = temp[:-1]
            length = len(feature)

            for i in range(length):
                feature[i] = float(feature[i])
            label = temp[-1]

            y_true.append(int(label))

            ans = model.predict([feature])
            y_pred.append(int(ans[0]))
            y_score.append(model.predict_proba([feature])[0][1])

            if ans[0] == label:
                right_num += 1
            total += 1

    #print(y_pred)

    #print(f"{year}year:")
    #print("%.2lf%%"%(100.0*round(right_num / total ,4)))
    print("%.4lf"%( round(roc_auc_score(y_true=y_true, y_score=y_score),4)))
    #print("f1: ", sklearn.metrics.f1_score(y_true, y_pred))
    #print()