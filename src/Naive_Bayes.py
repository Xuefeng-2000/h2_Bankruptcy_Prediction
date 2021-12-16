from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
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

    model = MultinomialNB(alpha=0.01)
    model.fit(X, y)

    right_num = 0
    total = 0

    y_true = []  # 真实标签
    y_score = []  # 预测得分
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
            y_score.append(model.predict_proba([feature])[0][1])

            if ans[0] == label:
                right_num += 1
            total += 1

    print(f"{year}year:")
    print("acc:", right_num / total)
    print("auc:", roc_auc_score(y_true=y_true, y_score=y_score))
    print()