import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

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

    clf2 = RandomForestClassifier(random_state=0, n_estimators=500)
    # clf3 = LogisticRegression(random_state=1)
    # clf4 = GaussianNB()
    clf = VotingClassifier(estimators=[
        # ('gbdt',clf1),
        ('rf',clf2),
        # ('lr',clf3),
        # ('nb',clf4),
        # ('xgboost',clf5),
        ],
        voting='soft')
    clf.fit(X,y)

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

            ans = clf.predict([feature])
            print(clf.score([feature]))
            # y_score.append(clf.decision_function([feature]))

            if ans[0] == label:
                right_num += 1
            total += 1
            
    print(f"{year}year:")        
    print("acc:",right_num / total)
    print("auc:",roc_auc_score(y_true = y_true, y_score=y_score))
    print()