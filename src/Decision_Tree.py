
from sklearn.metrics import roc_auc_score
import sklearn

def gini_score(data_sets, classes):
    instances_num = sum([len(data_set) for data_set in data_sets])
    gini_score = 0.0
    for data_set in data_sets:
        data_set_num = len(data_set)
        if data_set_num == 0:
            continue
        gini_index = 1.0
        for c in classes:
            c_num = 0
            for i in data_set:
                if c == i[-1]:
                    c_num += 1
            c_p = c_num / data_set_num  # data_set_num == 0?
            gini_index = gini_index - c_p * c_p
        data_set_gini_score = gini_index * data_set_num / instances_num
        gini_score += data_set_gini_score
    return gini_score



def devision_data_set(index, devision_value, data_set):
    less = list()
    big = list()
    for row in data_set:
        if row[index] < devision_value:
            less.append(row)
        else:
            big.append(row)

    return less, big


import math


def get_devision(data_set):
    best_index = 0
    best_devision_value = 0
    best_gini = math.inf
    len_g = len(data_set)
    best_data_sets = None
    classes = list(set([row[-1] for row in data_set]))
    for i in range(len(data_set[0]) - 1):
        sort_list  = []
        for k in range(len_g):
            sort_list.append(data_set[k][i])
        sort_list = sorted(sort_list)
        tem_devision_values = []
        for k in range(0,len_g,20):
            tem_devision_values.append(sort_list[k])
        for devision_value in tem_devision_values:
            data_sets = devision_data_set(i, devision_value, data_set)
            gini = gini_score(data_sets, classes)

            if gini < best_gini:
                best_gini = gini
                best_index = i
                best_devision_value = devision_value
                best_data_sets = data_sets
    return {
        'index': best_index,
        'devision_value': best_devision_value,
        'data_sets': best_data_sets
    }

def leaves(data_set):
    outcomes = [row[-1] for row in data_set]
    return max(set(outcomes), key=outcomes.count)


def devision(node, max_depth, min_size, depth):
    left, right = node['data_sets']
    del (node['data_sets'])
    ##
    if not left or not right:
        node['left'] = node['right'] = leaves(left + right)
        return
    ##
    if depth >= max_depth:
        node['left'] = leaves(left)
        node['right'] = leaves(right)
        return

    if len(left) < min_size:
        node['left'] = leaves(left)
    else:
        node['left'] = get_devision(left)
        devision(node['left'], max_depth, min_size, depth + 1)

    if len(right) < min_size:
        node['right'] = leaves(right)
    else:
        node['right'] = get_devision(right)
        devision(node['right'], max_depth, min_size, depth + 1)


def build_tree(train_data_set, max_depth, min_size):
    root = get_devision(train_data_set)
    devision(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if isinstance(node, dict):
        index = node['index']
        devision_value = node['devision_value']
        if row[index] < devision_value:
            return predict(node['left'], row)
        else:
            return predict(node['right'], row)
    else:
        return node

def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    print('test', len(test))
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    #print(predictions)
    return predictions


for year in range(1, 6):
    enroll = f'../data_devision/enroll_{year}year.csv'
    test = f'../data_devision/test_{year}year.csv'

    train_data = []
    test_data = []

    y_true = []  # 真实标签
    y_score = []  # 预测得分

    cnt = 0
    with open(enroll) as lines:
        for id, data in enumerate(lines):
            if id == 0:
                continue

            temp = data.strip().devision(",")
            train_data.append(temp)
            cnt = cnt + 1
            #if(cnt == 1000):
             #   break

    print("=============Read finished!============")
    with open(test) as lines:
        for id, data in enumerate(lines):
            if id == 0:
                continue

            temp = data.strip().devision(",")
            test_data.append(temp)
            y_true.append(temp[-1])


    y_score = decision_tree(train_data,test_data,9,10)


    right_num = 0
    total = 0

    len_y = len(y_true)
    for i in range(len_y):
        if(y_true[i] == y_score[i]):
            right_num = right_num + 1
        y_true[i] = int(y_true[i])
        y_score[i] = int(y_score[i])
    #print(y_true)

    total = len_y
    print(f"{year}year:")
    print("acc:", right_num / total)
    print("auc:", roc_auc_score(y_true=y_true, y_score=y_score))
    print("f1: ", sklearn.metrics.f1_score(y_true, y_score))
    print()