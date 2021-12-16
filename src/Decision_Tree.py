
from sklearn.metrics import roc_auc_score

def gini_score(groups, classes):
    '''
    row = [col1, col2, col3, col4, class]
    group: [row, row, ..., row]
    groups: [group, group, ..., group]

    classes: [0, 1]
    '''
    # weight = sum(group) / sum(sum(group))
    # Gini index = sum(sum(one_class) / sum(group))
    instances_num = sum([len(group) for group in groups])
    gini_score = 0.0
    for group in groups:
        group_num = len(group)
        if group_num == 0:
            continue
        gini_index = 1.0
        for c in classes:
            c_num = 0
            for i in group:
                if c == i[-1]:
                    c_num += 1
            c_p = c_num / group_num  # group_num == 0?
            gini_index = gini_index - c_p * c_p
        group_gini_score = gini_index * group_num / instances_num
        gini_score += group_gini_score
    return gini_score


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


def split_group(index, split_value, group):
    less = list()
    big = list()
    for row in group:
        if row[index] < split_value:
            less.append(row)
        else:
            big.append(row)

    return less, big


import math


# 找到最佳分裂特征点，并分裂
def get_split(group):
    best_index = 0
    best_split_value = 0
    best_gini = math.inf
    len_g = len(group)
    best_groups = None
    classes = list(set([row[-1] for row in group]))
    for i in range(len(group[0]) - 1):  # minus 1, because the last value is label
        sort_list  = []
        for k in range(len_g):
            sort_list.append(group[k][i])
        sort_list = sorted(sort_list)
        tem_split_values = []#list(set([row[i] for row in group]))
        for k in range(0,len_g,5):
            tem_split_values.append(sort_list[k])
        for split_value in tem_split_values:
            groups = split_group(i, split_value, group)
            gini = gini_score(groups, classes)

            # print('X%d < %.3f Gini=%.3f' % ((i + 1), split_value, gini))
            if gini < best_gini:
                best_gini = gini
                best_index = i
                best_split_value = split_value
                best_groups = groups
    return {
        'index': best_index,
        'split_value': best_split_value,
        'groups': best_groups
    }

# Create a terminal node value, find the majority class and use it as the label or value.
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count) # 返回出现次数最多的值


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    ##
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    ##
    if depth >= max_depth:
        node['left'] = to_terminal(left)
        node['right'] = to_terminal(right)
        return

    if len(left) < min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)

    if len(right) < min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train_group, max_depth, min_size):
    root = get_split(train_group)
    split(root, max_depth, min_size, 1)
    return root


# Prediction
def predict(node, row):
    if isinstance(node, dict):
        index = node['index']
        split_value = node['split_value']
        if row[index] < split_value:
            return predict(node['left'], row)
        else:
            return predict(node['right'], row)
    else:
        return node


# Classification and Regression Tree Algorithm
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
    enroll = f'../data_split/enroll_{year}year.csv'
    test = f'../data_split/test_{year}year.csv'

    train_data = []
    test_data = []

    y_true = []  # 真实标签
    y_score = []  # 预测得分

    cnt = 0
    with open(enroll) as lines:
        for id, data in enumerate(lines):
            if id == 0:
                continue

            temp = data.strip().split(",")
            train_data.append(temp)
            cnt = cnt + 1
            #if(cnt == 1000):
             #   break

    print("=============Read finished!============")
    with open(test) as lines:
        for id, data in enumerate(lines):
            if id == 0:
                continue

            temp = data.strip().split(",")
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
    print()