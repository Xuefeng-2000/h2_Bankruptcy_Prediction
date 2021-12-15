import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import sys

BATCH_SIZE = 1

EPOCHS = 5  # 总共训练迭代的轮数

learning_rate = 0.1  # 设定初始的学习率


c_cnn = 0

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 卷积 1 ->  10   5*5 kernel
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 10 -> 20   5*5
        self.MaxPool = nn.MaxPool2d(2)  # 池化层
        self.fc1 = nn.Linear(63, 640)  # 全链接
        self.fc2 = nn.Linear(640, 2)  # 全链接
        self.fc3 = nn.Linear(320, 2)  # 全链接

    def forward(self, x):
        # batch_size = 64
        batch_size = x.size(0)
        #x = F.relu(self.MaxPool(self.conv1(x)))  # x: 64*10*12*12
        #x = F.relu(self.MaxPool(self.conv2(x)))  # x: 64*20*4*4
        x = x.view(-1, 63)  # x: 64*320

        x = self.fc1(x)  # x: 64*10
        x = self.fc2(x)  # x: 64*10
        #x = self.fc3(x)  # x: 64*10
        #print(x)
        return F.log_softmax(x, dim=1)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=63,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 2)  # 输出层

    def forward(self, x):
        x = x.view(-1, 1, 63)
        # print(x.shape)
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return F.log_softmax(out, dim=1)


class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)



model = RNN()
if sys.argv[1] == 'CNN':
    print("========================CNN======================")
    model = CNN()
else:
    print("========================RNN======================")
    model = RNN()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

fig, ax = plt.subplots()
x = []
y = []
res = 0
point_cnt = 0


# loss 可视化
def vision_loss(pt_y):
    global point_cnt
    global ax
    global x, y
    x.append(point_cnt)
    point_cnt = point_cnt + 1
    y.append(pt_y)
    ax.cla()
    ax.plot(x, y, 'r', lw=1)
    plt.pause(0.0001)


# 训练
def train(epoch,train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        # 优化器内部参数梯度置为0
        optimizer.zero_grad()
        # 前向结果
        output = model.forward(data)
        # 计算损失函数

        loss = F.cross_entropy(output, target)
        pt_y = loss.tolist()
        # print(pt_y)
        # vision_loss(pt_y) #loss收敛可视化

        # 反向传播
        loss.backward()
        # 更新模型
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# F1计算三种方法
def Macro_average(Precision, Recall):
    lenp = len(Precision)
    sump = 0.0
    sumr = 0.0
    for i in range(lenp):
        sump = sump + Precision[i]
        sumr = sumr + Recall[i]
    sump = sump / lenp
    sumr = sumr / lenp
    F1 = 2 * sump * sumr / (sump + sumr)
    print("\nF1 Macro_average :%.6f" % (F1))


def Weighted_average(Precision, Recall, Num_tar):
    sum_sample = 0
    for i in range(10):
        sum_sample = sum_sample + Num_tar[i]
    sump = 0.0
    sumr = 0.0
    for i in range(10):
        sump += Precision[i] * Num_tar[i] / sum_sample
        sumr += Recall[i] * Num_tar[i] / sum_sample
    F1 = 2 * sump * sumr / (sump + sumr)
    print("F1 Weighted_average :%.6f" % (F1))


def Micro_average(TP, FP, FN):
    p1 = p2 = p3 = p4 = 0
    for i in range(10):
        p1 = p1 + TP[i]
        p2 = p2 + TP[i] + FP[i]
        p3 = p3 + TP[i]
        p4 = p4 + TP[i] + FN[i]
    sump = 1.0 * p1 / p2
    sumr = 1.0 * p3 / p4
    F1 = 2 * sump * sumr / (sump + sumr)
    print("F1 Micro_average :%.6f" % (F1))


def test(ite,test_loader):
    sum_loss = 0
    correct = 0
    TP = [0 for i in range(10)]  # 正样本预测为正
    FP = [0 for i in range(10)]  # 正样本预测为负
    FN = [0 for i in range(10)]  # 负样本预测为正
    Precision = [0.0 for i in range(10)]  # 精确度
    Recall = [0.0 for i in range(10)]  # 精确度
    Num_tar = [0 for i in range(10)]  # 全部样本

    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        # print(target)
        # exit()
        output = model(data)
        # 求loss和
        sum_loss += F.nll_loss(output, target, reduction='sum').item()
        # 取最大概率结果
        pred = output.data.max(1, keepdim=True)[1]
        pred = pred.reshape(-1)
        targ_l = target.tolist()
        pred_l = pred.tolist()
        len_t = len(targ_l)
        for i in range(len_t):
            pre = int(pred_l[i])
            tar = int(targ_l[i])
            #if pre == 1:
                #print("************")
                #print(pre,tar)
            # if tar == 5:
            # print(pre, tar)
            Num_tar[tar] = Num_tar[tar] + 1
            if pre == tar:  # 预测正确 记录正例子
                TP[pre] = TP[pre] + 1
            else:
                FP[pre] = FP[pre] + 1
                FN[tar] = FN[tar] + 1

        correct += pred.eq(target).cpu().sum()

    for i in range(10):
        Precision[i] = 0 if (TP[i] + FP[i]) == 0 else 1.0 * TP[i] / (TP[i] + FP[i])
        Recall[i] = 0 if (TP[i] + FN[i]) == 0 else 1.0 * TP[i] / (TP[i] + FN[i])

    print("==============Epoch:%2d Test Result==============" % (ite))
    Macro_average(Precision, Recall)
    Weighted_average(Precision, Recall, Num_tar)
    Micro_average(TP, FP, FN)

    sum_loss /= len(test_loader.dataset)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        sum_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print("=================================================\n")





for year in range(1, 2):
    print(year )
    enroll = f'../data_split/enroll_{year}year.csv'
    test_f = f'../data_split/test_{year}year.csv'

    X = []
    y = []



    with open(enroll) as lines:

        cnt = 0
        for id, data in enumerate(lines):
            if id == 0:
                continue

            temp = data.strip().split(",")
            feature = temp[:-1]
            length = len(feature)

            for i in range(length):
                feature[i] = float(feature[i])
            label = int(temp[-1])

            X.append(feature)
            y.append(label)
            cnt = cnt +1
            #if( cnt == 10):
             #   break

        # 加载训练集
        data_tensor = torch.Tensor(X)
        #print(data_tensor)
        #print(y)
        target_tensor = torch.tensor(y,dtype = torch.long)
        #print(target_tensor)

        dataset = TensorDataset(data_tensor,target_tensor)
        train_loader = torch.utils.data.DataLoader(
            dataset,batch_size=BATCH_SIZE, shuffle=True)  # 指明批量大小，打乱，这是处于后续训练的需要。

    right_num = 0
    total = 0


    test_x = []
    test_y = []

    with open(test_f) as lines:
        for id, data in enumerate(lines):
            if id == 0:
                continue

            temp = data.strip().split(",")
            feature = temp[:-1]
            length = len(feature)

            for i in range(length):
                feature[i] = float(feature[i])
            label = int(temp[-1])

            test_x.append(feature)
            test_y.append(label)

    data_tensor = torch.Tensor(test_x)
    target_tensor = torch.tensor(test_y,dtype = torch.long)

    datasets = TensorDataset(data_tensor, target_tensor)
    test_loader = torch.utils.data.DataLoader(
        datasets,
        batch_size=BATCH_SIZE, shuffle=True)

    time_sum = 0
    for epoch in range(1, 2):
        st = time.time()
        train(epoch,train_loader)
        ed = time.time()
        time_sum += int(ed) - int(st)
        test(epoch,test_loader)

    print("%s Time cost : %10d s" % (sys.argv[1], time_sum))



    print()