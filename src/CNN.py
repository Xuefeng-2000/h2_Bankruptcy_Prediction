import numpy as np
import time
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset
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


BATCH_SIZE = 16

EPOCHS = 5  # 总共训练迭代的轮数

learning_rate = 0.01  # 设定初始的学习率

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



c_cnn = 0

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(1,1))  # 卷积 1 ->  10   5*5 kernel
        self.conv2 = nn.Conv2d(2, 4, kernel_size=(1,1))  # 10 -> 20   5*5
        self.MaxPool = nn.MaxPool2d(1,2)  # 池化层
        self.fc1 = nn.Linear(63, 256)  # 全链接
        self.re = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)  # 全链接
        self.fc3 = nn.Linear(128, 2)  # 全链接

    def forward(self, x):
        # batch_size = 64
        batch_size = x.size(0)
        x=x.view(batch_size,1,63,-1)
        #x = F.relu((self.conv1(x)))  # x: 64*10*12*12
        #x = F.relu((self.conv2(x)))  # x: 64*20*4*4
        x = x.view(batch_size, -1)
        #print(x)
        x = self.re(self.fc1(x))  # x: 64*10
        x = self.re(self.fc2(x))  # x: 64*10
        x = self.fc3(x)  # x: 64*10
        #x = self.fc3(x)  # x: 64*10
        #print(x)
        return F.log_softmax(x, dim=1)


'''
输入的维度：in_dim；
第一层神经网络的神经元个数n_hidden_1；
第二层神经网络神经元的个数n_hidden_2,out_dim
第三层网络(输出成)神经元的个数
'''


class simpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        hidden_1_out = self.layer1(x)
        hidden_2_out = self.layer2(hidden_1_out)
        out = self.layer3(hidden_2_out)
        return out


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





model = RNN()
if sys.argv[1] == 'CNN':
    print("========================CNN======================")
    model = CNN()#simpleNet(63,320,320,2)
else:
    print("========================RNN======================")
    model = RNN()

optimizer = optim.SGD(model.parameters(), lr=0.0001)

fig, ax = plt.subplots()
x = []
y = []
res = 0
point_cnt = 0



# 训练
def train(epoch,train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()

        data = data.view(data.size(0), -1)
        output = model(data)

        loss = nn.CrossEntropyLoss()(output, target)
        #pt_y = loss.tolist()
        # print(pt_y)
        # vision_loss(pt_y) #loss收敛可视化

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))




def test(ite,test_loader):
    sum_loss = 0
    correct = 0

    TP = [0 for i in range(10)]  # 正样本预测为正
    FP = [0 for i in range(10)]  # 正样本预测为负
    FN = [0 for i in range(10)]  # 负样本预测为正
    Precision = [0.0 for i in range(10)]  # 精确度
    Recall = [0.0 for i in range(10)]  # 精确度
    Num_tar = [0 for i in range(10)]  # 全部样本

    y_true = []
    y_score = []
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
            y_true.append(tar)
            y_score.append(pre)
            #if tar == 1:
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

    sum_loss /= len(test_loader.dataset)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        sum_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print("=================================================\n")

    right_num = 0
    total = 0

    len_y = len(y_true)
    for i in range(len_y):
        if (y_true[i] == y_score[i]):
            right_num = right_num + 1
        y_true[i] = int(y_true[i])
        y_score[i] = int(y_score[i])
    # print(y_true)

    total = len_y
    print(f"{year}year:")
    print("acc:", right_num / total)
    print("auc:", roc_auc_score(y_true=y_true, y_score=y_score))
    print()


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
            if(label == 1):
                for kk in range(9):
                    X.append(feature)
                    y.append(label)

            X.append(feature)
            y.append(label)
            cnt = cnt +1
            #if( cnt == 2):
            #    break

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
    print("***" + str(len(test_x)))
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
    test(1,test_loader)

    print("%s Time cost : %10d s" % (sys.argv[1], time_sum))


    print()