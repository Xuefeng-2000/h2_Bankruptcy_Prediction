from scipy.io import arff
import functools
from io import StringIO
import csv
import pandas as pd
import random
import numpy as np


def csv_to_xls(csv_path, xls_path):
    with open(csv_path, 'r', encoding='gb18030', errors='ignore') as f:
        data = f.read()
    data_file = StringIO(data)
    print(data_file)
    csv_reader = csv.reader(data_file)
    list_csv = []
    for row in csv_reader:
        list_csv.append(row)
    df_csv = pd.DataFrame(list_csv).applymap(str)
    writer = pd.ExcelWriter(xls_path)
    # 写入Excel
    df_csv.to_excel(
        excel_writer=writer,
        index=False,
        header=False
    )
    writer.save()


def cmp1(n1, n2):
    if(n1[1] > n2[1]):
        return -1
    elif(n1[1] < n2[1]):
        return 1
    return 0

def profiling_miss(data):
    len_d = len(data)
    miss_num_attr = [0 for i in range(attr_len+1)]#第i号属性有多少空
    miss_num_item = [0 for i in range(attr_len+1)]#空了i个属性的有多少条目
    data_profiling = [[0.0 for i in range(64)] for j in range(2)]
    data_attrNum = [[0 for i in range(64)] for j in range(2)]

    sum_haveMiss = 0

    num_0 = 0
    num_1 = 0

    cnt =0
    for i in range(len_d):
        data_tmp = list(data[i])
        #print(data_tmp)
        len_item = len(data_tmp)
        data_tmp[-1] = int(data_tmp[-1])

        classid = data_tmp[-1]
        num_0 = num_0+1 if classid == 0 else num_0+0
        num_1 = num_1+1 if classid == 1 else num_1+0
        cnt_miss = 0
        for j in range(len_item-1):
            if (np.isnan(data_tmp[j])):
                cnt_miss = cnt_miss+1
                miss_num_attr[j] = miss_num_attr[j] + 1
            else:
                data_profiling[classid][j] = data_profiling[classid][j] + data_tmp[j]
                data_attrNum[classid][j] = data_attrNum[classid][j] + 1
        miss_num_item[cnt_miss] = miss_num_item[cnt_miss] + 1
        if(cnt_miss != 0):
            sum_haveMiss = sum_haveMiss + 1
        #cnt = cnt + 1
        #if(cnt == 5):
        #    break
    print("====Before pre : ====")
    print("num_0 : %d"%(num_0) )
    print("num_1 : %d"%(num_1) )
    print("precent_0 : %.2f" % (100.0*num_0/(num_0 + num_1)))
    print("precent_1 : %.2f" % (100.0*num_1/(num_0 + num_1) ))

    sort_miss1 = []
    idx = 0
    for a in miss_num_item:
        if idx == 0:
            idx = idx+1
            continue
        sort_miss1.append([idx,a])
        idx = idx + 1
    sort_miss1 = sorted(sort_miss1, key=functools.cmp_to_key(cmp1))
    print(sort_miss1) #缺i个的

    sort_miss2 = []
    idx = 0
    for a in miss_num_attr:
        sort_miss2.append([idx, a])
        idx = idx + 1
    sort_miss2 = sorted(sort_miss2, key=functools.cmp_to_key(cmp1))
    print(sort_miss2)#缺i号的

    #print(miss_num_item)
    bigger_than4 = 0
    for i in range(3,64): #缺3条以上的
        bigger_than4 = bigger_than4 + sort_miss1[i][1]
    print("There are %d (%.2f%%) items having missing data , %.2f%% Missing 3 attr at least !"%(sum_haveMiss,100*sum_haveMiss/len_d,100*bigger_than4/len_d))
    print(bigger_than4)
    cnt = 0
    data_pro = []

    num_0 = 0
    num_1 = 0
    for i in range(len_d): #数据填充

        data_tmp = []
        # print(data_tmp)
        len_item = len(data[i])

        classid = int(data[i][-1])
        sum_nan = 0
        for j in range(len_item - 1):
            if (np.isnan(data[i][j])):
                sum_nan = sum_nan + 1
        if(sum_nan >=4):
            continue
        num_0 = num_0 + 1 if classid == 0 else num_0 + 0
        num_1 = num_1 + 1 if classid == 1 else num_1 + 0
        for j in range(len_item-1):
            if j == 36: #删除36号元素
                continue
            if (np.isnan(data[i][j])): #无值填入平均
                rd = random.random()
                rd = rd % 10
                rd = rd - 10
                kk = 100 - rd
                data_tmp.append( kk/100*(data_profiling[0][j]+data_profiling[0][j])/(data_attrNum[0][j] + data_attrNum[1][j]))
            else: #有值填入原数值
                data_tmp.append(data[i][j])
        data_tmp.append(classid)

        data_pro.append(tuple(data_tmp))
    print("====After pre : ====")
    print("num_0 : %d" % (num_0))
    print("num_1 : %d" % (num_1))
    print("precent_0 : %.2f" % (100.0 * num_0 / (num_0 + num_1)))
    print("precent_1 : %.2f" % (100.0 * num_1 / (num_0 + num_1)))
    print("sum : %d"%(num_0+num_1))

    return data_pro


if __name__ == '__main__':
    for file_id in range(1,6):

        attr_len = 64

        c_path = r"../processed_data/"+str(file_id)+"year.csv" #清洗结果路径
        x_path = r"../data_excel/"+str(file_id)+"year.xls"  #excel路径
        file_name='../data/'+str(file_id)+'year.arff'   #源文件路径

        data,meta=arff.loadarff(file_name)

        print("==============Data_Infomation=============")#打印数据集信息
        len_d = len(data)

        print("File_id : " + str(file_id))
        print("Size : " + str(len_d) )
        print("=================Data_Miss================")

        data = profiling_miss(data)

        df=pd.DataFrame(data)
        out_file='../processed_data/'+str(file_id)+'year.csv'
        output=pd.DataFrame(df)
        output.to_csv(out_file,index=False)
        print("============End of data cleaning==========")

        #csv_to_xls(c_path,x_path)