from scipy.io import arff

from io import StringIO
import csv
import pandas as pd

c_path = r"../data/1year.csv"
x_path = r"../data_excel/1year.xls"  # 路径中的xls文件在调用to_excel时会自动创建


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


csv_to_xls(c_path, x_path)

file_name='../data/1year.arff'

data,meta=arff.loadarff(file_name)

print("==============Data_example=============")
print(data[0])

print("==================Meta=================")
print(meta)

df=pd.DataFrame(data)
out_file='../data/1year.csv'
output=pd.DataFrame(df)
output.to_csv(out_file,index=False)

csv_to_xls(c_path,x_path)