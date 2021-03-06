import random

random.seed(20211213)
for year in range(1,6):
    source = f'../processed_data/{year}year.csv'
    enroll = f'../new_data_split/enroll_{year}year.csv'
    valid = f'../new_data_split/valid_{year}year.csv'
    test = f'../new_data_split/test_{year}year.csv'
    enroll_list_1 = []
    enroll_list_0 = []

    with open(source) as lines, open(enroll,'w') as enroll_writer, open(valid,'w') as valid_writer, open(test,'w') as test_writer:
        for id,i in enumerate(lines):
            if id == 0:
                enroll_writer.write(i)
                test_writer.write(i)
                continue

            label = i.strip().split(",")[-1]
            if year < 3:
                temp = i.strip().split(",")
                i = ','.join(temp[:20] + temp[21:])
                i += '\n'
            if label == "1":
                enroll_list_1.append(i)
            else:
                enroll_list_0.append(i)
        

        random.shuffle(enroll_list_0)
        random.shuffle(enroll_list_1)
        num_0 = int(0.8 * len(enroll_list_0))
        num_1 = int(0.9 * len(enroll_list_0))
        for id,i in enumerate(enroll_list_0):
            if id < num_0:
                enroll_writer.write(i)
            elif id < num_1:
                valid_writer.write(i)
            else:
                test_writer.write(i)

        num_0 = int(0.8 * len(enroll_list_1))
        num_1 = int(0.9 * len(enroll_list_1))
        for id,i in enumerate(enroll_list_1):
            if id < num_0:
                enroll_writer.write(i)
            elif id < num_1:
                valid_writer.write(i)
            else:
                test_writer.write(i)
