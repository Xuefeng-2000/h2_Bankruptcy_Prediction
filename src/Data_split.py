import random

random.seed(20211213)
for i in range(1,6):
    source = f'../processed_data/{i}year.csv'
    enroll = f'../data_split/enroll_{i}year.csv'
    test = f'../data_split/test_{i}year.csv'
    enroll_list_1 = []
    enroll_list_0 = []

    with open(source) as lines, open(enroll,'w') as enroll_writer, open(test,'w') as test_writer:
        for id,i in enumerate(lines):
            if id == 0:
                enroll_writer.write(i)
                test_writer.write(i)
                continue

            label = i.strip().split(",")[-1]
            if label == "1":
                enroll_list_1.append(i)
            else:
                enroll_list_0.append(i)
        

        random.shuffle(enroll_list_0)
        random.shuffle(enroll_list_1)
        num_0 = int(0.7 * len(enroll_list_0))
        for id,i in enumerate(enroll_list_0):
            if id < num_0:
                enroll_writer.write(i)
            else:
                test_writer.write(i)

        num_1 = int(0.7 * len(enroll_list_1))
        for id,i in enumerate(enroll_list_1):
            if id < num_1:
                enroll_writer.write(i)
            else:
                test_writer.write(i)