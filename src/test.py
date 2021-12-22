for year in range(1,6):
    enroll  = f'../processed_data/{year}year.csv'
    total = 0
    num = 0
    with open(enroll) as lines:
        for id,data in enumerate(lines):
            if id == 0:
                continue

            temp = data.strip().split(",")
            feature = temp[20]
            total += 1
            if feature == '':
                num+=1
    print(num,total,num/total)