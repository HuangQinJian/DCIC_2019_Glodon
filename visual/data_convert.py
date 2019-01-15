import pandas as pd

train = pd.read_csv('data/train_labels.csv')
data = pd.DataFrame()
data['format'] = train['ID']

for i in range(data.shape[0]):
    data['format'][i] = 'data/train_dataset/' + data['format'][i]

for i in range(data.shape[0]):
    x1 = train['Detection'][i].split(' ')[0]
    y1 = train['Detection'][i].split(' ')[1]
    x2 = train['Detection'][i].split(' ')[2]
    y2 = train['Detection'][i].split(' ')[3]
    data['format'][i] = data['format'][i] + ',' + \
        x1 + ',' + y1 + ',' + x2 + ',' + y2 + ',' + 'fe'

data.to_csv('annotate.txt', header=None, index=None, sep=' ')