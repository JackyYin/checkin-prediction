import keras
import numpy as np
import csv
from keras import utils as np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


# process train data
def get_staff_dict():
    with open('train.csv', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        staffs = []

        for row in reader:
            staffs.append(row['staff_id'])

        staffs = sorted(list(map(int, set(staffs))))

        staffs_train = np_utils.to_categorical(range(len(staffs)), len(staffs))
        staffs_dict = dict(zip(staffs, staffs_train))

        # print (len(staffs))
        # print (staffs)
        # print (staffs_dict)
    return staffs_dict


def get_train_data():
    with open('train.csv', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        weekdays_train = np_utils.to_categorical(range(7), 7)
        staffs_dict = get_staff_dict()
        x_train = []
        y_train = []
        for row in reader:
            col1 = staffs_dict[int(row['staff_id'])]
            col2 = weekdays_train[int(row['weekday'])]
            col3 = int(row['yesterday'])
            col4 = int(row['before_yesterday'])
            col5 = int(row['three_days_ago'])
            res = np.concatenate((col1, col2, col3, col4, col5), axis=None)
            x_train.append(res)
            y_train.append(int(row['is_leave']))
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        dic = {}
        dic['x_train'] = x_train
        dic['y_train'] = y_train
    return dic


# train
def train():
    train_data = get_train_data()
    x_train = train_data['x_train']
    y_train = train_data['y_train']
    model = Sequential()
    model.add(Dense(64, input_dim=73, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss='binary_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])

    model.fit(x_train[0:-1000], y_train[0:-1000],
        epochs=20,
        batch_size=32)

    model.save('my_model.h5')
    score = model.evaluate(x_train[-1000:-1], y_train[-1000:-1], batch_size=32)
    return score


def predict(arr):
    model = load_model('my_model.h5')
    return model.predict(arr)
