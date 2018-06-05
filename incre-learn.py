import numpy as np
import os
import math
import model as cnn
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adadelta

path = os.getcwd()

channel_axis = -1

nb_classes = 10
img_rows, img_cols = 28, 28


def conv(x, n_filter, row, col, strides=(1, 1), padding='same', bias=False):
    x = Convolution2D(n_filter, [row, col], strides=strides, padding=padding, use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    return x


def get_y_aver(model, x_train, y_train):
    output = model.predict(x_train)
    y_aver = {}
    for i in range(10):
        y_aver[i] = [0] * 10
    count = [0] * 10

    for y_out, y_truth in zip(output, y_train):
        y_out = list(y_out)
        y_truth = list(y_truth)

        i_truth = y_truth.index(1)
        i_out = y_out.index(max(y_out))
        if i_truth == i_out:
            y_aver[i_out] = [a + b for a, b in zip(y_aver[i_out], y_out)]
            count[i_out] += 1

    #print(y_aver)
    #print(count)
    for i in range(10):
        y_aver[i] = [x/count[i] for x in y_aver[i]]

    return y_aver

def f_similarity(y_aver, y_out):
    y_tmp = y_aver - y_out
    res = 0
    for data in y_tmp:
        tmp = math.log(1-abs(data))
        res += pow(tmp, 2)

    if res <= 1:
        return 1
    else:
        return 0

def get_newTrain(model, x_train, y_aver):
    x_tmp = []
    y_tmp = []
    for x_data in x_train:

        x_data = x_data.reshape(1, 28, 28, 1)
        result = model.predict(x_data)
        y_out = result[0]
        i = list(y_out).index(max(y_out))

        y = f_similarity(y_aver[i], y_out)
        if y <= 1:
            x_tmp.append(x_data)
            y_tmp.append(i)

    x_tmp = np.array(x_tmp)
    x_tmp = x_train.reshape(x_tmp.shape[0], img_rows, img_cols, 1)
    y_tmp = np_utils.to_categorical(y_tmp, nb_classes)

    return x_tmp, y_tmp

def creat_data():
    data = np.array(x_train[0] + x_train[1])
    for i in range(len(data)):
        for j in range(len(data)):
            data[i][j] += 1


def get_minist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    print(y_train)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return [x_train, y_train, x_train, y_train]

def generate_batch(index, len, data):
    return data[index*len : (index+1)*len]

epochs = 20
batch_size = 100

minist_data = get_minist()
x_train, y_train = minist_data[0], minist_data[1]
x_test, y_test = minist_data[2], minist_data[3]

first_len = 2000
total_len = x_train.shape[0]
train_len = int(1 / 3 * len(x_train))
x_train1 = x_train[:first_len]
y_train1 = y_train[:first_len]
x_train2 = x_train[first_len:]
y_train2 = y_train[first_len:]

model = cnn.create_model()
model.fit(x_train1, y_train1, batch_size=batch_size, epochs=epochs, verbose=0)
#model.save('cnn.h5')


num_train = first_len
q_epochs = int((total_len-first_len) / num_train)
x_pass = x_train1
y_pass = y_train1

for n in range(q_epochs-1):
    x_batch = generate_batch(n, num_train, x_train2)
    y_aver = get_y_aver(model,x_pass, y_pass)
    x_train    , y_train = get_newTrain(model, x_batch, y_aver)
    x_new = np.concatenate((x_train, x_pass))
    y_new = np.concatenate((y_train, y_pass))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    result = model.evaluate(x_test, y_test, verbose=0)
    print(result)

    x_pass = x_new
    y_pass = y_new

#model = load_model('cnn.h5')

'''y_aver = get_y_aver(model, x_train1, y_train1)

x_train3, y_train3 = get_newTrain(model, x_train3, y_aver)
x_train3 = np.concatenate((x_train3,x_train1))
y_train3 = np.concatenate((y_train3, y_train1))
model.fit(x_train3, y_train3, batch_size=batch_size, epochs=epochs, verbose=0)
result = model.evaluate(x_test, y_test, verbose=0)
print(result)'''
