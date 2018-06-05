import math
import numpy as np
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.utils import np_utils

'''def f_similarity(y_aver, y_out):
    y_tmp = y_aver - y_out
    res = 0
    for data in y_tmp:
        tmp = math.exp(abs(data))
        tmp = pow((tmp-1),2)
        print(data,tmp)
        res += tmp
    return res'''

def f_similarity(y_aver, y_out):
    print(type(y_aver),type(y_out))
    y_tmp = y_aver - y_out
    res = 0
    for data in y_tmp:
        tmp = math.log(1-abs(data))
        res += pow(tmp,2)
    return res

def cal_average():
    dict_aver = {}


epochs = 15
batch_size = 100
nb_classes = 10


img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


half_len = int(0.5 * len(x_train))
x_train = x_train[half_len:]
y_train = y_train[half_len:]

model = load_model('cnn.h5')
result = model.predict(x_train)

y_aver = [0] * 10
y1 = []
y2 = []

count = 0
for y_out, y_truth in zip(result,y_train):
    if y_truth[0] == 1:
        y_aver = [a+b for a,b in zip(y_aver,y_out)]
        count += 1
        y1.append(y_out)
    if y_truth[1] == 1:
        y2.append((y_out))

data = np.array(x_train[0]+x_train[1])
for i in range(len(data)):
    for j in range(len(data)):
        data[i][j] += 1
data = data.reshape(1,28,28,1)
y_d = model.predict(data)

y_aver = [x/count for x in y_aver]
for y_out in y2:
    y = f_similarity(y_aver, y_out)
    if y < 1:
        print(y)

'''print(f_similarity(y_aver, y1[0]))
print(f_similarity(y_aver, y1[1]))
print(f_similarity(y_aver, y2[0]))
print(f_similarity(y_aver, y_d[0]))'''