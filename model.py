import time
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation, add
from keras.layers import Lambda, concatenate
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adadelta, Adam
from keras import optimizers, regularizers
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

channel_axis = -1

nb_classes = 10
img_rows, img_cols = 32, 32
img_channel = 3
epochs = 100
batch_size = 128
dropout = 0.8
cardinality = 8
weight_decay = 0.0001
iterations = 417

mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]

def scheduler(epoch):
    if epoch <= 75:
        return 0.05
    elif epoch <= 150:
        return 0.005
    elif epoch <= 210:
        return 0.0005

    return 0.0001

def conv(x, n_filter, row, col, strides=(1, 1), padding='same', bias=False):
    x = Convolution2D(n_filter, [row, col], strides=strides, padding=padding,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x

def group_conv(x, n_filter, row, col, strides=(1, 1), padding='same', bias=False):
    h = int(n_filter / cardinality)
    groups = []
    for i in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, i * h: i * h + h])(x)
        groups.append(Convolution2D(h, kernel_size=(3, 3), strides=strides,
                             kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(weight_decay),
                             padding='same', use_bias=bias)(group))
    x = concatenate(groups)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x

def res_group_block(x, n_filter, row, col, strides=(1, 1), padding='same', bias=False):
    x1 = group_conv(x, n_filter, row, col)
    x2 = group_conv(x, n_filter, row, col)
    x = add([x2, x])
    return x

def res_block(x, n_filter, row, col, strides=(1, 1), padding='same', bias=False):
    x1 = conv(x, n_filter, row, col)
    x2 = conv(x1, n_filter, row, col)
    x = add([x2, x])
    return x


def dense_block(x, n_filter, row, col, strides=(1, 1), padding='same', bias=False):
    concat = x
    for _ in range(3):
        #x = res_group_block(x, n_filter, row, col)
        x = res_block(x, n_filter, row, col)
        concat = concatenate([concat, x])
    return concat

def transition(x, n_filter, row, col, strides=(1, 1), padding='same', bias=False):
    x = conv(x, n_filter, row, col)
    x = AveragePooling2D((2, 2))(x)
    return x

def dense(size, x):
    x = Dense(size)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x

def create_model():
    input_shape = (img_rows, img_cols, img_channel)
    input = Input(input_shape)

    '''x = conv(input,32,3,3)
    for _ in range(2):
        x = conv(x,32,3,3)
    x = MaxPooling2D((2,2))(x)

    for _ in range(7):
        x = conv(x,64,3,3)
    x = AveragePooling2D((2,2))(x)'''

    '''x = conv(input, 64, 3, 3)
    x = residual_block(x, 64, 3, 3)
    x = residual_block(x, 64, 3, 3)
    x = residual_block(x, 64, 3, 3)
    x = MaxPooling2D((2, 2))(x)'''

    x = conv(input, 64, 3, 3)
    x = dense_block(x, 64, 3, 3)
    x = transition(x, 32, 3, 3)
    x = dense_block(x, 32, 3, 3)
    x = transition(x, 32, 3, 3)
    x = dense_block(x, 32, 3, 3)
    x = transition(x, 32, 3, 3)


    x = Dropout(dropout)(x)
    x = Flatten()(x)

    x = dense(128, x)
    for _ in range(3):
        x = dense(128, x)
    for _ in range(3):
        x = dense(64, x)

    x = Dropout(dropout)(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(input, output)
    adam = Adam()
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    return model



def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    return x_train, x_test


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

x_train, x_test = color_preprocessing(x_train, x_test)

if __name__ == '__main__':
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr]

    model = create_model()
    time1 = time.time()
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, verbose=1)
    time2 = time.time()
    model.save('cifar10.h5')

    e = model.evaluate(x_test, y_test, verbose=1)
    print('acc:', e[1])
    print('time:', (time2 - time1) / 3600)

