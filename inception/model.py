import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, MaxPool1D, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, Concatenate, Activation, Input, Add, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def fc(input_shape=(None, None)) -> Sequential:
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    return model


def cnn_1d(input_shape=(None, None)) -> Sequential:
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model


def inception_time(input_shape=(None, None)) -> Sequential:
    def _inception_module(input_tensor, stride=1, activation='linear'):
        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=bottleneck_size, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i], strides=stride, padding='same',
                                    activation=activation, use_bias=False)(input_inception))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = Conv1D(filters=nb_filters, kernel_size=1, padding='same', activation=activation,
                        use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        return x

    def _shortcut_layer(input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1, padding='same',
                            use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_tensor])
        x = Activation('relu')(x)
        return x

    nb_filters = 32
    use_residual = True
    use_bottleneck = True
    model_depth = 3
    kernel_size = 17
    bottleneck_size = 32

    input_layer = Input(shape=input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(model_depth):
        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = GlobalAveragePooling1D()(x)

    output_layer = Dense(1, activation='linear')(gap_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
