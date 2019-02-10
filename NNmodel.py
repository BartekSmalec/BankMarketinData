import keras
from keras import Sequential
from keras.layers import Dense


def create_model(optimizer='rmsprop', neuron_in_first=32, neuron_in_second=64, dropout_in_first=0, dropout_in_second=0,
                 activation_in_first='relu', activation_in_second='relu'):
    nn = Sequential()
    nn.add(Dense(neuron_in_first, input_dim=14, activation=activation_in_first))
    nn.add(Dense(neuron_in_second, activation=activation_in_second))

    nn.add(Dense(1, activation='sigmoid'))

    nn.compile(loss=keras.losses.binary_crossentropy,
               optimizer=optimizer,
               metrics=['accuracy'])
    return nn


def best_model():
    nn = Sequential()
    nn.add(Dense(32, input_dim=14, activation='relu'))
    nn.add(Dense(64, activation='relu'))

    nn.add(Dense(1, activation='sigmoid'))

    nn.compile(loss=keras.losses.binary_crossentropy,
               optimizer='rmsprop',
               metrics=['accuracy'])
    return nn
