from keras import Sequential
from keras.layers import Dense
import keras


def create_model(optimizer='rmsprop',init='glorot_uniform'):
    nn = Sequential()
    nn.add(Dense(32, input_dim=14, activation='relu'))
    nn.add(Dense(64,init=init, activation='relu'))

    nn.add(Dense(1, activation='sigmoid'))

    nn.compile(loss=keras.losses.binary_crossentropy,
               optimizer=optimizer,
               metrics=['accuracy'])
    return nn
