from time import time

import numpy as np
import talos as tl
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import BankMarketinData as bm
import NNmodel as nnmodel

def main():

    # df = pd.read_csv('bank-full.csv',sep=';')
    # print(df.head())
    #
    # print(df.shape)
    #
    # print(df.columns)
    # print(df.info)
    # df.columns = [col.replace('"', '') for col in df.columns]
    #
    #
    # df.drop(columns=['day', 'poutcome'], axis =1 , inplace=True)
    #
    #
    # print(df.head())
    # print(df.shape)
    #
    # le = preprocessing.LabelEncoder()
    # df.job = le.fit_transform(df.job)
    # df.education = le.fit_transform(df.education)
    # df.housing = le.fit_transform(df.housing)
    # df.loan = le.fit_transform(df.loan)
    # #df.poutcome = le.fit_transform(df.poutcome)
    # df.month = le.fit_transform(df.month)
    # df.contact = le.fit_transform(df.contact)
    # df.marital = le.fit_transform(df.marital)
    # df.default = le.fit_transform(df.default)
    # df.y = le.fit_transform(df.y)
    #
    #
    #
    # print(df.head())
    #
    # X = df.iloc[:, 0:14]
    # y = df.iloc[:, 14]
    # X = np.array(X, dtype="float64")
    # y = np.array(y,dtype="float64")
    #
    # scaler = Normalizer()
    # X = scaler.fit_transform(X)
    #
    #
    #
    #
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=0)
    # model = LogisticRegression(penalty='l2', max_iter=1000)
    # model.fit(x_train, y_train)
    # prediction = model.predict(x_test)
    # from sklearn.metrics import accuracy_score
    # print("ACC: {} ".format(accuracy_score(y_test, prediction)))
    #
    #
    # print(x_train.shape)
    #
    # nn = Sequential()
    # nn.add(Dense(120,input_dim = 14, activation='relu'))
    # nn.add(Dense(240,activation='relu'))
    #
    #
    # nn.add(Dense(1))
    # nn.add(Activation('sigmoid'))
    #
    # nn.compile(loss=keras.losses.binary_crossentropy,
    #                optimizer='sgd',
    #                metrics=['accuracy'])
    #
    # nn.fit(x_train, y_train,
    #            batch_size=10,
    #            epochs=10,
    #            verbose=1,
    #
    #            validation_data=(x_test, y_test))
    #
    # loss_acc = nn.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', loss_acc[0])
    # print('Test accuracy:', loss_acc[1])


    data = bm.load_data('bank-full.csv')
    data = bm.preprocess_data(data)
    X,y = bm.split_data(data)

    scaler = Normalizer()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=0)


    start = time()
    model = KerasClassifier(build_fn=nnmodel.create_model, verbose = 0)
    optimizers = ['rmsprop', 'adam','sgd']
    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = np.array([10, 20, 30])
    batches = np.array([5, 10, 20])
    param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
    grid = GridSearchCV(estimator=model, param_grid=param_grid , n_jobs= -1)
    grid_result = grid.fit(x_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    print("total time:", time() - start)











if __name__ == "__main__":
    main()

