from time import time

import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer

import BankMarketinData as bm
import NNmodel as nnmodel


def main():
    data = bm.load_data('bank-full.csv')
    data = bm.preprocess_data(data)
    X,y = bm.split_data(data)

    scaler = Normalizer()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=0)


    start = time()
    model = KerasClassifier(build_fn=nnmodel.create_model, verbose=1)
    optimizers = ['rmsprop', 'adam','sgd']
    activations = ['relu', 'selu', 'tanh']
    epochs = np.array([10, 20, 30])
    batches = np.array([5, 10, 20])
    param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches)
    grid = GridSearchCV(estimator=model, param_grid=param_grid , n_jobs= -1)
    grid_result = grid.fit(x_train, y_train)

    print("total time:", time() - start)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.cv_results_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    print("total time:", time() - start)











if __name__ == "__main__":
    main()

