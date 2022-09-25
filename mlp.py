######################################################################
# Author : Nandhakumar Thangavelu
# Project : CS7641 Machine Learning - Assignment #1
######################################################################


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from plots import *

# Neural Network

#############################################################
#####                      Neural networks              #####
#############################################################

from sklearn.neural_network import MLPClassifier


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=DeprecationWarning)
def mlp_classifier(x_data, y_data, dataset):

    grid_param = {"activation": ["relu", "tanh"],
                  "learning_rate_init": [0.01, 0.001, 0.0001],
                  "shuffle": [True, False],
                  "alpha": [0.01, 0.001, 0.0001],
                  }

    hidden_layer_size = (10,5)
    mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_size,
                            solver='adam', random_state=42, max_iter=2000)

    grid_search = GridSearchCV(estimator=mlp_clf, param_grid=grid_param,
                               scoring='accuracy', cv=4, n_jobs=-1, verbose=1)
    grid_search.fit(x_data, y_data)

    best_param = grid_search.best_params_

    mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_size, solver='adam', random_state=42, verbose=False,
                              activation=best_param['activation'],
                              learning_rate_init=best_param['learning_rate_init'],
                              shuffle=best_param['shuffle'],max_iter=2000,
                              )

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    start_time = time.time()
    mlp_model.fit(X_train, y_train)
    end_time = time.time()

    training_time = end_time - start_time

    y_predicted = mlp_model.predict(X_test)

    score = accuracy_score(y_test, y_predicted)

    print('Accuracy of MLP : {}'.format(score))

    loss_values = mlp_model.loss_curve_

    plt.figure()
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.title("Loss Curve : " + dataset + " Learning rate = " + str(best_param['learning_rate_init']) )
    plt.plot(loss_values)

    loss_file_name = 'mlp' + '/' + dataset + '_NN_Loss_Curve.png'
    plt.savefig(loss_file_name)
    plt.close()

    training_sizes = np.arange(0.1, 0.95, 0.01).tolist()
    training_accuracy = list()
    testing_accuracy = list()

    for train_size in training_sizes:
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_size, random_state=42)

        mlp_model.fit(X_train, y_train)

        y_in_pred = mlp_model.predict(X_train)
        training_accuracy.append(accuracy_score(y_train, y_in_pred))

        y_predicted = mlp_model.predict(X_test)
        testing_accuracy.append(accuracy_score(y_test, y_predicted))

    result = {'Training Size': training_sizes,
              'Training Accuracy': training_accuracy,
              'Testing Accuracy': testing_accuracy
              }
    dt_result_df = pd.DataFrame(result).set_index('Training Size')

    dt_result_df.plot.line()
    image_file_name = 'mlp' + '/' + dataset + '_NN_Learning_Curve.png'
    plt.title(dataset + ' - NN Learning Curve')
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(image_file_name)
    plt.close()

    grid_param = {"activation": ["relu", "tanh"],
                  "learning_rate_init": [0.01, 0.001, 0.0001],
                  "shuffle": [True, False],
                  "alpha": [0.01, 0.001, 0.0001],
                  "hidden_layer_sizes": [(10,), (50,), (100,), (150,), (200,), (250,)]
                  }

    for param in grid_param.keys():
        plot_validation_curve(mlp_clf, 'mlp', x_data, y_data, param, grid_param[param], dataset)
    return training_time, score

