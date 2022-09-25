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

## SVM

#############################################################
#####                      SVM                          #####
#############################################################

from sklearn.svm import SVC


@ignore_warnings(category=ConvergenceWarning)
def svm_classifier(x_data, y_data, dataset):
    svm_scaler = StandardScaler().fit(x_data)
    X_scaled = svm_scaler.transform(x_data)

    grid_param = {
        "kernel": ['rbf', 'linear', 'sigmoid'],
        "gamma": [0.1, 0.3, 0.5],
        'C': np.arange(0.1, 1.0, 0.05).tolist(),
        'max_iter': np.arange(10, 250, 10).tolist()
    }

    grid_search = GridSearchCV(SVC(), grid_param, scoring='accuracy')
    grid_search.fit(X_scaled, y_data)

    C_value = grid_search.best_params_['C']
    kernel_type = grid_search.best_params_['kernel']
    max_iter = grid_search.best_params_['max_iter']
    gamma = grid_search.best_params_['gamma']

    print(
        'Best Parameters: C : {}, kernel : {}, gamma : {}, max_iter : {}'.format(C_value, kernel_type, gamma, max_iter))

    svc_model = SVC(C=C_value, kernel=kernel_type, max_iter=max_iter)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size=0.2, random_state=42)

    svc_model.fit(X_train, y_train)

    y_predicted = svc_model.predict(X_test)

    score = accuracy_score(y_test, y_predicted)

    print('Accuracy of SVM : {}'.format(score))

    training_sizes = np.arange(0.1, 0.95, 0.01).tolist()
    training_time = 0
    training_accuracy = list()
    testing_accuracy = list()

    for train_size in training_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, train_size=train_size, random_state=42)

        start_time = time.time()
        svc_model.fit(X_train, y_train)
        end_time = time.time()

        training_time += end_time - start_time

        y_in_pred = svc_model.predict(X_train)
        training_accuracy.append(accuracy_score(y_train, y_in_pred))

        y_predicted = svc_model.predict(X_test)
        testing_accuracy.append(accuracy_score(y_test, y_predicted))

    result = {'Training Size': training_sizes,
              'Training Accuracy': training_accuracy,
              'Testing Accuracy': testing_accuracy
              }
    dt_result_df = pd.DataFrame(result).set_index('Training Size')

    dt_result_df.plot.line()
    image_file_name = 'svm' + '/' + dataset + '_SVM_Learning_Curve.png'
    plt.title(dataset + ' - SVM Learning Curve')
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(image_file_name)
    plt.close()

    for param in grid_param.keys():
        plot_validation_curve(SVC(kernel=kernel_type), 'svm', X_scaled, y_data, param, grid_param[param], dataset)

    return training_time, score
