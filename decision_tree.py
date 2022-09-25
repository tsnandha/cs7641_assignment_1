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

# Decision Tree
#############################################################
#####                      Decision Tree             #####
#############################################################

from sklearn.tree import DecisionTreeClassifier


def decision_tree_train_test(x_data, y_data, dataset):
    training_accuracy = list()
    testing_accuracy = list()

    grid_param = {"max_depth": range(2, 20, 1),
                  "min_samples_leaf": range(1, 50, 5),
                  "min_samples_split": range(2, 6, 2)
                  }

    dt_clf = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=dt_clf, param_grid=grid_param,
                               scoring='accuracy', cv=4, n_jobs=-1, verbose=1)
    grid_search.fit(x_data, y_data)

    best_param = grid_search.best_params_

    dt_model = DecisionTreeClassifier(max_depth=best_param['max_depth'],
                                      min_samples_leaf=best_param['min_samples_leaf'],
                                      min_samples_split=best_param['min_samples_split'])

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    dt_model.fit(X_train, y_train)

    y_predicted = dt_model.predict(X_test)

    score = accuracy_score(y_test, y_predicted)

    print('Accuracy of Decision Tree : {}'.format(score))

    training_sizes = np.arange(0.1, 0.95, 0.01).tolist()
    training_time = 0

    for train_size in training_sizes:
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_size, random_state=42)

        start_time = time.time()
        dt_model.fit(X_train, y_train)
        end_time = time.time()

        training_time += end_time - start_time

        y_in_pred = dt_model.predict(X_train)
        training_accuracy.append(accuracy_score(y_train, y_in_pred))

        y_predicted = dt_model.predict(X_test)
        testing_accuracy.append(accuracy_score(y_test, y_predicted))


    result = {'Training Size': training_sizes,
              'Training Accuracy': training_accuracy,
              'Testing Accuracy': testing_accuracy
              }
    dt_result_df = pd.DataFrame(result).set_index('Training Size')

    dt_result_df.plot.line()
    image_file_name = 'decision_tree' + '/' + dataset + '_Decision_Tree_Learning_Curve.png'
    plt.title(dataset + ' - Decision Tree Learning Curve')
    plt.savefig(image_file_name)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid(True)
    plt.close()

    for param in grid_param.keys():
        dt_clf = DecisionTreeClassifier()
        plot_validation_curve(dt_clf, 'decision_tree', x_data, y_data, param, grid_param[param], dataset)

    return training_time, score
