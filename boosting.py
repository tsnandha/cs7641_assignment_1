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

## Boosting Decision Tree

#############################################################
#####                      Boosting Decision Tree       #####
#############################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


def boosted_decision_tree(x_data, y_data, dataset):
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    dt_clf = DecisionTreeClassifier(max_depth=4, random_state=42)

    grid_param = {
        "n_estimators": [1, 5, 10, 20, 40, 50, 100, 200],
        "learning_rate": np.arange(0.1, 2.1, 0.1)
    }
    grid_search = GridSearchCV(AdaBoostClassifier(dt_clf), grid_param,scoring='accuracy', cv=4, n_jobs=-1, verbose=1)
    grid_search.fit(x_data, y_data)

    best_params = grid_search.best_params_
    n_estimators = best_params['n_estimators']
    learning_rate = best_params['learning_rate']

    boost_clf = AdaBoostClassifier(dt_clf, n_estimators=n_estimators, learning_rate=learning_rate)
    boost_clf.fit(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    start_time = time.time()
    boost_clf.fit(X_train, y_train)
    end_time = time.time()

    training_time = end_time - start_time

    y_predicted = boost_clf.predict(X_test)

    score = accuracy_score(y_test, y_predicted)

    print('Accuracy of Boosting : {}'.format(score))

    for param in grid_param.keys():
        plot_validation_curve(AdaBoostClassifier(dt_clf), 'boosting', x_data, y_data, param, grid_param[param], dataset)

    training_accuracy = list()
    testing_accuracy = list()
    training_sizes = np.arange(0.1, 0.95, 0.01).tolist()
    for train_size in training_sizes:
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_size, random_state=42)

        boost_clf.fit(X_train, y_train)

        y_in_pred = boost_clf.predict(X_train)
        training_accuracy.append(accuracy_score(y_train, y_in_pred))

        y_predicted = boost_clf.predict(X_test)
        testing_accuracy.append(accuracy_score(y_test, y_predicted))

    result = {'Training Size': training_sizes, 'Testing Accuracy': testing_accuracy,
              'Training Accuracy': training_accuracy}
    dt_result_df = pd.DataFrame(result).set_index('Training Size')
    dt_result_df.head()

    dt_result_df.plot.line()

    image_file_name = 'boosting' + '/' + dataset + '_Decision_Tree_Results.png'
    plt.title(dataset + ' - Boosting DT Learning Curve')
    plt.savefig(image_file_name)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid(True)
    plt.close()

    return training_time, score
