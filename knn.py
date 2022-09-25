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

# KNN Classifier

#############################################################
#####                      KNN                          #####
#############################################################

from sklearn.neighbors import KNeighborsClassifier


def knn_classifier(x_data, y_data, dataset):
    knn_scaler = StandardScaler().fit(x_data)
    X_scaled = knn_scaler.transform(x_data)

    grid_param = {
        'n_neighbors': np.arange(1, 15, 1).tolist(),
        'p': np.arange(1, 6)
    }

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=grid_param, scoring='accuracy', cv=4, n_jobs=-1, verbose=1)
    grid_search.fit(X_scaled, y_data)

    n_neighbors = grid_search.best_params_['n_neighbors']
    p = grid_search.best_params_['p']

    print('Best Parameters: n_neighbors : {}, p : {}'.format(n_neighbors, p))

    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size=0.2, random_state=42)

    knn_model.fit(X_train, y_train)

    y_predicted = knn_model.predict(X_test)

    score = accuracy_score(y_test, y_predicted)

    print('Accuracy of KNN : {}'.format(score))

    training_sizes = np.arange(0.1, 0.95, 0.01).tolist()
    training_accuracy = list()
    testing_accuracy = list()

    training_time = 0

    for train_size in training_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, train_size=train_size, random_state=42)

        start_time = time.time()
        knn_model.fit(X_train, y_train)
        end_time = time.time()

        training_time += end_time - start_time

        y_in_pred = knn_model.predict(X_train)
        training_accuracy.append(accuracy_score(y_train, y_in_pred))

        y_predicted = knn_model.predict(X_test)
        testing_accuracy.append(accuracy_score(y_test, y_predicted))

    result = {'Training Size': training_sizes, 'Training Accuracy': training_accuracy,
              'Testing Accuracy': testing_accuracy}
    knn_result_df = pd.DataFrame(result).set_index('Training Size')

    knn_result_df.plot.line()
    image_file_name = 'knn' + '/' + dataset + '_KNN_Learning_Curve.png'
    plt.title(dataset + ' - KNN Learning Curve')
    plt.savefig(image_file_name)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()
    plt.close()

    for param in grid_param.keys():
        plot_validation_curve(KNeighborsClassifier(), 'knn', X_scaled, y_data, param, grid_param[param], dataset)

    return training_time, score
