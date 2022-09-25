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

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def plot_validation_curve(classifier, classifier_name, X_train, y_train, param_name, param_range, dataset_name):

    train_scores, test_scores = validation_curve(classifier, X_train, y_train, param_name=param_name,
                                                 param_range=param_range, scoring='accuracy')
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

    validation_parameters = {
        'param': param_range,
        'Training Score': train_mean,
        'Validation Score': test_mean
    }

    val_df = pd.DataFrame(validation_parameters).set_index('param')
    val_df.plot.line()

    title = "{} {} Validation Curve".format(dataset_name, param_name)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()

    image_file_name = classifier_name + '/' + dataset_name + '_' + param_name + '_Validation_Curve.png'
    plt.savefig(image_file_name)
    plt.switch_backend('agg')
    plt.close()

def plot_wall_clock_time(training_time, dataset_name):

    plt.close()
    dataset_alg = list(training_time.keys())
    values = list(training_time.values())

    plt.bar(range(len(training_time)), values, tick_label=dataset_alg)
    plt.title("Training Time vs Algorithms - " + dataset_name)
    plt.xlabel("Learning Algorithms")
    plt.ylabel("Training Time")
    plt.grid()

    image_file_name = dataset_name + '_Training_Time_Algorithms.png'
    plt.savefig(image_file_name)
    plt.switch_backend('agg')
    plt.close()

def plot_accuracy(accuracy, dataset_name):

    plt.close()
    dataset_alg = list(accuracy.keys())
    values = list(accuracy.values())

    plt.bar(range(len(accuracy)), values, tick_label=dataset_alg)
    plt.title("Accuracy vs Algorithms - " + dataset_name)
    plt.xlabel("Learning Algorithms")
    plt.ylabel("Training Time")
    plt.grid()

    image_file_name = dataset_name + '_Accuracy_Algorithms.png'
    plt.savefig(image_file_name)
    plt.switch_backend('agg')
    plt.close()
