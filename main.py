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
from decision_tree import *
from boosting import *
from knn import *
from mlp import *
from svm import *

## Dataset locations
wine_quality_data_location = 'Data/winequality-white.csv'  ## Wine Quality data set
diabetes_data_location = 'Data/diabetes.csv'  ## diabetes  data
training_time_diabetes = {}
accuracy_diabetes = {}
training_time_wineq = {}
accuracy_wineq = {}

# Load our data set
def create_wine_data():

    wine_quality_df = pd.read_csv(wine_quality_data_location, sep=';')
    y = wine_quality_df['quality'] >= 6
    X = wine_quality_df.drop(columns='quality')
    
    return X, y


def create_diabetes_data():
    diabetes_df = pd.read_csv(diabetes_data_location, sep=',').dropna().reset_index(drop=True)
    diabetes_df.head()
    y = diabetes_df['Outcome']
    X = diabetes_df.drop(columns='Outcome')

    return X, y

def run_dt(diabetes_x, diabetes_y, wine_quality_x, wine_quality_y):
    print('Start Decision Tree - Diabetes')
    clock_time, accuracy = decision_tree_train_test(diabetes_x, diabetes_y, 'Diabetes')
    training_time_diabetes['decision_tree'] = clock_time
    accuracy_diabetes['decision_tree'] = accuracy
    print('End Decision Tree - Diabetes')
    print('Start Decision Tree - Wine')
    clock_time, accuracy = decision_tree_train_test(wine_quality_x, wine_quality_y, 'Wine Quality')
    training_time_wineq['decision_tree'] = clock_time
    accuracy_wineq['decision_tree'] = accuracy
    print('End Decision Tree - Wine')

def run_mlp(diabetes_x, diabetes_y, wine_quality_x, wine_quality_y):
    print('Start MLP - Diabetes')
    clock_time, accuracy = mlp_classifier(diabetes_x, diabetes_y, 'Diabetes')
    training_time_diabetes['mlp'] = clock_time
    accuracy_diabetes['mlp'] = accuracy
    print('End MLP - Diabetes')
    print('Start MLP - Wine')
    clock_time, accuracy = mlp_classifier(wine_quality_x, wine_quality_y, 'Wine Quality')
    training_time_wineq['mlp'] = clock_time
    accuracy_wineq['mlp'] = accuracy
    print('End MLP - Wine')

def run_boost(diabetes_x, diabetes_y, wine_quality_x, wine_quality_y):
    print('Start Boosted Decision Tree - Diabetes')
    clock_time, accuracy = boosted_decision_tree(diabetes_x, diabetes_y, 'Diabetes')
    training_time_diabetes['boosting'] = clock_time
    accuracy_diabetes['boosting'] = accuracy
    print('End Boosted Decision Tree - Diabetes')
    print('Start Boosted Decision Tree - Wine')
    clock_time, accuracy = boosted_decision_tree(wine_quality_x, wine_quality_y, 'Wine Quality')
    training_time_wineq['boosting'] = clock_time
    accuracy_wineq['boosting'] = accuracy
    print('End Boosted Decision Tree - Wine')

def run_svm(diabetes_x, diabetes_y, wine_quality_x, wine_quality_y):
    print('Start SVM - Diabetes')
    clock_time, accuracy = svm_classifier(diabetes_x, diabetes_y, 'Diabetes')
    training_time_diabetes['svm'] = clock_time
    accuracy_diabetes['svm'] = accuracy
    print('End SVM - Diabetes')
    print('Start SVM - Wine')
    clock_time, accuracy = svm_classifier(wine_quality_x, wine_quality_y, 'Wine Quality')
    training_time_wineq['svm'] = clock_time
    accuracy_wineq['svm'] = accuracy
    print('End SVM - Wine')

def run_knn(diabetes_x, diabetes_y, wine_quality_x, wine_quality_y):
    print('Start KNN - Diabetes')
    clock_time, accuracy = knn_classifier(diabetes_x, diabetes_y, 'Diabetes')
    training_time_diabetes['knn'] = clock_time
    accuracy_diabetes['knn'] = accuracy
    print('End KNN - Diabetes')
    print('Start KNN - Wine')
    clock_time, accuracy = knn_classifier(wine_quality_x, wine_quality_y, 'Wine Quality')
    training_time_wineq['knn'] = clock_time
    accuracy_wineq['knn'] = accuracy
    print('End KNN - Wine')


if __name__ == "__main__":

    diabetes_x, diabetes_y = create_diabetes_data()
    wine_quality_x, wine_quality_y = create_wine_data()

    run_dt(diabetes_x, diabetes_y, wine_quality_x, wine_quality_y)

    run_mlp(diabetes_x, diabetes_y, wine_quality_x, wine_quality_y)

    run_boost(diabetes_x, diabetes_y, wine_quality_x, wine_quality_y)

    run_svm(diabetes_x, diabetes_y, wine_quality_x, wine_quality_y)

    run_knn(diabetes_x, diabetes_y, wine_quality_x, wine_quality_y)

    print(training_time_diabetes)
    print(accuracy_diabetes)
    print(training_time_wineq)
    print(accuracy_wineq)

    plot_wall_clock_time(training_time_diabetes, 'Diabetes')
    plot_accuracy(accuracy_diabetes, 'Diabetes')
    plot_wall_clock_time(training_time_wineq, 'Wine Quality')
    plot_accuracy(accuracy_wineq, 'Wine Quality')




