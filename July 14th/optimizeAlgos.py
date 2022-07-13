import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from pathlib import Path
import pandas as pd

def readInputLabels():
    inputs_dst = Path(f"../inputs.pkl")
    labels_dst = Path(f"../labels.pkl")
    inputs = pd.read_pickle(inputs_dst)
    labels = pd.read_pickle(labels_dst)
    return inputs, labels

def optimizeKNN():
    num_neighbors = [1,2,3,4,5,6,7,8,9,10]
    kn_a = []
    kn_d = []
    knr_a = []
    knr_d = []
    for i in num_neighbors:
        kneighbors = KNeighborsClassifier(n_neighbors=i).fit(train_inputs, train_labels)
        predictions = kneighbors.predict(validation_inputs)
        kn_a.append((predictions == validation_labels).astype(int).mean())
        kn_d.append(((predictions - validation_labels) ** 2).mean())
        kneighborsreg = KNeighborsRegressor(n_neighbors=i).fit(train_inputs, train_labels)
        predictions = kneighborsreg.predict(validation_inputs)
        knr_a.append((np.round(predictions,0) == validation_labels).astype(int).mean())
        knr_d.append(((predictions - validation_labels) ** 2).mean())
    plt.plot(num_neighbors, kn_a, label="Classifier Accuracy")
    plt.plot(num_neighbors, knr_a, label="Regressor Accuracy")
    plt.legend()
    plt.show()
    plt.plot(num_neighbors, kn_d, label="Classifier Distance")
    plt.plot(num_neighbors, knr_d, label="Regressor Distance")
    plt.legend()
    plt.show()

def optimizeLogRegress():
    c_values = [(i+1)*50 for i in range(20)]
    lg_accuracy = []
    lg_distance = []
    for i in c_values:
        logregress = LogisticRegression(random_state=0, C=i, max_iter=7000).fit(train_inputs, train_labels)
        predictions = logregress.predict(validation_inputs)
        lg_accuracy.append((predictions == validation_labels).astype(int).mean())
        lg_distance.append(((predictions - validation_labels) ** 2).mean())
    plt.plot(c_values, lg_accuracy, label="Logistic Regression Accuracy")
    plt.plot(c_values, lg_distance, label="Logistic Regression Distance")
    plt.legend()
    plt.show()

def optimizeDecTree():
    max_depths = [(i+1) for i in range(20)]
    dc_accuracy = []
    dc_distance = []
    for i in max_depths:
        tree = DecisionTreeClassifier(max_depth=i).fit(train_inputs, train_labels)
        predictions = tree.predict(validation_inputs)
        dc_accuracy.append((predictions == validation_labels).astype(int).mean())
        dc_distance.append(((predictions - validation_labels) ** 2).mean())
    plt.plot(max_depths, dc_accuracy, label="Decision Tree Accuracy")
    plt.plot(max_depths, dc_distance, label="Decision Tree Distance")
    plt.legend()
    plt.show()

def optimizeRandomForest():
    n_estimators = [(i+1)*10 for i in range(10)]
    rf_accuracy = []
    rf_distance = []
    for i in n_estimators:
        forest = RandomForestClassifier(n_estimators=i).fit(train_inputs, train_labels)
        predictions = forest.predict(validation_inputs)
        rf_accuracy.append((predictions == validation_labels).astype(int).mean())
        rf_distance.append(((predictions - validation_labels) ** 2).mean())
    plt.plot(n_estimators, rf_accuracy, label="Random Forest Accuracy")
    plt.plot(n_estimators, rf_distance, label="Random Forest Distance")
    plt.legend()
    plt.show()

np.random.seed(0)
inputs, labels = readInputLabels()

train_inputs = inputs[:2930]
train_labels = labels[:2930]
validation_inputs = inputs[2930:]
validation_labels = labels[2930:]
optimizeLogRegress()