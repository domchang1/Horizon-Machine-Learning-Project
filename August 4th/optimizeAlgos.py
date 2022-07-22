import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.semi_supervised import LabelSpreading
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from pathlib import Path
import pandas as pd

def readInputLabels():
    inputs_dst = Path(f"../inputs2.pkl")
    labels_dst = Path(f"../labels2.pkl")
    inputs = pd.read_pickle(inputs_dst)
    labels = pd.read_pickle(labels_dst)
    return inputs, labels

def turnIntoDetection(labels):
    new_labels = []
    for i in range(len(labels)):
        if labels[i] == 0:
            new_labels.append(0)
        else:
            new_labels.append(1)
    return new_labels

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
        # print(kn_a[-1] - knr_a[-1])
    fig, ax = plt.subplots()
    ax.plot(num_neighbors, kn_a, color="red")
    ax.set_xlabel("Number of Neighbors", fontsize=14)
    ax.set_ylabel("Accuracy", color="red", fontsize=14)
    ax.set_ylim([0,1])
    ax2 = ax.twinx()
    ax2.plot(num_neighbors, kn_d, color="blue")
    ax2.set_ylabel("Distance (MSE)", color="blue", fontsize=14)
    ax2.set_ylim([0,3])
    plt.title("K Nearest Neighbors Classifier Accuracy and Distance")
    plt.show()
    fig, ax = plt.subplots()
    ax.plot(num_neighbors, knr_a, color="red")
    ax.set_xlabel("Number of Neighbors", fontsize=14)
    ax.set_ylabel("Accuracy", color="red", fontsize=14)
    ax.set_ylim([0,1])
    ax2 = ax.twinx()
    ax2.plot(num_neighbors, knr_d, color="blue")
    ax2.set_ylabel("Distance (MSE)", color="blue", fontsize=14)
    ax2.set_ylim([0,3])
    plt.title("K Nearest Neighbors Regressor Accuracy and Distance")
    plt.show()

def optimizeLogRegress():
    c_values = [(i+1)*50 for i in range(20)]
    lg_accuracy = []
    lg_distance = []
    for i in c_values:
        logregress = LogisticRegression(random_state=0, C=i, max_iter=10000).fit(train_inputs, train_labels)
        predictions = logregress.predict(train_inputs) #validation_inputs
        lg_accuracy.append((predictions == train_labels).astype(int).mean())
        lg_distance.append(((predictions - train_labels) ** 2).mean())
    fig, ax = plt.subplots()
    ax.plot(c_values, lg_accuracy, color="red")
    ax.set_xlabel("C value", fontsize=14)
    ax.set_ylabel("Accuracy", color="red", fontsize=14)
    ax.set_ylim([0,1])
    ax2 = ax.twinx()
    ax2.plot(c_values, lg_distance, color="blue")
    ax2.set_ylabel("Distance (MSE)", color="blue", fontsize=14)
    ax2.set_ylim([0,3])
    plt.title("Logisitic Regression Accuracy and Distance")
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
    fig, ax = plt.subplots()
    ax.plot(max_depths, dc_accuracy, color="red")
    ax.set_xlabel("Max Depth", fontsize=14)
    ax.set_ylabel("Accuracy", color="red", fontsize=14)
    ax.set_ylim([0,1])
    ax2 = ax.twinx()
    ax2.plot(max_depths, dc_distance, color="blue")
    ax2.set_ylabel("Distance (MSE)", color="blue", fontsize=14)
    ax2.set_ylim([0,3])
    plt.title("Decision Tree Accuracy and Distance")
    plt.show()

def optimizeRandomForest():
    n_estimators = [(i+1)*10 for i in range(10)]
    rf_accuracy = []
    rf_distance = []
    for i in n_estimators:
        forest = RandomForestClassifier(n_estimators=i, max_depth=4).fit(train_inputs, train_labels)
        predictions = forest.predict(validation_inputs)
        rf_accuracy.append((predictions == validation_labels).astype(int).mean())
        rf_distance.append(((predictions - validation_labels) ** 2).mean())
    fig, ax = plt.subplots()
    ax.plot(n_estimators, rf_accuracy, color="red")
    ax.set_xlabel("Num Estimators", fontsize=14)
    ax.set_ylabel("Accuracy", color="red", fontsize=14)
    ax.set_ylim([0,1])
    ax2 = ax.twinx()
    ax2.plot(n_estimators, rf_distance, color="blue")
    ax2.set_ylabel("Distance (MSE)", color="blue", fontsize=14)
    ax2.set_ylim([0,3])
    plt.title("Random Forest Accuracy and Distance")
    plt.show()

np.random.seed(0)
inputs, labels = readInputLabels()
# train_inputs = inputs[:2930]
# train_labels = labels[:2930]
# validation_inputs = inputs[2930:]
# validation_labels = labels[2930:]
train_inputs = inputs[:8000]
train_labels = labels[:8000]
validation_inputs = inputs[8000:]
validation_labels = labels[8000:]
optimizeLogRegress()
# optimizeKNN()
# optimizeRandomForest()
# optimizeDecTree()
