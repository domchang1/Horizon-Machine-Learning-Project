from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.neural_network import MLPClassifier
import numpy as np
from torchvision import transforms, models
import torch
import pandas as pd
import pickle
from PIL import Image
import glob

# load in images, resize and normalize, save each scenario (512 features + id + label) into new file

def loadFeatures():
    preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])
    resnet = models.resnet18(pretrained=True)
    newmodel = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    image_list = {}
    for filename in glob.glob('../test_images/*.png'): # switch depending on which images
        im=Image.open(filename)
        input = preprocessing(im)
        input_batch = input.unsqueeze(0)
        # if torch.cuda.is_available():
        #     input = input_batch.to('cuda')
        #     newmodel.to('cuda')
        with torch.no_grad():
            output = newmodel(input_batch)
        id = str(filename[15:len(filename)-4]) #16 for train, 15 for test
        image_list[id] = output
    return image_list

def writeScenarios(image_list):
    training_set = pd.read_csv("../test.csv") #switch test to train
    for i in range(len(training_set)):
        dst = Path(f"../testingscenarios/{i:04d}.pkl") #switch testing to training
        dst.write_bytes(pickle.dumps(dict(id_code=training_set.id_code[i], diagnosis=-1, features=image_list[training_set.id_code[i]])))
    
def checkDuplicates():
    preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])
    image_list = []
    for filename in glob.glob('../test_images/*.png'): # switch depending on which images
        im=Image.open(filename)
        input = preprocessing(im)
        image_list.append(input)
    for input in image_list:
        count = 0
        for other in image_list:
            if np.array_equal(input.numpy(), other.numpy()):
                count += 1
        if count > 1:
            print(input)

def getInputsLabels():
    inputs = []
    labels = []
    filenames = glob.glob('../trainingscenarios/*.pkl')
    filenames = sorted(filenames)
    np.random.shuffle(filenames)
    for filename in filenames:
        #print(filename)
        values = pd.read_pickle(filename)
        inputs.append(values['features'][0,:,0,0])
        labels.append(values['diagnosis'])
    print(len(inputs))
    new_inputs = []
    new_labels = []
    for i in range(len(inputs)):
        if not any((inputs[i] == new_inputs_).all() for new_inputs_ in new_inputs):
            new_inputs.append(inputs[i])
            new_labels.append(labels[i])
    print(len(new_inputs))

    inputs = np.stack(new_inputs)
    labels = new_labels
    inputs_dst = Path(f"../inputs.pkl")
    labels_dst = Path(f"../labels.pkl")
    inputs_dst.write_bytes(pickle.dumps(inputs))
    labels_dst.write_bytes(pickle.dumps(labels))

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

np.random.seed(0)
inputs, labels = readInputLabels()

train_inputs = inputs[:2930]
train_labels = labels[:2930]
validation_inputs = inputs[2930:]
validation_labels = labels[2930:]

kneighbors = KNeighborsClassifier(n_neighbors=1).fit(train_inputs, train_labels)
kneighborsreg = KNeighborsRegressor(n_neighbors=3).fit(train_inputs, train_labels)
logregress = LogisticRegression(random_state=0, C=700, max_iter=4400).fit(inputs, labels)
linregress = LinearRegression().fit(inputs, labels)
tree = DecisionTreeClassifier(max_depth=5).fit(inputs, labels)
#forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(inputs, labels)

models = [kneighbors, logregress]


exit()
print(kneighborsreg)
predictions = kneighborsreg.predict(validation_inputs)
print((np.round(predictions,0) == validation_labels).astype(int).mean())
print(((predictions - validation_labels) ** 2).mean()) #should I remove the exponent?
exit()
cm = confusion_matrix(validation_labels, predictions, labels=kneighbors.classes_, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=kneighbors.classes_)
disp.plot()      
plt.show()
#lsvm = SVC(kernel="linear", C=1000).fit(inputs, labels) #higher C means higher accuracy, lower distance
#rbfsvm =  SVC(gamma=2, C=1).fit(inputs, labels) #no tuning needed
#gausspc = GaussianProcessClassifier(1.0 * RBF(1.0), max_iter_predict=200, n_jobs=-1).fit(inputs, labels) #really slow
#tree = DecisionTreeClassifier(max_depth=5).fit(inputs, labels)
# forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(inputs, labels)
# mlp =  MLPClassifier(alpha=1, max_iter=1000).fit(inputs, labels)
# ada = AdaBoostClassifier().fit(inputs, labels)
# gaussnb = GaussianNB().fit(inputs, labels)
# qda =  QuadraticDiscriminantAnalysis().fit(inputs, labels)
# logregress = LogisticRegression(random_state=0, C=1000, max_iter=10000).fit(inputs, labels)
# linregress = LinearRegression().fit(inputs, labels)
# predictions = [kneighbors]
# #kneighbors, lsvm, rbfsvm, gausspc, tree, forest, mlp, ada, gaussnb, qda, logregress, linregress
# x = 1
# for i in predictions:
#     train_predictions = i.predict(train_inputs)
#     print((train_predictions == train_labels).astype(int).mean())
#     print(((train_predictions - train_labels) ** 2).mean())
#     print(x)
#     if x != 12:
#         cm = confusion_matrix(train_labels, train_predictions, labels=i.classes_)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=i.classes_)
#         disp.plot()      
#         plt.show()
#     x += 1

#test_predictions = kneighbors.predict(inputs)

# dst = Path("../trainingscenarios/0000.pkl")    
# print(pd.read_pickle(dst))

    