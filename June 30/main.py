from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neural_network import MLPClassifier
import imageio.v3 as iio
import numpy as np
from torchvision import transforms, models
import torch
from torchvision.utils import save_image
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
    
# image_list = loadFeatures()
# writeScenarios(image_list)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# newmodel = newmodel.to(device) # on gpu
# #inputs, labels = data[0].to(device), data[1].to(device) 
inputs = []
labels = []
for filename in glob.glob('../trainingscenarios/*.pkl'):
    #print(filename)
    values = pd.read_pickle(filename)
    inputs.append(values['features'][0,:,0,0])
    labels.append(values['diagnosis'])
inputs = np.stack(inputs)
print(inputs.shape, inputs.dtype)
kneighbors = KNeighborsClassifier(n_neighbors=5).fit(inputs, labels)
lsvm = SVC(kernel="linear", C=0.025).fit(inputs, labels)
rbfsvm =  SVC(gamma=2, C=1).fit(inputs, labels)
gausspc = GaussianProcessClassifier(1.0 * RBF(1.0)).fit(inputs, labels)
tree = DecisionTreeClassifier(max_depth=5).fit(inputs, labels)
forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(inputs, labels)
mlp =  MLPClassifier(alpha=1, max_iter=1000).fit(inputs, labels)
ada = AdaBoostClassifier().fit(inputs, labels)
gaussnb = GaussianNB().fit(inputs, labels)
qda =  QuadraticDiscriminantAnalysis().fit(inputs, labels)
logregress = LogisticRegression(random_state=0, C=1000, max_iter=10000).fit(inputs, labels)
linregress = LinearRegression().fit(inputs, labels)
predictions = [kneighbors, lsvm, rbfsvm, gausspc, tree, forest, mlp, ada, gaussnb, qda, logregress, linregress]

x = 1
for i in predictions:
    train_predictions = i.predict(inputs)
    print(x)
    x += 1
    print((train_predictions == labels).astype(int).mean())
    print(((train_predictions - labels) ** 2).mean())


# dst = Path("../trainingscenarios/0000.pkl")    
# print(pd.read_pickle(dst))

    