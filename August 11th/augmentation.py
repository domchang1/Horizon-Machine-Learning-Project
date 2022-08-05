import Augmentor
from pathlib import Path
import glob
from PIL import Image
from torchvision import transforms, models
import torch
import pandas as pd
import pickle
import numpy as np

def createAugmentedImages():
    p = Augmentor.Pipeline("../modified_train_images/")
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.flip_left_right(probability=0.5)
    p.sample(27000)

preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])
resnet = models.resnet18(pretrained=True)
newmodel = torch.nn.Sequential(*(list(resnet.children())[:-1]))
inputs = []
labels = []
training_set = pd.read_csv("../train.csv", index_col=0)
training_set = training_set.to_dict()
# print(training_set)
# print(training_set['diagnosis']['000c1434d8d7'])
print("looking through files")
counter = 0
for filename in glob.glob('../augmented_train/*.png'):
    if (counter % 100 == 0):
        print(counter) 
    im = Image.open(filename)
    input = preprocessing(im)
    input_batch = input.unsqueeze(0)
    with torch.no_grad():
        output = newmodel(input_batch)[0,:,0,0]
    id = str(filename[50:62])
    # print(id)
    label = training_set['diagnosis'][id]
    inputs.append(output)
    labels.append(label)
    # print(output.shape)
    counter += 1
for filename in glob.glob('../modified_train_images/*.png'):
    im = Image.open(filename)
    input = preprocessing(im)
    input_batch = input.unsqueeze(0)
    with torch.no_grad():
        output = newmodel(input_batch)[0,:,0,0]
    id = str(filename[25:len(filename)-4])
    if (counter % 100 == 0):
        print(counter)
        # print(id)
    label = training_set['diagnosis'][id]
    inputs.append(output)
    labels.append(label)
    counter += 1

# print(len(inputs))
# print(len(labels))
inputs = np.stack(inputs)
inputs_dst = Path(f"../inputs3.pkl")
labels_dst = Path(f"../labels3.pkl")
inputs_dst.write_bytes(pickle.dumps(inputs))
labels_dst.write_bytes(pickle.dumps(labels))
#remove duplicates
     
