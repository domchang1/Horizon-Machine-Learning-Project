from pathlib import Path
from statistics import median
from skimage.io import imread_collection
from skimage.transform import resize
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
    
 
# dst = Path("../trainingscenarios/0000.pkl")    
# print(pd.read_pickle(dst))

    