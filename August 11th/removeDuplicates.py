import os
from pathlib import Path
import shutil
import glob
import numpy as np
import pandas as pd
from torchvision import transforms, models
import torch
from PIL import Image
import pickle


preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])
resnet = models.resnet18(pretrained=True)
newmodel = torch.nn.Sequential(*(list(resnet.children())[:-2]))
ids = []
not_ids = []
inputs = []
image_list = {}
for filename in glob.glob('../modified_train_images/*.png'): # switch depending on which images
    im=Image.open(filename)
    input = preprocessing(im)
    input_batch = input.unsqueeze(0)
    # if torch.cuda.is_available():
    #     input = input_batch.to('cuda')
    #     newmodel.to('cuda')
    with torch.no_grad():
        output = newmodel(input_batch)
    id = str(filename[16:len(filename)-4]) #16 for train, 15 for test
    image_list[id] = output
    # if not any((output == new_id).all() for new_id in inputs):
    #     ids.append(id)
    #     inputs.append(output)
    # else:
    #     not_ids.append(id)
print("done processing")
ids = []
duplicates = []
inputs = []
count = 0
for id in image_list:
    # print(count)
    # count += 1
    if not any((image_list[id] == new_id).all() for new_id in inputs):
        ids.append(id)
        inputs.append(image_list[id])
    else:
        duplicates.append(id)
print(len(ids))
print(len(duplicates))
dst = Path(f"../duplicates.pkl")
dst.write_bytes(pickle.dumps(inputs))


for i in duplicates:
    os.remove(f"../modified_train_images/{i[9:]}.png") # one file at a time
