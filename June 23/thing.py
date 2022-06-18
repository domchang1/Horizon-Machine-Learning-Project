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

preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])
resnet = models.resnet18(pretrained=True)
newmodel = torch.nn.Sequential(*(list(resnet.children())[:-1]))
# filename = '../test_images\\fef8e645d030.png'
# id = filename[15:len(filename)-4]
# print(id)

image_list = {}
for filename in glob.glob('../train_images/*.png'): 
    im=Image.open(filename)
    input = preprocessing(im)
    input_batch = input.unsqueeze(0)
    # if torch.cuda.is_available():
    #     input = input_batch.to('cuda')
    #     newmodel.to('cuda')
    with torch.no_grad():
        output = newmodel(input_batch)
    id = filename[15:len(filename)-4]
    image_list[filename] = output
#print(image_list)

#for i in l:
    # new_image = transforms.ToPILImage(i)
    # new_image.save(f"../new_test_images")



# resnet = torchvision.models.resnet18(pretrained=True)
# newmodel = torch.nn.Sequential(*(list(resnet.children())[:-1]))
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# newmodel = newmodel.to(device) # on gpu
# #inputs, labels = data[0].to(device), data[1].to(device) 
# print(newmodel)
# # newmodel.train()
# l = list(imread_collection("../train_images/*.png"))
training_set = pd.read_csv("../train.csv")
#delimiter=",", skip_header=1, dtype=[("id_code", np.string_), ("diagnosis", int)]
for i in range(len(training_set)):
    dst = Path(f"../trainingscenarios/{i:04d}.pkl")
    dst.write_bytes(pickle.dumps(dict(id_code=training_set.id_code[i], diagnosis=training_set.diagnosis[i], features=image_list[training_set.id_code[i]])))
    # np.savez(f"../trainingscenarios/{i:04d}.npz", id_code=training_set.id_code[i], diagnosis=training_set.diagnosis[i], features=None)
    
# for i in sorted(Path("../train_images").iterdir()):
    
exit()
imagepaths = sorted(Path("../images").glob("*.jpg"))
imagepaths.insert(9, imagepaths.pop(1))
imagepaths.insert(19, imagepaths.pop(11))
for i,p in enumerate(imagepaths):
    #text = Path(f"{i}.csv").read_text()
    arr = np.genfromtxt(f"{i}.csv", delimiter="\n")
    print(p,iio.imread(p).shape,median(arr))

    