from pathlib import Path
from statistics import median
from skimage.io import imread_collection
from skimage.transform import resize
import imageio.v3 as iio
import numpy as np
import torchvision
import torch
from torchvision.utils import save_image

resnet = torchvision.models.resnet18(pretrained=True)
newmodel = torch.nn.Sequential(*(list(resnet.children())[:-1]))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
newmodel = newmodel.to(device) # on gpu
#inputs, labels = data[0].to(device), data[1].to(device) 
print(newmodel)
# newmodel.train()
l = list(imread_collection("../train_images/*.png"))
for x in len(l):
    l[x]= resize(l[x], (256,256))

exit()
imagepaths = sorted(Path("../images").glob("*.jpg"))
imagepaths.insert(9, imagepaths.pop(1))
imagepaths.insert(19, imagepaths.pop(11))
for i,p in enumerate(imagepaths):
    #text = Path(f"{i}.csv").read_text()
    arr = np.genfromtxt(f"{i}.csv", delimiter="\n")
    print(p,iio.imread(p).shape,median(arr))

    