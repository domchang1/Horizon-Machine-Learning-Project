from pathlib import Path
from statistics import median
import imageio.v3 as iio
import numpy as np
import torchvision
import torch

resnet = torchvision.models.resnet18(pretrained=True)
newmodel = torch.nn.Sequential(*(list(resnet.children())[:-1]))
print(newmodel)
exit()
imagepaths = sorted(Path("../images").glob("*.jpg"))
imagepaths.insert(9, imagepaths.pop(1))
imagepaths.insert(19, imagepaths.pop(11))
for i,p in enumerate(imagepaths):
    #text = Path(f"{i}.csv").read_text()
    arr = np.genfromtxt(f"{i}.csv", delimiter="\n")
    print(p,iio.imread(p).shape,median(arr))

    