import Augmentor
from pathlib import Path
import glob
from PIL import Image

p = Augmentor.Pipeline("../train_images/")
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.flip_left_right(probability=0.5)
p.sample(10000)
exit()
for filename in glob.glob('../tainr_images/*.png'): 
    im=Image.open(filename)

     
