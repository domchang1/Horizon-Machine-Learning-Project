import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, rotate, zoom
from skimage.io import imread_collection
from skimage.transform import rescale, resize, AffineTransform, warp
import os

def convolve_rgb(image:np.ndarray, kernel:np.ndarray) -> np.ndarray:
    r_image = image[:,:,0]
    g_image = image[:,:,1]
    b_image = image[:,:,2]
    r_kernel= kernel[:,:,0]
    g_kernel = kernel[:,:,1]
    b_kernel = kernel[:,:,2]
    return convolve(r_image, r_kernel) + convolve(g_image, g_kernel) + convolve(b_image, b_kernel)


#venv\Scripts\activate.bat to activate venv !!

# def convolve(kernel:np.ndarray, image:np.ndarray) -> np.ndarray:
#     h_k,w_k = kernel.shape
#     output = np.zeros_like(image) #creates same shape but empty
#     h_i,w_i = image.shape
#     support_h = (h_k-1)//2
#     support_w = (w_k-1)//2
#     image = np.pad(image, (support_h, support_w))
#     for y_i in range (h_i):
#         for x_i in range (w_i):
#             for dy in range(-1*support_h, support_h + 1):
#                 for dx in range(-1*support_w, support_w + 1):
#                     output[y_i, x_i] += image[y_i + dy + support_h, x_i + dx + support_w] * kernel[support_h + dy,support_w + dx]
#     return output

def rectify(arr):
    for i in range(len(arr)):
        if (arr[i] < 0):
            arr[i] = 0

def bw(image):
    r = image[:,:,0] #coefficients correspond to affect of color, more measurements of brightness for green 
    g = image[:,:,1]
    b = image[:,:,2]
    return r*0.3 + g*0.6 + b*0.1

def get_kernelvariants(k):
    for r in [-30, 0, 30]:
        for s in [(1,1), (1,0.75), (0.75, 1)]:
            # yield rotate(k, angle=r)
            yield rotate(zoom(k,(*s,1)), angle=r)

            
stopsignkernel = iio.imread(os.path.join("../images/","stop-sign-kernel2.png"))/255.0
stopsignkernel = resize(stopsignkernel, (19,19))
#l = list("imread_collection("../images/*.jpg))
l = [iio.imread(os.path.join("../images/","stop-sign1.jpg"))/255.0,
 iio.imread(os.path.join("../images/","stop-sign5.jpg"))/255.0,
  iio.imread(os.path.join("../images/","no-stop-sign1.jpg"))/255.0,
   iio.imread(os.path.join("../images/","no-stop-sign5.jpg"))/255.0]
# plt.subplot(1,2,1)
# plt.imshow(stopsignkernel)
# plt.subplot(1,2,2)
# tform = AffineTransform(shear=10*5*np.pi/180)
# img = warp(stopsignkernel, tform)
# plt.imshow(img)
# plt.show()
# exit()
# transforms = [
#     AffineTransform(rotation=r, scale=s)
#      #for r in [-np.pi/6, 0, np.pi/6] 
#      for r in [0]
#      for s in [(1,1), (1,0.75), (0.75, 1)]]
rows = 6
cols = 8
# print(stopsignkernel.dtype)
# print(l[11].dtype)
#kernel_f = rescale(stopsignkernel, j*0.25)
for i in range(0, len(l)):
    maxscores = []
    for j in range(3, 8):
        plt.figure(figsize=(18, 18))
        plt.subplot(rows,cols,1)
        img = rescale(l[i], (0.2*j, 0.2*j, 1))
        ####
        # stopsignkernel[:,:,1] = 0
        # stopsignkernel[:,:,2] = 0
        ####
        plt.imshow(np.uint8(img*255))
        plt.axis("off")
        for z,transformedkernel in enumerate(get_kernelvariants(stopsignkernel)):
            #plt.subplot(rows,cols,2+z*5)
            # stopsignkernel = warp(stopsignkernel, tform)
            convolved = convolve_rgb(img, transformedkernel)
            #plt.imshow(np.clip(transformedkernel,0,1))
            #plt.axis("off")
            #plt.subplot(rows,cols,3+z*5)
            #plt.imshow(convolved, cmap='gray')
            #plt.axis("off")
            # curr_max = max(np.amax(convolved), 0)
            # print(f"current max: {curr_max}")
            #plt.subplot(rows,cols,4+z*5)
            localsumkernel = np.ones_like(stopsignkernel)
            localbrightness = convolve_rgb(img,localsumkernel) # 
            #plt.imshow(localbrightness,cmap='gray')
            #plt.axis("off")
            #plt.subplot(rows,cols,5+z*5)
            score = convolved / (localbrightness + 0.01)
            #plt.imshow(score,cmap='gray')
            #plt.axis("off")
            #plt.subplot(rows,cols,6+z*5)
            # mask = score > np.quantile(score, 0.999)
            mask = score > 0.63
            maxscores.append(np.amax(score))
            # print(f"max score: {np.amax(score)}")
            #plt.imshow(mask, cmap='gray')
            #plt.axis("off")
        print("Finished another plot")
        #plt.savefig(f"../outputs/{i}-{j}.png")
        #plt.close()
    print(sorted(maxscores))
    
    # if 1 in mask:
        #     print("There is a stop sign")
        # else:
        #     print("No stop sign")
        #plt.tight_layout()
# plt.show()
exit()
for i in range(0, len(l)):
    curr_max = 0
    img = l[i]
    for j in range(1, 5):
        kernel_f = rescale(stopsignkernel, j*0.25)
        # for z in range(0, 10):
        plt.subplot(10,10,index)
        # kernel_f = AffineTransform(shear=z*5*np.pi/180)
        convolved = bw(convolve(img, kernel_f))
        plt.imshow(convolved)
        plt.colorbar()
        index += 1
        curr_max = max(np.amax(convolved), curr_max)

    print(curr_max)
    threshold = np.max(bw(img))
    print(threshold)
        
plt.show()
exit()