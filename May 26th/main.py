import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
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


vertical_kernel_a = np.array([ #check for vertical lines
    [-1, 2, -1],
    [-1, 2, -1],
    [-1, 2, -1]
])

vertical_kernel_b = np.array([ #check for dark to bright
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

vertical_kernel_c = np.array([ #check for vertical lines but bigger
     [-1 ,-1, 4, -1, -1],
     [-1 ,-1, 4, -1, -1],
     [-1 ,-1, 4, -1, -1],
     [-1 ,-1, 4, -1, -1],
     [-1 ,-1, 4, -1, -1],
])

vertical_kernel_d = np.array([ #check for vertical lines but bigger and bigger values
     [-50 ,-50, 200, -50, -50], #[-2 ,-1, 6, -1, -2],
     [-50 ,-50, 200, -50, -50],
     [-50 ,-50, 200, -50, -50],
     [-50 ,-50, 200, -50, -50],
     [-50 ,-50, 200, -50, -50],
])
stopsignkernel = iio.imread(os.path.join("../images/","stop-sign-kernel2.png"))/255.0
stopsignkernel = resize(stopsignkernel, (19,19))
localsumkernel = np.ones_like(stopsignkernel)
l = list(imread_collection("../images/*.jpg"))
# plt.subplot(1,2,1)
# plt.imshow(stopsignkernel)
# plt.subplot(1,2,2)
# tform = AffineTransform(shear=10*5*np.pi/180)
# img = warp(stopsignkernel, tform)
# plt.imshow(img)
# plt.show()
# exit()

rows = 5
cols = 4
index = 1
# print(stopsignkernel.dtype)
# print(l[11].dtype)
#kernel_f = rescale(stopsignkernel, j*0.25)
# for z in range(0, 10):
for i in range(0, len(l)):
    # plt.subplot(rows,cols,index)
    # index += 1
    for j in range(3, 8):
        img = rescale(l[i], (0.2*j, 0.2*j, 1))
        ####
        # stopsignkernel[:,:,1] = 0
        # stopsignkernel[:,:,2] = 0
        ####
        # plt.imshow(np.uint8(img*255))
        for z in range(0, 10):
            # plt.subplot(rows,cols,index)
            # index += 1
            tform = AffineTransform(shear=z*5*np.pi/180)
            stopsignkernel = warp(stopsignkernel, tform)
            convolved = convolve_rgb(img, stopsignkernel)
            # plt.imshow(convolved, cmap='gray')
            # plt.colorbar()
            # index += 1
            # curr_max = max(np.amax(convolved), 0)
            # print(curr_max)
            # threshold = np.max(bw(l[0]))
            # print(threshold)
            # plt.subplot(rows,cols,index)
            # index += 1
            localbrightness = convolve_rgb(img,localsumkernel)
            # plt.imshow(localbrightness,cmap='gray')
            # plt.subplot(rows,cols,index)
            # index += 1
            score = convolved / (localbrightness + 0.01)
            # plt.imshow(score,cmap='gray')
            # plt.subplot(rows,cols,index)
            index += 1
            mask = score > np.quantile(score, 0.99999)
            # plt.imshow(mask, cmap='gray')
            if 1 in mask:
                print("There is a stop sign")
            else:
                print("No mask")
plt.show()
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