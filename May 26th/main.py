import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.transform import rescale, resize, AffineTransform
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
'''
9x9 kernel, 100x100 image, 100*100*81 operations, about 1 million, 10 imgs/s
'''
#print(kernel)
l = [iio.imread(os.path.join("../images/","stop-sign-10.jpg")) / 255.0]
# l.append(iio.imread(os.path.join("../images/","no-stop-sign.jpg")))
# l.append(iio.imread(os.path.join("../images/","no-stop-sign2.jpg")))
# l.append(iio.imread(os.path.join("../images/","no-stop-sign3.jpg")))
# l.append(iio.imread(os.path.join("../images/","no-stop-sign4.jpg")))
# l.append(iio.imread(os.path.join("../images/","no-stop-sign5.jpg")))
# l.append(iio.imread(os.path.join("../images/","no-stop-sign6.jpg")))
# l.append(iio.imread(os.path.join("../images/","no-stop-sign7.jpg")))
# l.append(iio.imread(os.path.join("../images/","no-stop-sign8.webp")))
# l.append(iio.imread(os.path.join("../images/","no-stop-sign9.png")))
# l.append(iio.imread(os.path.join("../images/","no-stop-sign10.webp")))
# l.append(iio.imread(os.path.join("../images/","stop-sign-1.jpg")))
# l.append(iio.imread(os.path.join("../images/","stop-sign-2.jpg")))
# l.append(iio.imread(os.path.join("../images/","stop-sign-3.webp")))
# l.append(iio.imread(os.path.join("../images/","stop-sign-4.webp")))
# l.append(iio.imread(os.path.join("../images/","stop-sign-5.jpg")))
# l.append(iio.imread(os.path.join("../images/","stop-sign-6.jpg")))
# l.append(iio.imread(os.path.join("../images/","stop-sign-7.jpg")))
# l.append(iio.imread(os.path.join("../images/","stop-sign-8.jpg")))
# l.append(iio.imread(os.path.join("../images/","stop-sign-9.jpg")))
# l.append(iio.imread(os.path.join("../images/","stop-sign-10.jpg")))

stopsignkernel = iio.imread(os.path.join("../images/","stop-sign-kernel2.png"))/255.0
stopsignkernel = resize(stopsignkernel, (19,19))
localsumkernel = np.ones_like(stopsignkernel)
index = 2
# print(stopsignkernel.dtype)
# print(l[11].dtype)
#kernel_f = rescale(stopsignkernel, j*0.25)
# for z in range(0, 10):
plt.figure(figsize=(10, 8))
plt.subplot(5,1,1)
#for j in range(1, 5):
img = rescale(l[0], (0.9, 0.9, 1))
####
stopsignkernel[:,:,1] = 0
stopsignkernel[:,:,2] = 0
####
plt.imshow(np.uint8(img*255))
# for z in range(0, 10):
plt.subplot(5,1,index)
# kernel_f = AffineTransform(shear=z*5*np.pi/180)
convolved = convolve_rgb(img, stopsignkernel)
plt.imshow(convolved, cmap='gray')
plt.colorbar()
# index += 1
# curr_max = max(np.amax(convolved), 0)
# print(curr_max)
# threshold = np.max(bw(l[0]))
# print(threshold)
plt.subplot(5,1,3)
localbrightness = convolve_rgb(img,localsumkernel)
plt.imshow(localbrightness,cmap='gray')
plt.subplot(5,1,4)
score = convolved / (localbrightness + 0.01)
plt.imshow(score,cmap='gray')
plt.subplot(5,1,5)
mask = score > np.quantile(score, 0.998)
plt.imshow(mask, cmap='gray')
plt.tight_layout()
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
skyscraper = iio.imread('skyscraper.webp')
cat = iio.imread('imageio:chelsea.png') #numpy.ndarray, [row, col, color]
stop_sign = bw(iio.imread('stop-sign.jpg'))
stopsignkernel = iio.imread('stop-sign-png-14.png') #for later

# print(type(cat))
# print(cat[0,2,0]) # , separated means not slicing
# print(cat[:,:,0]) #all red values
grey_cat = bw(cat)
skyscraper = bw(skyscraper)
#::-1 flips array, start, end, step size
filter_output = signal.convolve(vertical_kernel_c, stop_sign) 
plt.subplot(2,2,1) #index is 1-based
plt.imshow(filter_output)
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(signal.convolve(vertical_kernel_c, skyscraper))
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(signal.convolve(vertical_kernel_c, grey_cat))
plt.colorbar()

# plt.subplot(2,2,2)
# plt.imshow(convolve(vertical_kernel_a[:,::-1], grey_cat))
# plt.colorbar()

# plt.subplot(2,2,3)
# plt.imshow(convolve(vertical_kernel_a.transpose(), grey_cat)) #switch row and column, horizontal edge detector
# plt.colorbar()

# plt.subplot(2,2,4)
# plt.imshow(convolve(vertical_kernel_a[:,::-1].transpose(), grey_cat))
# plt.colorbar()

plt.show()
exit()
#r * 0.3 + g*0.6 + b*0.1
print(cat.shape)
img = np.array(cat)
plt.imshow(cat)
plt.show()
# print(img)
#new = np.dot(img, kernel)
# new = apply(kernel, img)
# print (new)
# im = plt.imshow(new)


