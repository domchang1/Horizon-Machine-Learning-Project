import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
#venv\Scripts\activate.bat to activate venv !!

def convolve(kernel:np.ndarray, image:np.ndarray) -> np.ndarray:
    h_k,w_k = kernel.shape
    output = np.zeros_like(image) #creates same shape but empty
    h_i,w_i = image.shape
    support_h = (h_k-1)//2
    support_w = (w_k-1)//2
    image = np.pad(image, (support_h, support_w))
    for y_i in range (h_i):
        for x_i in range (w_i):
            for dy in range(-1*support_h, support_h + 1):
                for dx in range(-1*support_w, support_w + 1):
                    output[y_i, x_i] += image[y_i + dy + support_h, x_i + dx + support_w] * kernel[support_h + dy,support_w + dx]
    return output

def rectify(arr):
    for i in range(len(arr)):
        if (arr[i] < 0):
            arr[i] = 0

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
#print(kernel)
#cat = iio.imread('skyscraper.webp')
cat = iio.imread('imageio:chelsea.png') #numpy.ndarray, [row, col, color]
print(type(cat))
# print(cat[0,2,0]) # , separated means not slicing
# print(cat[:,:,0]) #all red values
r = cat[:,:,0]
g = cat[:,:,1]
b = cat[:,:,2] #coefficients correspond to affect of color, more measurements of brightness for green 
grey_cat = r*0.3 + g*0.6 + b*0.1

filter_output = convolve(vertical_kernel_a, grey_cat) #::-1 flips array, start, end, step size
plt.subplot(3,1,1) #index is 1-based
plt.imshow(filter_output)
plt.colorbar()

plt.subplot(3,1,2)
plt.imshow(convolve(vertical_kernel_c, grey_cat))
plt.colorbar()

plt.subplot(3,1,3)
plt.imshow(convolve(vertical_kernel_d, grey_cat))
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


