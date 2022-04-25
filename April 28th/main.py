import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

def apply(kernel, image):
    h_k = len(kernel)
    w_k = len(kernel[0])
    output = np.copy(image)
    for y_i in range (len(image[0])):
        for x_i in range (len(image[0][0])):
            for dy in range(int(-(h_k-1)/2), int((h_k-1)/2) + 1):
                for dx in range(int(-(w_k-1)/2),int((w_k-1)/2) + 1):
                    for c in range(3):
                        output[y_i, x_i] += image[y_i + dy][x_i + dx][c] * kernel[c][int((h_k-1)/2) + dy][int((w_k-1)/2)+ dx]
    return output

def rectify(arr):
    for i in range(len(arr)):
        if (arr[i] < 0):
            arr[i] = 0

kernel =[[ [ -1, 0, 1 ],
        [ -2, 0, 2 ],
        [ -1, 0, 1 ]], [ [ -1, 0, 1 ],
        [ -2, 0, 2 ],
        [ -1, 0, 1 ]], [ [ -1, 0, 1 ],
        [ -2, 0, 2 ],
        [ -1, 0, 1 ]] ]
kernel = np.array(kernel)
print(kernel)
cat = iio.imread('imageio:chelsea.png')
print(cat.shape)
img = np.array(cat)
plt.imshow(img)
# print(img)
#new = np.dot(img, kernel)
new = apply(kernel, img)
print (new)
im = plt.imshow(new)


