from matplotlib import pyplot as plt 
import numpy as np 
import cv2
import sys
plt.style.use('ggplot')


chosen_image = 'moun_tains.jpg'
img = plt.imread('../sceneimages/{}'.format(chosen_image))
a = np.zeros_like(img)

print (img.max())

a[100:250, 50:150] = 200
a[300:450, 410:490] = 50
a = cv2.GaussianBlur(a, (27,27), 0)

plt.imshow(img, cmap='gray')
plt.imshow(a, alpha=0.7, cmap='inferno')
plt.xticks([])
plt.yticks([])
plt.show()