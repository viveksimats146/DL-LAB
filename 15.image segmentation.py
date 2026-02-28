import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\vivek\Pictures\IMG_20250522_104404.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pixels = np.float32(img.reshape((-1,3)))

_, labels, centers = cv2.kmeans(
    pixels, 3, None,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.2),
    10, cv2.KMEANS_RANDOM_CENTERS)

segmented = np.uint8(centers)[labels.flatten()].reshape(img.shape)

plt.subplot(121), plt.imshow(img), plt.title("Original"), plt.axis('off')
plt.subplot(122), plt.imshow(segmented), plt.title("Segmented"), plt.axis('off')
plt.show()
