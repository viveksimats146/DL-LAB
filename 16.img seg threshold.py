import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image from your PC   
img = cv2.imread(r"C:\Users\vivek\Pictures\IMG_20250522_104404.jpg")
# Keep image in same folder OR give full path

if img is None:
    print("Error: Image not found")
else:
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu Threshold
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphological Closing
    kernel = np.ones((2,2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 2)
    # Dilation
    dilation = cv2.dilate(closing, kernel, 3)
    # Display Results
    titles = ["Original", "Gray", "Threshold", "Closing", "Dilation"]
    images = [img_rgb, gray, thresh, closing, dilation]

    for i in range(5):
        plt.subplot(2,3,i+1)
        plt.imshow(images[i], cmap='gray' if i!=0 else None)
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Save result
    cv2.imwrite("dilation.jpg", dilation)
