import cv2
import numpy as np

image1 = "./unseg-output/img.png"
image2 = "./output/img.png"

image1 = cv2.imread(image1)
image2 = cv2.imread(image2)

err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
err /= float(image1.shape[0] * image1.shape[1])
print("Mean squared error between images:", err)
