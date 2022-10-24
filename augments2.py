import random

import cv2
import numpy as np
from matplotlib import pyplot as plt


# SALT & PEPPER NOISE
def add_noise(img):
    row, col, ignore = img.shape
    number_of_pixels = random.randint(7000, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
    number_of_pixels = random.randint(7000, 10000)
    number_of_pixels = random.randint(7000, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0
    return img



def low_pass(img):
    kernel = np.ones((10, 10), np.float32) / 20
    dst = cv2.filter2D(img,-1,kernel)
    return dst

# HIGH PASS
def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma) + 127


def gaussianBlurring(img,kernalSizes):
    # apply a "Gaussian" blur to the image
    for (kX, kY) in kernalSizes:
        blurred = cv2.GaussianBlur(img, (kX, kY), 0)
        cv2.imshow("Gaussian ({}, {})".format(kX, kY), blurred)
        cv2.waitKey(0)


def median_blurring(img,low, mid, high):
    cv2.destroyAllWindows()
    for k in (low, mid, high):
        # apply a "median" blur to the image
        blurred = cv2.medianBlur(img, k)
        cv2.imshow("Median {}".format(k), blurred)
        cv2.waitKey(0)

# img = cv2.imread('filter.jpeg')
# kernelSizes = [(3, 3), (9, 9), (15, 15)]
# gaussianBlurring(img, kernelSizes)
#median_blurring(img, 5,10,15)


"""plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()"""
