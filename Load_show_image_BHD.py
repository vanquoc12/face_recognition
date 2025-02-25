import cv2 as cv
import numpy as np

img=cv.imread('E:\XLA_py\mobile_robot.jpg',cv.IMREAD_GRAYSCALE)
print('so chieu data', img.ndim)
print('the sum of pixel', img.size)
print('the number of pixel in each dimension', img.shape)
#imgae enhancement
#imgae processing
cv.imshow('Mobile robot',img)
cv.waitKey(0)
cv.destroyAllWindow()