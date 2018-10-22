# -*- coding: UTF-8 -*-

# 把mnist数据集转成图片做测试，图片更为通用

import cv2
from keras.datasets import mnist

import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

for i in range(0, 60000):  # 迭代 0 到 59999 之间的数字
    fileName = "/data/datasets/mnist_pics/mnist_train/" + str(Y_train[i]) + "-" + str(i) + ".png"
    cv2.imwrite(fileName, X_train[i])

for i in range(0, 10000):  # 迭代 0 到 9999 之间的数字
    fileName = "/data/datasets/mnist_pics/mnist_test/" + str(Y_test[i]) + "-" + str(i) + ".png"
    cv2.imwrite(fileName, X_test[i])
