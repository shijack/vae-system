# coding=utf-8

import numpy as np

import skimage.io

img = skimage.io.imread('tmp/test_data/tiger.jpeg')
data = img - 0.

data = np.expand_dims(data, 0)

means = [123.68, 116.779, 103.939]

data[..., 0] -= means[0]
data[..., 1] -= means[1]
data[..., 2] -= means[2]

print data
