import numpy as np
import tensorflow as tf

import vgg16
from utils import utils

img1 = utils.load_image("./tmp/test_data/tiger.jpeg")
img2 = utils.load_image("./tmp/test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    # with tf.device('/cpu:0'):
    images = tf.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = vgg16.Vgg16('./checkpoints/vgg16.npy')
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    print(prob)
    utils.print_prob(prob[0], './tmp/test_data/synset.txt')
    utils.print_prob(prob[1], './tmp/test_data/synset.txt')
