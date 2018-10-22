# coding=utf-8
import matplotlib

matplotlib.use('Agg')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from image_reader import ImageReader
from siamense_model import *
import numpy as np

flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('train_iter', 50000, 'Total training iter')
flags.DEFINE_integer('step', 500, 'Save after ... iteration')


def read_images_queue_siamense(data_list, batch_size, image_channels, label_channels, image_h, image_w, is_scale,
                               is_mirror,
                               is_crop, is_shuffle, basenet):
    reader = ImageReader('', data_list, img_channels=image_channels,
                         label_channels=label_channels,
                         input_size=[image_h, image_w], random_scale=is_scale, random_mirror=is_mirror,
                         random_crop=is_crop,
                         shuffle=is_shuffle, basenet=basenet)
    img_name_batch, img_batch, label_batch = reader.dequeue(batch_size)
    return img_name_batch, img_batch, label_batch, reader.data_list_len


with tf.name_scope('batch'):
    train_imgs_file = './data/image_list_train.txt'
    print 'the train imags txt is : ' + train_imgs_file
    img_name_batch, img_batch, label_batch, n_samples = read_images_queue_siamense(train_imgs_file, 100,
                                                                                   image_channels=3,
                                                                                   label_channels=3,
                                                                                   image_h=224,
                                                                                   image_w=224,
                                                                                   is_scale=False,
                                                                                   is_mirror=True,
                                                                                   is_crop=False,
                                                                                   is_shuffle=False,
                                                                                   basenet='base_vgg')

left = tf.placeholder(tf.float32, [None, 224, 224, 3], name='left')
right = tf.placeholder(tf.float32, [None, 224, 224, 3], name='right')
with tf.name_scope("similarity"):
    label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
    label = tf.to_float(label)
margin = 0.2

left_output = net_vgg16(left, reuse=False)

right_output = net_vgg16(right, reuse=True)

loss = contrastive_loss(left_output, right_output, label, margin)

global_step = tf.Variable(0, trainable=False)

# starter_learning_rate = 0.0001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
# tf.scalar_summary('lr', learning_rate)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)

saver = tf.train.Saver(max_to_keep=0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # setup tensorboard
    tf.summary.scalar('step', global_step)
    tf.summary.scalar('loss', loss)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('train.log', sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # train iter
    for i in range(FLAGS.train_iter):

        img_name_batch_r, img_batch_r, lab_batch_r = sess.run([img_name_batch, img_batch, label_batch])

        b_l = img_batch_r

        tmp_a = lab_batch_r[:lab_batch_r.shape[0] / 2 + 1, ...]
        tmp_b = lab_batch_r[lab_batch_r.shape[0] - lab_batch_r.shape[0] / 2:, ...]

        tmp_sim_a = np.ones(tmp_a.shape[0])
        tmp_sim_b = np.zeros(tmp_b.shape[0])
        b_sim = np.concatenate((tmp_sim_a, tmp_sim_b), axis=0)

        mess_ok = False
        tmp_index = list(np.random.permutation(tmp_b.shape[0]))

        while not mess_ok and tmp_index:
            for i in range(lab_batch_r.shape[0]):
                # print i == tmp_index[i]
                if i == tmp_index[i]:
                    tmp_index = list(np.random.permutation(tmp_b.shape[0]))
                    break
            mess_ok = True
        print tmp_index
        tmp_b = tmp_b[tmp_index]

        b_r = np.concatenate((tmp_a, tmp_b), axis=0)

        _, l, summary_str = sess.run([train_step, loss, merged],
                                     feed_dict={left: b_l, right: b_r, label: b_sim})

        writer.add_summary(summary_str, i)
        print "\r#%d - Loss" % i, l

        if (i + 1) % FLAGS.step == 0:
            # generate test
            # feat = sess.run(left_output, feed_dict={left: test_im})

            # labels = mnist.test.labels
            # # plot result
            # f = plt.figure(figsize=(16, 9))
            # for j in range(10):
            #     plt.plot(feat[labels == j, 0].flatten(), feat[labels == j, 1].flatten(),
            #              '.', tmp_index=c[j], alpha=0.8)
            # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            # plt.savefig('img/%d.jpg' % (i + 1))
            saver.save(sess, "model/model.ckpt")
