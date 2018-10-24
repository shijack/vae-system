# coding=utf-8
import os
import time

import cv2
import numpy as np
import tensorflow as tf

import nets.encoder_factory
from preprocessing.image_reader import ImageReader
from utils.utils import make_dir

# 只使用一个gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LATENT_DIM = 4096  # 跟sift+color保持一致:800
# HEIGHT, WIDTH, DEPTH = 144, 112, 3
# H1, W1, D1, D2, D3, D4 = 9, 7, 16, 32, 64, 128 #for original vae
HEIGHT, WIDTH, DEPTH = 224, 224, 3
H1, W1, D1, D2, D3, D4 = 14, 14, 16, 32, 64, 128

BATCH_SIZE, LEARNING_RATE, TRAIN_ITERS = 32, 1e-4, 1000000
EVAL_ROWS, EVAL_COLS, SAMPLES_PATH, EVAL_INTERVAL = 4, 4, './samples_vae_resnetv2_101', 500
MODEL_PATH, SAVE_INTERVAL = './model_vae_resnetv2_101', 500

IS_SCALE = True
IS_MIRROR = True
IS_CROP = False  # 是否随机crop
IS_SHUFFLE = True

# TODO this parameter how to set?
PARAMATER_KL_RATE = -0.5


def valid_the_input(i, lab_batch_r_f, train_data_, input_h, input_w, input_c):
    # 验证输入文件的正确性
    offset = 0
    batch_xs_input = train_data_[offset:(offset + EVAL_ROWS * EVAL_COLS), :]
    batch_xs_label = lab_batch_r_f[offset:(offset + EVAL_ROWS * EVAL_COLS), :]
    train_data_ = unpreprocess(batch_xs_input)
    lab_batch_r_f = unpreprocess(batch_xs_label)

    data = np.reshape(train_data_.astype(int), (EVAL_ROWS, EVAL_COLS, input_h, input_w, input_c))
    data = np.concatenate(np.concatenate(data, 1), 1)
    cv2.imwrite(SAMPLES_PATH + '/iter-input-train-%d.png' % i, data[:, :, ::-1])

    data = np.reshape(lab_batch_r_f.astype(int), (EVAL_ROWS, EVAL_COLS, input_h, input_w, input_c))
    data = np.concatenate(np.concatenate(data, 1), 1)
    cv2.imwrite(SAMPLES_PATH + '/iter-inut-labl-%d.png' % i, data[:, :, ::-1])


def assign_decay(orig_val, new_val, momentum, name):
    with tf.name_scope(name):
        scaled_diff = (1 - momentum) * (new_val - orig_val)

    return tf.assign_add(orig_val, scaled_diff)


def batch_norm(x, train_logical, decay, epsilon, scope=None, shift=True, scale=False):
    channels = x.get_shape()[-1]
    ndim = len(x.shape)

    with tf.variable_scope(scope):

        moving_m = tf.get_variable('mean', [channels], initializer=tf.zeros_initializer, trainable=False)
        moving_v = tf.get_variable('var', [channels], initializer=tf.ones_initializer, trainable=False)

        if train_logical == True:

            m, v = tf.nn.moments(x, range(ndim - 1))
            update_m = assign_decay(moving_m, m, decay, 'update_mean')
            update_v = assign_decay(moving_v, v, decay, 'update_var')

            with tf.control_dependencies([update_m, update_v]):
                output = (x - m) * tf.rsqrt(v + epsilon)

        else:
            m, v = moving_m, moving_v
            output = (x - m) * tf.rsqrt(v + epsilon)

        if scale:
            output *= tf.get_variable('gamma', [channels], initializer=tf.ones_initializer)

        if shift:
            output += tf.get_variable('beta', [channels], initializer=tf.zeros_initializer)

    return output


def lrelu(x, leak, name):
    return tf.maximum(x, leak * x, name=name)


def conv_layer(x, out_depth, kernel, strides, w_initializer, b_initializer, name):
    in_depth = x.get_shape()[3]

    with tf.name_scope(name):
        w = tf.get_variable('w', shape=[kernel[0], kernel[1], in_depth, out_depth], initializer=w_initializer)
        b = tf.get_variable('b', shape=[out_depth], initializer=b_initializer)

        conv = tf.nn.conv2d(x, filter=w, strides=[1, strides[0], strides[1], 1], padding="SAME", name='conv')
        conv = tf.add(conv, b, name='add')

    return conv


def conv_block(x, out_depth, train_logical, w_initializer, b_initializer, scope):
    with tf.variable_scope(scope):
        conv = conv_layer(x, out_depth, [5, 5], [2, 2], w_initializer, b_initializer, 'conv')
        bn = batch_norm(conv, train_logical=train_logical, epsilon=1e-5, decay=0.99, scope='bn')
        act = lrelu(bn, leak=0.2, name='act')

    return act


def fc_block(input, in_num, out_num, w_initializer, b_initializer, scope):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [in_num, out_num], dtype=tf.float32, initializer=w_initializer)
        b = tf.get_variable('b', [out_num], dtype=tf.float32, initializer=b_initializer)

        variable_summaries(w, scope)
        variable_summaries(b, scope)

        fc = tf.matmul(input, w, name='matmul')
        fc = tf.add(fc, b, name='add')

    return fc


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


# -------------------------------------------------------------
# -------------------------- decoder --------------------------
# -------------------------------------------------------------
#
def conv_transpose_layer(x, out_depth, kernel, strides, w_initializer, b_initializer, scope):
    with tf.name_scope(scope):
        in_shape = x.get_shape().as_list()
        in_batch = tf.shape(x)[0]
        in_height, in_width, in_depth = in_shape[1:]
        out_shape = [in_batch, in_height * strides[0], in_width * strides[1], out_depth]

        w = tf.get_variable('w', shape=[kernel[0], kernel[1], out_depth, in_depth], initializer=w_initializer)
        b = tf.get_variable('b', shape=[out_depth], initializer=b_initializer)

        conv = tf.nn.conv2d_transpose(x, filter=w, output_shape=out_shape, strides=[1, strides[0], strides[1], 1],
                                      padding='SAME', name='deconv')
        conv = tf.add(conv, b, name='add')

    return conv


def decoder_conv_block(x, depth, train_logical, w_initializer, b_initializer, scope, final=False):
    with tf.variable_scope(scope):

        conv = conv_transpose_layer(x, depth, kernel=[5, 5], strides=[2, 2], w_initializer=w_initializer,
                                    b_initializer=b_initializer, scope='conv')

        if final:
            act = tf.nn.sigmoid(conv, name='act')
        else:
            bn = batch_norm(conv, train_logical=train_logical, epsilon=1e-5, decay=0.99, scope='bn')
            act = tf.nn.relu(bn, name='act')

    return act


def decoder_fc_block(x, height, width, depth, train_logical, w_initializer, b_initializer, scope):
    latent_dim = x.get_shape()[1]

    with tf.variable_scope(scope):
        w = tf.get_variable('w', shape=[latent_dim, height * width * depth], dtype=tf.float32,
                            initializer=w_initializer)
        b = tf.get_variable('b', shape=[height * width * depth], dtype=tf.float32, initializer=b_initializer)
        flat_conv = tf.add(tf.matmul(x, w), b, name='flat_conv')

        conv = tf.reshape(flat_conv, shape=[-1, height, width, depth], name='conv')
        bn = batch_norm(conv, train_logical=train_logical, epsilon=1e-5, decay=0.99, scope='bn')
        act = tf.nn.relu(bn, name='act')

    return act


def decoder(input, train_logical):
    xavier_initializer_conv = tf.contrib.layers.xavier_initializer_conv2d()
    xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
    zeros_initializer = tf.zeros_initializer()

    act1 = decoder_fc_block(input, H1, W1, D4, train_logical, xavier_initializer_fc, zeros_initializer, 'fc1')

    act2 = decoder_conv_block(act1, D3, train_logical, xavier_initializer_conv, zeros_initializer, 'conv2')
    act3 = decoder_conv_block(act2, D2, train_logical, xavier_initializer_conv, zeros_initializer, 'conv3')
    act4 = decoder_conv_block(act3, D1, train_logical, xavier_initializer_conv, zeros_initializer, 'conv4')
    act5 = decoder_conv_block(act4, DEPTH, train_logical, xavier_initializer_conv, zeros_initializer, 'conv5',
                              final=True)

    return act5


# -------------------------------------------------------------------
# -------------------------- record reader --------------------------
# -------------------------------------------------------------------


def read_images_queue(data_list, batch_size, image_channels, label_channels, image_h, image_w, is_scale, is_mirror,
                      is_crop, is_shuffle, basenet):
    reader = ImageReader('', data_list, img_channels=image_channels,
                         label_channels=label_channels,
                         input_size=[image_h, image_w], random_scale=is_scale, random_mirror=is_mirror,
                         random_crop=is_crop,
                         shuffle=is_shuffle, basenet=basenet)
    img_name_batch, img_batch, label_batch = reader.dequeue(batch_size)
    return img_name_batch, img_batch, label_batch, reader.data_list_len


# -------------------------------------------------------------
# -------------------------- training --------------------------
# --------------------------------------------------------------


def train():
    make_dir(MODEL_PATH)
    make_dir(SAMPLES_PATH)

    with tf.name_scope('batch'):
        train_imgs_file = './data_trans_imgs_centos/image_list_train.txt'
        print 'the train imags txt is : ' + train_imgs_file
        img_name_batch, img_batch, label_batch, n_samples = read_images_queue(train_imgs_file, BATCH_SIZE,
                                                                              image_channels=DEPTH,
                                                                              label_channels=DEPTH, image_h=HEIGHT,
                                                                              image_w=WIDTH,
                                                                              is_scale=IS_SCALE,
                                                                              is_mirror=IS_MIRROR,
                                                                              is_crop=IS_CROP,
                                                                              is_shuffle=IS_SHUFFLE,
                                                                              basenet='resnetv2_101')

    with tf.variable_scope('encoder') as scope:
        x_input = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH], name='input_img')
        x = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH], name='target_img')
        with tf.name_scope('train'):
            latent_mean, latent_stddev = nets.encoder_factory.encoder(x_input, True, LATENT_DIM, D1, D2, D3, D4)
        scope.reuse_variables()
        with tf.name_scope('eval'):
            latent_mean_eval, latent_stddev_eval = nets.encoder_factory.encoder(x_input, False, LATENT_DIM, D1, D2, D3,
                                                                                D4)
    # latent_mean, latent_stddev = encoder_vgg16(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_vgg19(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_inceptionv1(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_inceptionv4(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_inception_resnetv2(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_resnetv2_152(x_input, latent_dim=LATENT_DIM)#参数过多，训练很慢
    # latent_mean, latent_stddev = nets.encoder_factory.encoder_resnetv2_101(x_input, latent_dim=LATENT_DIM)

    # add input 10 images to the tensorboard
    with tf.name_scope('input_reshape'):
        means = [123.68, 116.779, 103.939]
        num_channels = x_input.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        channels = tf.split(axis=3, num_or_size_splits=num_channels, value=x_input)
        for i in range(num_channels):
            channels[i] += means[i]
        image_shaped_input = tf.concat(axis=3, values=channels)
        tf.summary.image('input', image_shaped_input, 10)

    with tf.variable_scope('variance'):
        # todo 这里用laten_mean or add  ？？？待检测！！
        # latent_encoder只在图像重建和生成特征中使用。
        latent_encoder = tf.add(latent_mean_eval, latent_stddev_eval, name='latent_feature')
    with tf.name_scope('train'):
        random_normal = tf.random_normal([BATCH_SIZE, LATENT_DIM], 0.0, 1.0, dtype=tf.float32)
        latent_vec = latent_mean + tf.multiply(random_normal, latent_stddev)

    latent_sample = tf.placeholder(tf.float32, shape=[None, LATENT_DIM], name='latent_input')

    with tf.variable_scope('decoder') as scope:
        with tf.name_scope('train'):
            y = decoder(latent_vec, train_logical=True)
        scope.reuse_variables()
        with tf.name_scope('generate'):
            gen_image = decoder(latent_sample, train_logical=False)
        scope.reuse_variables()
        with tf.name_scope('reconstruct'):
            reconst_image = decoder(latent_encoder, train_logical=False)

    with tf.name_scope('loss'):
        kl_divergence = PARAMATER_KL_RATE * tf.reduce_sum(
            1 + 2 * latent_stddev - tf.square(latent_mean) - tf.exp(2 * latent_stddev))
        reconstruction_loss = tf.reduce_sum(tf.square(y - x))

    with tf.name_scope('optimizer'):
        vae_loss = reconstruction_loss + kl_divergence
        tf.summary.scalar('reconstruction_loss', reconstruction_loss)
        tf.summary.scalar('loss', vae_loss)

        # todo 获取可训练的var_list
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        var_list_train = g_vars + d_vars

        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(vae_loss, var_list_train=None)

    """ prepare  data """

    # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./tmp/vae_train', sess.graph)

        # init the encoder of inception with pretrained weights
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=0)
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        g_step = 0
        if ckpt and ckpt.model_checkpoint_path:
            print 'train continue!'
            saver.restore(sess, ckpt.model_checkpoint_path)
            g_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        else:
            print 'train from basenet weights'
            nets.encoder_factory.get_init_fn_resnetv2_101(sess, './checkpoints/resnet_v2_101/resnet_v2_101.ckpt')
            # encoder_factory.get_init_fn_inceptonv1(sess, './checkpoints/inception_v1.ckpt')
            # encoder_factory.get_init_fn_inceptonv4(sess, './checkpoints/inception_v4.ckpt')
            # encoder_factory.get_init_fn_incepton_resnetv2(sess, './checkpoints/inception_resnet_v2_2016_08_30.ckpt')
            # encoder_factory.get_init_fn_resnetv2_152(sess, './checkpoints/resnet_v2_152/resnet_v2_152.ckpt')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for i in range(g_step + 1 if g_step > 0 else 0, TRAIN_ITERS):
            img_name_batch_r, img_batch_r, lab_batch_r = sess.run([img_name_batch, img_batch, label_batch])

            # img_batch_r_f = img_batch_r / 255.0
            # lab_batch_r_f = lab_batch_r / 255.0
            # the image process has been done in image_reader.py
            img_batch_r_f = img_batch_r

            train_data_ = img_batch_r_f
            lab_batch_r_f = lab_batch_r
            # Random shuffling
            # np.random.shuffle(train_data_)

            start_time = time.time()

            if (i + 1) % SAVE_INTERVAL == 0:
                valid_the_input(i, lab_batch_r_f, train_data_, HEIGHT, WIDTH, DEPTH)

            _, summary, loss = sess.run([train_step, merged, vae_loss],
                                        feed_dict={x_input: train_data_, x: lab_batch_r_f})
            train_writer.add_summary(summary, i)

            end_time = time.time()
            print('iter: %d, loss: %f, time: %f' % (i, loss, end_time - start_time))

            if (i + 1) % SAVE_INTERVAL == 0:
                saver.save(sess, MODEL_PATH + '/vae', global_step=i + 1)
                offset = 0
                batch_xs_input = train_data_[offset:(offset + EVAL_ROWS * EVAL_COLS), :]

                data = sess.run(reconst_image, feed_dict={x_input: batch_xs_input})

                data = unpreprocess(data)

                data = np.reshape(data.astype(int), (EVAL_ROWS, EVAL_COLS, HEIGHT, WIDTH, DEPTH))
                data = np.concatenate(np.concatenate(data, 1), 1)
                cv2.imwrite(SAMPLES_PATH + '/iter-recon-new-' + str(i) + '.png', data[:, :, ::-1])

            if (i + 1) % EVAL_INTERVAL == 0:
                latent_random = np.random.normal(0.0, 1.0, size=[EVAL_ROWS * EVAL_COLS, LATENT_DIM]).astype(np.float32)
                data = sess.run(gen_image, feed_dict={latent_sample: latent_random})

                data = unpreprocess(data)
                # data = data * 255.

                data = np.reshape(data.astype(int), (EVAL_ROWS, EVAL_COLS, HEIGHT, WIDTH, DEPTH))
                data = np.concatenate(np.concatenate(data, 1), 1)
                cv2.imwrite(SAMPLES_PATH + '/iter-genera-' + str(i) + '.png', data[:, :, ::-1])

        saver.save(sess, MODEL_PATH + '/vae', global_step=TRAIN_ITERS)
        train_writer.close()


def unpreprocess(data, basenet='vgg'):
    # todo 处理不同basenet情况下,重建或者恢复的图像乱问题。
    if 'vgg' or 'resnet' in basenet:
        means = [123.68, 116.779, 103.939]
        data[..., 0] += means[0]
        data[..., 1] += means[1]
        data[..., 2] += means[2]
    else:
        data = None
        raise Exception
    return data


if __name__ == "__main__":
    train()
