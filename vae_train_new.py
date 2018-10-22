# coding=utf-8
import time

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import slim

# the encoder basenet,vgg16 and vgg19 use .npy file else use tf.slim
import vgg16
import vgg19
from image_reader import ImageReader
from nets import inception, resnet_v2

# 只使用一个gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LATENT_DIM = 800  # 跟sift+color保持一致
# HEIGHT, WIDTH, DEPTH = 144, 112, 3
# H1, W1, D1, D2, D3, D4 = 9, 7, 16, 32, 64, 128 #for original vae
HEIGHT, WIDTH, DEPTH = 224, 224, 3
H1, W1, D1, D2, D3, D4 = 14, 14, 16, 32, 64, 128

BATCH_SIZE, LEARNING_RATE, TRAIN_ITERS = 32, 1e-4, 1000000
EVAL_ROWS, EVAL_COLS, SAMPLES_PATH, EVAL_INTERVAL = 4, 4, './samples_vae_resnetv2_101', 1000
MODEL_PATH, SAVE_INTERVAL = './model_vae_resnetv2_101', 1000

is_scale = True
is_mirror = True
is_crop = False  # 是否随机crop
is_shuffle = True


# -------------------------------------------------------------
# -------------------------- encoder --------------------------
# -------------------------------------------------------------

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

        fc = tf.matmul(input, w, name='matmul')
        fc = tf.add(fc, b, name='add')

    return fc


def encoder(input, train_logical, latent_dim):
    xavier_initializer_conv = tf.contrib.layers.xavier_initializer_conv2d()
    xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
    zeros_initializer = tf.zeros_initializer()

    act1 = conv_block(input, D1, train_logical, xavier_initializer_conv, zeros_initializer, 'conv1')
    act2 = conv_block(act1, D2, train_logical, xavier_initializer_conv, zeros_initializer, 'conv2')
    act3 = conv_block(act2, D3, train_logical, xavier_initializer_conv, zeros_initializer, 'conv3')
    act4 = conv_block(act3, D4, train_logical, xavier_initializer_conv, zeros_initializer, 'conv4')

    act4_num = int(np.prod(act4.get_shape()[1:]))
    act4_flat = tf.reshape(act4, [-1, act4_num])

    mean = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'mean')
    stddev = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'stddev')

    return mean, stddev


def encoder_vgg16(input, latent_dim):
    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        act4 = vgg.build_as_encoder(input)

    xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
    zeros_initializer = tf.zeros_initializer()

    act4_num = int(np.prod(act4.get_shape()[1:]))
    act4_flat = tf.reshape(act4, [-1, act4_num])

    mean = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'mean')
    stddev = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'stddev')

    return mean, stddev


def encoder_vgg19(input, latent_dim):
    vgg = vgg19.Vgg19()
    with tf.name_scope("content_vgg"):
        act4 = vgg.build_as_encoder(input)

    xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
    zeros_initializer = tf.zeros_initializer()

    act4_num = int(np.prod(act4.get_shape()[1:]))
    act4_flat = tf.reshape(act4, [-1, act4_num])

    mean = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'mean')
    stddev = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'stddev')

    return mean, stddev


def encoder_inceptionv1(input, latent_dim):
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        act4, _ = inception.inception_v1(input, num_classes=0, is_training=True)
    xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
    zeros_initializer = tf.zeros_initializer()

    act4_num = int(np.prod(act4.get_shape()[1:]))
    act4_flat = tf.reshape(act4, [-1, act4_num])

    mean = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'mean')
    stddev = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'stddev')

    return mean, stddev


def encoder_inceptionv4(input, latent_dim):
    with slim.arg_scope(inception.inception_v4_arg_scope()):
        act4, _ = inception.inception_v4(input, num_classes=0, is_training=True)
    xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
    zeros_initializer = tf.zeros_initializer()

    act4_num = int(np.prod(act4.get_shape()[1:]))
    act4_flat = tf.reshape(act4, [-1, act4_num])

    mean = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'mean')
    stddev = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'stddev')

    return mean, stddev


def encoder_inception_resnetv2(input, latent_dim):
    with slim.arg_scope(inception.inception_v4_arg_scope()):
        act4, _ = inception.inception_resnet_v2(input, num_classes=0, is_training=True)
    xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
    zeros_initializer = tf.zeros_initializer()

    act4_num = int(np.prod(act4.get_shape()[1:]))
    act4_flat = tf.reshape(act4, [-1, act4_num])

    mean = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'mean')
    stddev = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'stddev')

    return mean, stddev


def encoder_resnetv2_152(input, latent_dim):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        act4, _ = resnet_v2.resnet_v2_152(input, num_classes=0, is_training=True)
    xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
    zeros_initializer = tf.zeros_initializer()

    act4_num = int(np.prod(act4.get_shape()[1:]))
    act4_flat = tf.reshape(act4, [-1, act4_num])

    mean = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'mean')
    stddev = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'stddev')

    return mean, stddev


def encoder_resnetv2_101(input, latent_dim):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        act4, _ = resnet_v2.resnet_v2_101(input, num_classes=0, is_training=True)
    xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
    zeros_initializer = tf.zeros_initializer()

    act4_num = int(np.prod(act4.get_shape()[1:]))
    act4_flat = tf.reshape(act4, [-1, act4_num])

    mean = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'mean')
    stddev = fc_block(act4_flat, act4_num, latent_dim, xavier_initializer_fc, zeros_initializer, 'stddev')

    return mean, stddev


def get_init_fn_inceptonv1(sess, checkpoint_file):
    '''
    restore the bias and weights for the basenet of tf.slim
    :param sess:
    :param checkpoint_file:
    :return:
    '''
    resotre_var_global = {}
    for v in tf.global_variables():
        if 'InceptionV1' in v.name:
            resotre_var_global[v.name.split(':')[0]] = v
    # print resotre_var_global.keys()
    checkpoint_exclude_scopes = ["global_step", "InceptionV1/Logits", "InceptionV1/AuxLogits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    reader = tf.train.NewCheckpointReader(checkpoint_file)
    name_shape_dict = reader.get_variable_to_shape_map()
    names = name_shape_dict.keys()

    variables_to_restore = []
    for var in names:
        for exclusion in exclusions:
            if var.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    for index, name in enumerate(variables_to_restore):
        try:
            var = resotre_var_global[name]
            sess.run(var.assign(reader.get_tensor(name)))
        except:
            print('can not restore var: ' + name)

        print "{}/{}".format(index, len(variables_to_restore)), name, name_shape_dict[name]


def get_init_fn_resnetv2_101(sess, checkpoint_file):
    '''
    restore the bias and weights for the basenet of tf.slim
    :param sess:
    :param checkpoint_file:
    :return:
    '''
    resotre_var_global = {}
    for v in tf.global_variables():
        if 'resnet_v2_101' in v.name:
            resotre_var_global[v.name.split(':')[0]] = v
    # print resotre_var_global.keys()
    checkpoint_exclude_scopes = ["global_step", "resnet_v2_101/Logits", "resnet_v2_101/AuxLogits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    reader = tf.train.NewCheckpointReader(checkpoint_file)
    name_shape_dict = reader.get_variable_to_shape_map()
    names = name_shape_dict.keys()

    variables_to_restore = []
    for var in names:
        for exclusion in exclusions:
            if var.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    for index, name in enumerate(variables_to_restore):
        try:
            var = resotre_var_global[name]
            sess.run(var.assign(reader.get_tensor(name)))
        except:
            print('can not restore var: ' + name)

        print "{}/{}".format(index, len(variables_to_restore)), name, name_shape_dict[name]


def get_init_fn_resnetv2_152(sess, checkpoint_file):
    '''
    restore the bias and weights for the basenet of tf.slim
    :param sess:
    :param checkpoint_file:
    :return:
    '''
    resotre_var_global = {}
    for v in tf.global_variables():
        if 'resnet_v2_152' in v.name:
            resotre_var_global[v.name.split(':')[0]] = v
    # print resotre_var_global.keys()
    checkpoint_exclude_scopes = ["global_step", "resnet_v2_152/Logits", "resnet_v2_152/AuxLogits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    reader = tf.train.NewCheckpointReader(checkpoint_file)
    name_shape_dict = reader.get_variable_to_shape_map()
    names = name_shape_dict.keys()

    variables_to_restore = []
    for var in names:
        for exclusion in exclusions:
            if var.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    for index, name in enumerate(variables_to_restore):
        try:
            var = resotre_var_global[name]
            sess.run(var.assign(reader.get_tensor(name)))
        except:
            print('can not restore var: ' + name)

        print "{}/{}".format(index, len(variables_to_restore)), name, name_shape_dict[name]


def get_init_fn_inceptonv4(sess, checkpoint_file):
    '''
    restore the bias and weights for the basenet of tf.slim
    :param sess:
    :param checkpoint_file:
    :return:
    '''
    resotre_var_global = {}
    for v in tf.global_variables():
        if 'InceptionV4' in v.name:
            resotre_var_global[v.name.split(':')[0]] = v
    # print resotre_var_global.keys()
    checkpoint_exclude_scopes = ["global_step", "InceptionV4/Logits", "InceptionV4/AuxLogits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    reader = tf.train.NewCheckpointReader(checkpoint_file)
    name_shape_dict = reader.get_variable_to_shape_map()
    names = name_shape_dict.keys()

    variables_to_restore = []
    for var in names:
        for exclusion in exclusions:
            if var.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    for index, name in enumerate(variables_to_restore):
        try:
            var = resotre_var_global[name]
            sess.run(var.assign(reader.get_tensor(name)))
        except:
            print('can not restore var: ' + name)

        print "{}/{}".format(index, len(variables_to_restore)), name, name_shape_dict[name]


def get_init_fn_incepton_resnetv2(sess, checkpoint_file):
    '''
    restore the bias and weights for the basenet of tf.slim
    :param sess:
    :param checkpoint_file:
    :return:
    '''
    resotre_var_global = {}
    for v in tf.global_variables():
        if 'InceptionResnetV2' in v.name:
            resotre_var_global[v.name.split(':')[0]] = v
    # print resotre_var_global.keys()
    checkpoint_exclude_scopes = ["global_step", "InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    reader = tf.train.NewCheckpointReader(checkpoint_file)
    name_shape_dict = reader.get_variable_to_shape_map()
    names = name_shape_dict.keys()

    variables_to_restore = []
    for var in names:
        for exclusion in exclusions:
            if var.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    for index, name in enumerate(variables_to_restore):
        try:
            var = resotre_var_global[name]
            sess.run(var.assign(reader.get_tensor(name)))
        except:
            print('can not restore var: ' + name)

        print "{}/{}".format(index, len(variables_to_restore)), name, name_shape_dict[name]


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
# -------------------------- utils --------------------------
# -------------------------------------------------------------------


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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


# -------------------------------------------------------------------------
# -------------------------- batch normalization --------------------------
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------




# -------------------------------------------------------------
# -------------------------- training --------------------------
# --------------------------------------------------------------


def train():
    with tf.name_scope('batch'):
        train_imgs_file = './data/image_list_train.txt'
        print 'the train imags txt is : ' + train_imgs_file
        img_name_batch, img_batch, label_batch, n_samples = read_images_queue(train_imgs_file, BATCH_SIZE,
                                                                              image_channels=DEPTH,
                                                                              label_channels=DEPTH, image_h=HEIGHT,
                                                                              image_w=WIDTH,
                                                                              is_scale=is_scale,
                                                                              is_mirror=is_mirror,
                                                                              is_crop=is_crop,
                                                                              is_shuffle=is_shuffle,
                                                                              basenet='resnetv2_101')

    with tf.variable_scope('encoder'):
        x_input = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH], name='input_img')
        x = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH], name='target_img')

        # latent_mean, latent_stddev = encoder(x_input, train_logical=True, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_vgg16(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_vgg19(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_inceptionv1(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_inceptionv4(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_inception_resnetv2(x_input, latent_dim=LATENT_DIM)
    # latent_mean, latent_stddev = encoder_resnetv2_152(x_input, latent_dim=LATENT_DIM)#参数过多，训练很慢
    latent_mean, latent_stddev = encoder_resnetv2_101(x_input, latent_dim=LATENT_DIM)

    with tf.variable_scope('variance'):
        # with tf.name_scope('train'):
        #     random_normal = tf.random_normal([BATCH_SIZE, LATENT_DIM], 0.0, 1.0, dtype=tf.float32)
        #     latent_vec = latent_mean + tf.multiply(random_normal, latent_stddev)
        # with tf.name_scope('reconstruct'):
        latent_vec = tf.add(latent_mean, latent_stddev, name='latent_feature')

    latent_sample = tf.placeholder(tf.float32, shape=[None, LATENT_DIM], name='latent_input')

    with tf.variable_scope('decoder') as scope:
        with tf.name_scope('train'):
            y = decoder(latent_vec, train_logical=True)
        scope.reuse_variables()
        with tf.name_scope('generate'):
            gen_image = decoder(latent_sample, train_logical=False)
        with tf.name_scope('reconstruct'):
            reconst_image = decoder(latent_vec, train_logical=False)

    with tf.name_scope('loss'):
        kl_divergence = -0.5 * tf.reduce_sum(1 + 2 * latent_stddev - tf.square(latent_mean) - tf.exp(2 * latent_stddev))
        reconstruction_loss = tf.reduce_sum(tf.square(y - x))

    with tf.name_scope('optimizer'):
        vae_loss = reconstruction_loss + kl_divergence
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(vae_loss, )

    """ prepare  data """

    # train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()

    total_batch = int(n_samples / BATCH_SIZE)
    min_tot_loss = 1e99

    # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # init the encoder of inception with pretrained weights
        sess.run(tf.global_variables_initializer())
        # get_init_fn_inceptonv1(sess, './checkpoints/inception_v1.ckpt')
        # get_init_fn_inceptonv4(sess, './checkpoints/inception_v4.ckpt')
        # get_init_fn_incepton_resnetv2(sess, './checkpoints/inception_resnet_v2_2016_08_30.ckpt')
        # get_init_fn_resnetv2_152(sess, './checkpoints/resnet_v2_152/resnet_v2_152.ckpt')

        saver = tf.train.Saver(max_to_keep=0)
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        g_step = 0
        if ckpt and ckpt.model_checkpoint_path:
            print 'train continue!'
            saver.restore(sess, ckpt.model_checkpoint_path)
            g_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        else:
            print 'train from basenet weights'
            get_init_fn_resnetv2_101(sess, './checkpoints/resnet_v2_101/resnet_v2_101.ckpt')
        # sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for i in range(g_step + 1 if g_step > 0 else 0, TRAIN_ITERS):
            img_name_batch_r, img_batch_r, lab_batch_r = sess.run([img_name_batch, img_batch, label_batch])

            # img_batch_r_f = img_batch_r / 255.0
            # lab_batch_r_f = lab_batch_r / 255.0
            # the image process has been done in image_reader.py
            img_batch_r_f = img_batch_r
            lab_batch_r_f = lab_batch_r
            train_data_ = img_batch_r_f
            # Random shuffling
            # np.random.shuffle(train_data_)

            start_time = time.time()
            make_dir(MODEL_PATH)
            make_dir(SAMPLES_PATH)
            # # 验证输入文件的正确性
            # data = np.reshape((train_data_ * 255).astype(int), (20, 20, HEIGHT, WIDTH, DEPTH))
            # data = np.concatenate(np.concatenate(data, 1), 1)
            # cv2.imwrite(SAMPLES_PATH + '/iter-x-ori-%d.png' % i, data[:,:,::-1])
            #
            # data = np.reshape((lab_batch_r_f * 255).astype(int), (20, 20, HEIGHT, WIDTH, DEPTH))
            # data = np.concatenate(np.concatenate(data, 1), 1)
            # cv2.imwrite(SAMPLES_PATH + '/iter-x-label-%d.png' % i, data[:, :, ::-1])

            _, loss = sess.run([train_step, vae_loss], feed_dict={x_input: train_data_, x: lab_batch_r_f})

            # Loop over all batches
            # for j in range(total_batch):
            #     # Compute the offset of the current minibatch in the data.
            #     offset = (j * BATCH_SIZE) % (n_samples)
            #     batch_xs_input = train_data_[offset:(offset + BATCH_SIZE), :]
            #
            #     # 验证输入文件的正确性
            #     # data = np.reshape((batch_xs_input*255).astype(int), (20, 20, HEIGHT, WIDTH, DEPTH))
            #     # data = np.concatenate(np.concatenate(data, 1), 1)
            #     # cv2.imwrite(SAMPLES_PATH + '/iter-x-%d.png' % j, data)
            #
            #     # run_time = time.time()
            #     _, loss = sess.run([train_step, vae_loss], feed_dict={x_input: batch_xs_input, x: batch_xs_input})
            #     # run_end_time = time.time()
            #     # print('run batch_%d ,time: %f' % (j, (run_end_time - run_time)))
            end_time = time.time()
            print('iter: %d, loss: %f, time: %f' % (i, loss, end_time - start_time))

            if (i + 1) % SAVE_INTERVAL == 0:
                saver.save(sess, MODEL_PATH + '/vae', global_step=i + 1)
                offset = 0
                batch_xs_input = train_data_[offset:(offset + EVAL_ROWS * EVAL_COLS), :]

                data = sess.run(y, feed_dict={x_input: batch_xs_input})
                data = np.reshape((data * 255).astype(int), (EVAL_ROWS, EVAL_COLS, HEIGHT, WIDTH, DEPTH))
                data = np.concatenate(np.concatenate(data, 1), 1)
                cv2.imwrite(SAMPLES_PATH + '/iter-recon-new-' + str(i) + '.png', data[:, :, ::-1])
            if (i + 1) % EVAL_INTERVAL == 0:
                latent_random = np.random.normal(0.0, 1.0, size=[EVAL_ROWS * EVAL_COLS, LATENT_DIM]).astype(np.float32)
                data = sess.run(gen_image, feed_dict={latent_sample: latent_random})
                data = np.reshape((data * 255).astype(int), (EVAL_ROWS, EVAL_COLS, HEIGHT, WIDTH, DEPTH))
                data = np.concatenate(np.concatenate(data, 1), 1)
                cv2.imwrite(SAMPLES_PATH + '/iter-genera-' + str(i) + '.png', data[:, :, ::-1])

        saver.save(sess, MODEL_PATH + '/vae', global_step=i + 1)


if __name__ == "__main__":
    train()
