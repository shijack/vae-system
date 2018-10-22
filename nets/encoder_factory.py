# coding=utf-8


import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from nets import inception, resnet_v2
# the encoder basenet,vgg16 and vgg19 use .npy file else use tf.slim
# the vgg16.py variables cannot trained!
from siamense_network import vgg16, vgg19
from vae_train_new import fc_block, conv_block


def encoder(input, train_logical, latent_dim, output_a1, output_a2, output_a3, output_a4):
    xavier_initializer_conv = tf.contrib.layers.xavier_initializer_conv2d()
    xavier_initializer_fc = tf.contrib.layers.xavier_initializer()
    zeros_initializer = tf.zeros_initializer()

    act1 = conv_block(input, output_a1, train_logical, xavier_initializer_conv, zeros_initializer, 'conv1')
    act2 = conv_block(act1, output_a2, train_logical, xavier_initializer_conv, zeros_initializer, 'conv2')
    act3 = conv_block(act2, output_a3, train_logical, xavier_initializer_conv, zeros_initializer, 'conv3')
    act4 = conv_block(act3, output_a4, train_logical, xavier_initializer_conv, zeros_initializer, 'conv4')

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
