#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# Copyright 2017 izhangxm@gmail.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import tensorflow as tf

from preprocessing.image_reader import ImageReader
from utils.utils import get_all_files


def generate_data_txt_no_lable(file_name):
    data_list = get_all_files('/data/datasets/trans_imgs/')
    with open(file_name, 'w') as f:
        f.writelines('\n'.join(data_list))
    return data_list


def generate_data_txt(file_name):
    data_list = get_all_files('/data/datasets/trans_imgs/origin')
    random.shuffle(data_list)
    num_all = len(data_list)
    num_train = int(num_all * (7 / 10.0))
    num_valid = int(num_all * (2 / 10.0))
    data_list_train = data_list[:num_train]
    data_list_valid = data_list[num_train:num_train + num_valid]
    data_list_test = data_list[num_train + num_valid:num_all]

    data_set_train = set(data_list_train)
    data_set_valid = set(data_list_valid)
    data_set_test = set(data_list_test)

    dict_train = {}
    dict_valid = {}
    dict_test = {}

    data_list_all = get_all_files('/data/datasets/trans_imgs/')
    no_item_list = []
    for item_all in data_list_all:
        prefix_rkf = '/data/datasets/trans_imgs/origin/' + item_all.split('RKF')[0].split('/')[-1] + 'RKF.jpg'
        if prefix_rkf in data_set_train:
            if not dict_train.has_key(prefix_rkf):
                pre_list_train = []
                pre_list_train.append(item_all)
                dict_train[prefix_rkf] = pre_list_train
            else:
                dict_train[prefix_rkf].append(item_all)
        elif prefix_rkf in data_set_valid:
            if not dict_valid.has_key(prefix_rkf):
                pre_list_valid = []
                pre_list_valid.append(item_all)
                dict_valid[prefix_rkf] = pre_list_valid
            else:
                dict_valid[prefix_rkf].append(item_all)
        elif prefix_rkf in data_set_test:
            if not dict_test.has_key(prefix_rkf):
                pre_list_test = []
                pre_list_test.append(item_all)
                dict_test[prefix_rkf] = pre_list_test
            else:
                dict_test[prefix_rkf].append(item_all)
        else:
            no_item_list.append(item_all)

    with open(file_name.split('.txt')[0] + '_train.txt', 'w') as f:
        for (d, v) in dict_train.items():
            for item_v in v:
                f.write(item_v + ' ' + d + '\n')

    with open(file_name.split('.txt')[0] + '_valid.txt', 'w') as f:
        for (d, v) in dict_valid.items():
            for item_v in v:
                f.write(item_v + ' ' + d + '\n')

    with open(file_name.split('.txt')[0] + '_test.txt', 'w') as f:
        for (d, v) in dict_test.items():
            for item_v in v:
                f.write(item_v + ' ' + d + '\n')

    return data_list


def generate_data_txt_small_valid(file_img_valid, num_to_save=10000):
    tmp_img_list = []
    with open(file_img_valid, 'r') as f:
        tmp_img_list = f.readlines()
    random.shuffle(tmp_img_list)
    file_output = file_img_valid.split('.txt')[0] + '_small.txt'
    data_list = tmp_img_list[:num_to_save]
    with open(file_output, 'w') as f:
        f.writelines(data_list)
    return data_list


data_dir = ''
data_list = './data/image_list_train.txt'
# data_list = generate_data_txt('./data/image_list.txt')
# data_list = generate_data_txt_small_valid('./data/image_list_valid.txt')

batch_size = 2

image_channels = 3
label_channels = 3
image_h = 224  # 原始图片高
image_w = 224  # 原始图片宽

is_scale = True
is_mirror = True
is_crop = True  # 是否随机crop
is_shuffle = True

reader = ImageReader(data_dir, data_list, img_channels=image_channels,
                     label_channels=label_channels,
                     input_size=[image_h, image_w], random_scale=is_scale, random_mirror=is_mirror, random_crop=is_crop,
                     shuffle=is_shuffle, basenet='inception')
img_name_batch, img_batch, lab_batch = reader.dequeue(batch_size)

img_names = []
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # Start queue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # get input
    for i in range(50000):
        img_name_batch_r, img_batch_r, lab_batch_r = sess.run([img_name_batch, img_batch, lab_batch])
        img_names.append(img_name_batch_r)
        img_batch_r_f = img_batch_r
        if i > 0:
            print (img_names[i] == img_names[i - 1]).all()
