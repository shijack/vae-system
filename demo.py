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

import os
import shutil

import tensorflow as tf

from preprocessing.image_reader import ImageReader
from utils.utils import get_all_files


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


def util_tf_input_queue(file_data_list):
    batch_size = 2
    image_channels = 3
    label_channels = 3
    image_h = 224  # 原始图片高
    image_w = 224  # 原始图片宽
    is_scale = True
    is_mirror = True
    is_crop = True  # 是否随机crop
    is_shuffle = True
    reader = ImageReader('', file_data_list, img_channels=image_channels,
                         label_channels=label_channels,
                         input_size=[image_h, image_w], random_scale=is_scale, random_mirror=is_mirror,
                         random_crop=is_crop,
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
                print (img_names[i] == img_names[i - 1])


def get_dissim_videoid(dir_gt):
    subdir_list = os.listdir(dir_gt)
    dict_query = {}
    for item_file in subdir_list:
        tmp_videoid = []
        dict_query[item_file.split('.')[0].split('_')[-1]] = tmp_videoid
        with open(os.path.join(dir_gt, item_file), 'r') as f:
            content_gt = f.readlines()
            for item_line in content_gt:
                if 'X' in item_line.upper():
                    tmp_videoid.append(item_line.split('\t')[0])

    print dict_query
    with open('/home/shihuijie/Desktop/ml/tmp/Video_List.txt', 'r') as f:
        videoid_kfdir_list = f.readlines()
    dict_gt_idkf = {}
    for item_kfdir in videoid_kfdir_list:
        tmp_kddir = item_kfdir.split('\t')
        dict_gt_idkf[tmp_kddir[0]] = tmp_kddir[3].split('.')[0]

    for i_query in dict_query:
        tmp_vids = dict_query[i_query]
        true_vids = []
        for item_vid in tmp_vids:
            true_vids.append(dict_gt_idkf[item_vid])
        dict_query[i_query] = true_vids
    print dict_query

    dir_copy_list = []
    dict_seed={}
    with open('/home/shihuijie/Desktop/ml/tmp/Seed.txt','r') as f:
        seed_list = f.readlines()
    for item_seed in seed_list:
        tmp_seed = item_seed.split('\t')
        dict_seed[tmp_seed[0].replace('*','')]=dict_gt_idkf[tmp_seed[1].strip()]

    for k_query,value_query in dict_query.items():
        dir_copy_list.append(dict_seed[k_query])
        for item_tmp_vid in value_query:
            dir_copy_list.append(item_tmp_vid)
    print dir_copy_list
    # qu chu chongfu de shipin
    dir_copy_list = list(set(dir_copy_list))
    re_suce=[]
    re_fail=[]
    for item_copy in dir_copy_list:
        try:
            shutil.copytree(os.path.join('/home/shihuijie/Desktop/ml/tmp/datasets_ccweb_keyframe',item_copy),os.path.join('/home/shihuijie/Desktop/kf_ccweb_to_train',item_copy))
            re_suce.append(os.path.join('/home/shihuijie/Desktop/ml/tmp/datasets_ccweb_keyframe',item_copy))
        except Exception as e:
            print e
            re_fail.append(os.path.join('/home/shihuijie/Desktop/ml/tmp/datasets_ccweb_keyframe',item_copy))

    print re_fail
    #can not find /home/shihuijie/Desktop/ml/tmp/datasets_ccweb_keyframe/8_174_Y

if __name__ == '__main__':
    # data_list = generate_data_txt('./data/image_list.txt')
    # data_list = generate_data_txt_small_valid('./data/image_list_valid.txt')
    # util_tf_input_queue('./data/image_list_train.txt')
    get_dissim_videoid('/home/shihuijie/Desktop/ml/tmp/gt_ccweb')
