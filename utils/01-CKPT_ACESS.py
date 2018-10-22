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
import tensorflow as tf

isess = tf.InteractiveSession()

# init a checkpoint reader
# for v1 file, you should expecte a file name
# reader = tf.train.NewCheckpointReader('../checkpoints/deeplab_resnet.ckpt')

# With V2 saver format, the reader is not expecting a file name, but only the prefix of the filename.
reader = tf.train.NewCheckpointReader('./checkpoints/resnet_v2_101/resnet_v2_101.ckpt')
# reader = tf.train.NewCheckpointReader('/Users/izhangxm/Downloads/pix2pix-tensorflow-master 2/aaa/facades_1_256/pix2pix.model-1')

# {name:shaoe} dict
name_shape_dict = reader.get_variable_to_shape_map()

# get all variable name
names = name_shape_dict.keys()
for index, name in enumerate(names):
    var = tf.get_variable(name=name, shape=name_shape_dict[name])
    isess.run(var.assign(reader.get_tensor(name)))

    print "{}/{}".format(index, len(names)), name, name_shape_dict[name]

print('ok')

# get a tensor-data via tensorname
# a_data = reader.get_tensor(names[-1])
# print(names[-1]+' ----------------------')
# print(a_data)
