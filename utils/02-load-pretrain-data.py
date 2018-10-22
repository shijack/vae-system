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
# reader = tf.train.NewCheckpointReader('../checkpoints/deeplab_resnet.ckpt')
reader = tf.train.NewCheckpointReader('/Users/izhangxm/Downloads/backup/deepvoc/deeplab_resnet_init.ckpt')

# {name:shaoe} dict
name_shape_dict = reader.get_variable_to_shape_map()

# get all variable name
names = name_shape_dict.keys()

# get a tensor-data via tensorname
a_data = reader.get_tensor(names[-1])
print(a_data[:5])

a_tensor_like_data = tf.get_variable('a_tensor_like_data', shape=a_data.shape, initializer=tf.ones_initializer)

isess.run(tf.global_variables_initializer())

tensor_data = isess.run(a_tensor_like_data)

print(tensor_data[:5])

# load data

isess.run(tensor_data.name)

tensor_data2 = isess.run(a_tensor_like_data)
print(tensor_data2[:5])

for varl in tf.global_variables():
    print(varl.name)

for varl in tf.local_variables():
    print(varl.name)
