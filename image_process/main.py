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
import fnmatch
from multiprocessing import Process, Lock
from multiprocessing import Queue as ProcessQueen
import threading
import os
import subprocess as sp
from multiprocessing.sharedctypes import Value, Array
import sys

from PIL import Image
import cv2

import image_process

# 下载配置项===============================================================================================
process_num = 1  # 进程数
thread_num = 1  # 每个进程开启的下载的线程数量
# todo 遍历所有要处理的图片
datafile_list = [os.path.join(subdir, f)
                 for subdir, dirs, files in
                 os.walk('/Users/shijack/Desktop/keyframe/keyframe/d2015b438b70f022967713d6f977ebc67a16839e')
                 for f in fnmatch.filter(files, '*.jpg')]
dest_dir = '/Users/shijack/Desktop/trans_imgs'
dest_dir_cont_up = dest_dir + '/contrast_up/'
dest_dir_cont_down = dest_dir + '/contrast_down/'
dest_dir_crop = dest_dir + '/crop/'
dest_dir__blur = dest_dir + '/blur/'
dest_dir_letterbox = dest_dir + '/letterbox/'
dest_dir_text = dest_dir + '/add_text/'
dest_dir_zoom_up = dest_dir + '/zoom_up/'
dest_dir_zoom_down = dest_dir + '/zoom_down/'
dest_dir_flip = dest_dir + '/flip/'
dest_dir_shift = dest_dir + '/shift/'
dest_dir_rot_45 = dest_dir + '/rot_45/'

err_log_file_path = './trans_imgs/error.log'


# ===============================================================================================


def init_dir():
    if not os.path.exists(os.path.dirname(err_log_file_path)):
        os.makedirs(os.path.dirname(err_log_file_path))
    if not os.path.exists(os.path.join(dest_dir)):
        os.makedirs(os.path.join(dest_dir))
    if not os.path.exists(os.path.join(dest_dir_cont_up)):
        os.makedirs(os.path.join(dest_dir_cont_up))
    if not os.path.exists(os.path.join(dest_dir_cont_down)):
        os.makedirs(os.path.join(dest_dir_cont_down))
    if not os.path.exists(os.path.join(dest_dir_crop)):
        os.makedirs(os.path.join(dest_dir_crop))
    if not os.path.exists(os.path.join(dest_dir__blur)):
        os.makedirs(os.path.join(dest_dir__blur))
    if not os.path.exists(os.path.join(dest_dir_letterbox)):
        os.makedirs(os.path.join(dest_dir_letterbox))
    if not os.path.exists(os.path.join(dest_dir_text)):
        os.makedirs(os.path.join(dest_dir_text))
    if not os.path.exists(os.path.join(dest_dir_zoom_up)):
        os.makedirs(os.path.join(dest_dir_zoom_up))
    if not os.path.exists(os.path.join(dest_dir_zoom_down)):
        os.makedirs(os.path.join(dest_dir_zoom_down))
    if not os.path.exists(os.path.join(dest_dir_flip)):
        os.makedirs(os.path.join(dest_dir_flip))
    if not os.path.exists(os.path.join(dest_dir_shift)):
        os.makedirs(os.path.join(dest_dir_shift))
    if not os.path.exists(os.path.join(dest_dir_rot_45)):
        os.makedirs(os.path.join(dest_dir_rot_45))


def finish_display(task_info, update_box_n=False):
    screen_print_lock.acquire()
    finished_n.value += 1
    print(" %s/%s:%s" % (finished_n.value, total_num, task_info))
    screen_print_lock.release()


# ------------------------------------------------------------------------------------------
screen_print_lock = Lock()  # 屏幕输出锁
err_file_lock = Lock()  # 错误日志写入锁
processQueen = ProcessQueen(process_num * thread_num * 10)  # 任务队列

# 初始化文件夹
init_dir()
error_files_fp = open(err_log_file_path, 'a+')

finished_n = Value('i', 0)
# box_finished_n = Value('i',0)

total_num = len(datafile_list)  # 总图片数量,已提前统计过
# box_num = 1737937 # 物体检测文件数量
print('init......')


class TaskgenerateProcess(Process):
    def __init__(self, processQueen):
        Process.__init__(self)

        self.processQueen = processQueen

    def run(self):
        for file_name in datafile_list:
            self.processQueen.put(file_name)


class DownloadThread(threading.Thread):
    def __init__(self, processQueen, name=None):
        threading.Thread.__init__(self)
        self.processQueen = processQueen
        self.name = name

    def run(self):
        while True:
            try:
                task_info = self.processQueen.get()
                img_name = task_info
                img_contrast_up = image_process.apply_contrast(img_name, contrast=1.25)
                img_contrast_down = image_process.apply_contrast(img_name, contrast=.75)
                img_crop = image_process.apply_crop(img_name, y=.1, x=.1)
                img_blur = image_process.apply_blur(img_name)
                img_letterbox = image_process.apply_letter_box(img_name)
                img_text = image_process.apply_text(img_name)
                img_zoom_up = image_process.rescale(img_name, 1.2)
                img_zoom_down = image_process.rescale(img_name, .2)

                img_flip = image_process.apply_flip(img_name)
                img_shift = image_process.apply_shift(img_name, x_ratio=0.1, y_ratio=0.1)
                img_rotate = image_process.apply_rotate(img_name, 45)

                dir_tmp = dest_dir_cont_up + img_name.split('/')[-1].split('.')[0] + '_con_25+.jpg'
                img_contrast_up.save(dest_dir_cont_up + img_name.split('/')[-1].split('.')[0] + '_con_25+.jpg')
                img_contrast_down.save(dest_dir_cont_down + img_name.split('/')[-1].split('.')[0] + '_con_25-.jpg')
                cv2.imwrite(dest_dir_crop + img_name.split('/')[-1].split('.')[0] + '_crop_1.jpg', img_crop)
                cv2.imwrite(dest_dir__blur + img_name.split('/')[-1].split('.')[0] + '_blur.jpg', img_blur)
                cv2.imwrite(dest_dir_letterbox + img_name.split('/')[-1].split('.')[0] + '_letterbox.jpg',
                            img_letterbox)
                cv2.imwrite(dest_dir_text + img_name.split('/')[-1].split('.')[0] + '_text.jpg', img_text)
                cv2.imwrite(dest_dir_zoom_up + img_name.split('/')[-1].split('.')[0] + '_zoom_20+.jpg', img_zoom_up)
                cv2.imwrite(dest_dir_zoom_down + img_name.split('/')[-1].split('.')[0] + '_zoom_20-.jpg', img_zoom_down)
                cv2.imwrite(dest_dir_flip + img_name.split('/')[-1].split('.')[0] + '_flip_ho.jpg', img_flip)
                cv2.imwrite(dest_dir_shift + img_name.split('/')[-1].split('.')[0] + '_shift.jpg', img_shift)
                cv2.imwrite(dest_dir_rot_45 + img_name.split('/')[-1].split('.')[0] + '_rot_45+.jpg', img_rotate)
                finish_display(task_info, update_box_n=True)
            except Exception as e:
                err_file_lock.acquire()
                error_files_fp.write(task_info)
                error_files_fp.flush()
                err_file_lock.release()


class DownloadProcess(Process):
    def __init__(self, processQueen, name=None):
        Process.__init__(self)
        self.continueFlag = True
        self.processQueen = processQueen
        self.name = name

    def run(self):
        threads = []
        for i in range(thread_num):
            dt = DownloadThread(processQueen, "%s TD%s" % (self.name, i))
            dt.start()
            threads.append(dt)

        # 合并到父进程
        for t in threads:
            t.join()
        print "End Main threading" + self.name


# 初始化文件夹
init_dir()

# 进程数
process_list = []

# 下载任务创建，此进程必须启动
task_p = TaskgenerateProcess(processQueen)
task_p.start()
process_list.append(task_p)

# 创建下载进程
for i in range(process_num):
    d = DownloadProcess(processQueen, 'PID%s' % i)
    d.start()
    process_list.append(d)
# 合并到主进程
for p in process_list:
    p.join()

# 关闭错误日志文件
error_files_fp.flush()
error_files_fp.close()
print('Finished!')
