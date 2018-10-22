import argparse

import cv2

p = argparse.ArgumentParser()
p.add_argument("--images-path", required=False, type=str, help='input folder with images',
               default='/shihuijie/project/img_process/trans_imgs')
p.add_argument("--record-path", required=False, type=str, help='output TFRecord', default='./train.record')
p.add_argument("--depth", required=False, type=int, choices=set((3, 1)), default=3, help='image output depth')
p.add_argument("--resize", required=False, type=str, help='image output size wxh')
args = p.parse_args()

resize = args.resize
depth = args.depth
images_path = args.images_path
record_path = args.record_path

if resize is not None:
    width, height = [int(val) for val in resize.split('x')]

from utils.utils import get_all_files

with open(record_path, 'w') as record:
    img_list = get_all_files(images_path)
    for img_path in img_list:
        img = cv2.imread(img_path)

        if depth == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if resize is not None:
            img = cv2.resize(img, (112, 144))
        else:
            img = cv2.resize(img, (112, 144))
        record.write(img.tostring())
