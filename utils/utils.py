# coding=utf-8
import os
import shutil

import skimage
import skimage.io
import skimage.transform
import numpy as np


def get_all_files(path):
    all_file = []
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            if name.endswith('.jpg'):
                all_file.append(os.path.join(dirpath, name))
    return all_file


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


def copy_size_max(dirname, targetDir):
    subdir_list = os.listdir(dirname)
    video_max = []
    for sub_dirname in subdir_list:
        allsize = []
        for (curDir, subDir, fileHere) in os.walk(os.path.join(dirname, sub_dirname)):
            print(curDir)
            for filename in fileHere:
                fullname = os.path.join(curDir, filename)
                filesize = os.path.getsize(fullname)
                allsize.append((filesize, fullname))

        allsize.sort(key=lambda x: x[0])
        print(allsize[-1])
        video_max.append(allsize[-1])
    print video_max

    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    for eachfile in video_max:
        eachfile = eachfile[1]
        if not os.path.exists(eachfile):
            print "src path not exist:" + eachfile
        shutil.copy(eachfile, os.path.join(targetDir, os.path.basename(eachfile)))
        print eachfile + " copy succeeded!"


if __name__ == '__main__':
    copy_size_max('/home/shihuijie/Desktop/ml/tmp/vcdb/core_dataset', '/home/shihuijie/Desktop/video_vcdb_to_train')
