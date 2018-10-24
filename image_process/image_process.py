# coding=utf-8

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import cv2
import numpy as np
import matplotlib

from PIL import Image, ImageEnhance, ImageDraw, ImageFont

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def apply_contrast(image_name, contrast=1):
    image = Image.open(image_name)
    enh_con = ImageEnhance.Contrast(image)
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


def apply_crop(image_name, y=.1, x=.1):
    img = cv2.imread(image_name)
    (h, w) = img.shape[:2]
    x = int(w * x / 2)
    y = int(h * y / 2)
    crop_img = img[y:h + y, x:w - x]
    return crop_img


def apply_blur(image_name):
    image = cv2.imread(image_name)
    return cv2.GaussianBlur(image, (5, 5), 10)


def apply_letter_box(image_name, ratio=0.15):
    BLACK = [0, 0, 0]
    img = cv2.imread(image_name)
    (h, w) = img.shape[:2]
    box_img = cv2.copyMakeBorder(img, int(w * ratio / 2), int(w * ratio / 2), 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    return box_img


def apply_text(image_name):
    img_OpenCV = cv2.imread(image_name)
    # 图像从OpenCV格式转换成PIL格式
    img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV, cv2.COLOR_BGR2RGB))

    # 字体  字体*.ttc的存放路径一般是： /usr/share/fonts/opentype/noto/ 查找指令locate *.ttc
    font = ImageFont.load_default().font
    # font = ImageFont.truetype('/Library/Fonts/Mishafi.ttc', 40)
    # 字体颜色
    fillColor = (255, 0, 0)
    # 文字输出位置
    position = (100, 100)
    # 输出内容
    str = '在图片上输出中文'
    str = 'what the !'

    # 需要先把输出的中文字符转换成Unicode编码形式
    if not isinstance(str, unicode):
        str = str.decode('utf8')

    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, str, font=font, fill=fillColor)
    # 使用PIL中的save方法保存图片到本地
    # img_PIL.save('02.jpg', 'jpeg')

    # 转换回OpenCV格式
    image_text = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return image_text


def rescale(image_name, scale=None):
    '''
    等比例缩放
    :param image_name:
    :param scale:
    :return:
    '''
    image = cv2.imread(image_name)
    img_res = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return img_res


def apply_flip(image_name):
    image = cv2.imread(image_name)
    return cv2.flip(image, 1)


def apply_shift(image_name, x_ratio=0.1, y_ratio=0.1):
    image = cv2.imread(image_name)
    (h, w) = image.shape[:2]
    x = 0
    y = 0
    if x_ratio != 0:
        x = w * x_ratio
    if y_ratio != 0:
        y = h * y_ratio
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


def apply_rotate(image_name, angle, center=None, scale=1.0):
    image = cv2.imread(image_name)

    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def resize(image_name, output_shape=(32, 32)):
    image = cv2.imread(image_name)
    return cv2.resize(image, output_shape)


def img2array(img, dtype=np.float32):
    return np.array(img, dtype=dtype)


if __name__ == '__main__':
    img_name = 'b.jpeg'
    img1 = cv2.imread(img_name)

    img_contrast_up = apply_contrast(img_name, contrast=1.25)
    img_contrast_down = apply_contrast(img_name, contrast=.75)
    img_crop = apply_crop(img_name, y=.4, x=.4)
    img_blur = apply_blur(img_name)
    img_letterbox = apply_letter_box(img_name)
    img_text = apply_text(img_name)
    img_zoom_up = rescale(img_name, 1.2)
    img_zoom_down = rescale(img_name, .2)

    img_flip = apply_flip(img_name)
    img_shift = apply_shift(img_name, x_ratio=0.1, y_ratio=0.1)
    img_rotate = apply_rotate(img_name, 45)

    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), plt.title('ORIGINAL')
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(img_rotate, cv2.COLOR_BGR2RGB)), plt.title('ROTATE_45')
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), plt.title('ORIGINAL')
    plt.subplot(3, 4, 2)
    plt.imshow(img_contrast_up), plt.title('CONTRAST_125%')
    plt.subplot(3, 4, 3)
    plt.imshow(img_contrast_down), plt.title('CONTRAST_75%')
    plt.subplot(3, 4, 4)
    plt.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)), plt.title('CROP_5%')
    plt.subplot(3, 4, 5)
    plt.imshow(cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)), plt.title('BLUR')
    plt.subplot(3, 4, 6)
    plt.imshow(cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)), plt.title('LETTER_BOX')
    plt.subplot(3, 4, 7)
    plt.imshow(cv2.cvtColor(img_text, cv2.COLOR_BGR2RGB)), plt.title('ADD_TEXT')
    plt.subplot(3, 4, 8)
    plt.imshow(cv2.cvtColor(img_shift, cv2.COLOR_BGR2RGB)), plt.title('SHIFT')
    plt.subplot(3, 4, 9)
    plt.imshow(cv2.cvtColor(img_zoom_up, cv2.COLOR_BGR2RGB)), plt.title('ZOOM_120%')
    plt.subplot(3, 4, 10)
    plt.imshow(cv2.cvtColor(img_zoom_down, cv2.COLOR_BGR2RGB)), plt.title('ZOOM_80%')
    plt.subplot(3, 4, 11)
    plt.imshow(cv2.cvtColor(img_flip, cv2.COLOR_BGR2RGB)), plt.title('FLIP_HORI')
    plt.subplot(3, 4, 12)
    plt.imshow(cv2.cvtColor(img_rotate, cv2.COLOR_BGR2RGB)), plt.title('ROTATE_45+')

    plt.show()
