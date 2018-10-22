# coding=utf-8

import os

import numpy as np
import tensorflow as tf


class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, img_channels=3, label_channels=1, random_scale=False,
                 random_mirror=False, img_mean=None, random_crop=False, shuffle=False, epoch=None):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: useless
          data_list: path to the image file with lines of the form '/path/to/image'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          img_mean: vector of mean colour values, it will not be performed if it is None
        '''
        assert input_size is not None, 'input_size should not be None'
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size

        # self.image_path_list, self.label_path_list = self.read_labeled_image_list(self.data_dir, self.data_list)
        self.image_path_list = self.read_image_list(self.data_dir, self.data_list)
        self.label_path_list = []

        # list size
        self.data_list_len = len(self.image_path_list)

        self.image_paths = tf.convert_to_tensor(self.image_path_list, dtype=tf.string)
        self.label_paths = tf.convert_to_tensor(self.label_path_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.image_paths, self.label_paths], shuffle=shuffle,
                                                   num_epochs=epoch)

        # self.queue = tf.train.slice_input_producer([self.image_paths], shuffle=shuffle,
        #                                            num_epochs=epoch)
        self.image_name, self.image, self.label = self.read_images_from_disk(self.queue, self.input_size, random_scale,
                                                                             random_mirror, img_mean,
                                                                             random_crop=random_crop,
                                                                             img_channels=img_channels,
                                                                             label_channels=label_channels)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        Args:
          num_elements: the batch size.
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        # image_name_batch, image_batch, label_batch = tf.train.batch([self.image_name, self.image, self.label],
        #                                                             num_elements)
        # return image_name_batch, image_batch, label_batch

        # image_name_batch, image_batch, label_batch = tf.train.batch([self.image_name, self.image, self.label],
        #                                                             num_elements)
        # return image_name_batch, image_batch, label_batch
        image_name_batch, image_batch = tf.train.batch([self.image_name, self.image],
                                                       num_elements)
        return image_name_batch, image_batch

    def image_scaling(self, img, label):
        """
        Randomly scales the images between 0.5 to 1.5 times the original size.

        Args:
          img: Training image to scale.
          label: Segmentation mask to scale.
        """

        scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
        h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
        w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
        new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
        img = tf.image.resize_images(img, new_shape)
        if label != None:
            label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
            label = tf.squeeze(label, squeeze_dims=[0])

            return img, label
        else:
            return img, None

    def image_mirroring(self, img, label):
        """
        Randomly mirrors the images.

        Args:
          img: Training image to mirror.
          label: Segmentation mask to mirror.
        """

        distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
        mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
        mirror = tf.boolean_mask([0, 1, 2], mirror)
        img = tf.reverse(img, mirror)
        if label != None:
            label = tf.reverse(label, mirror)
            return img, label
        else:
            return img, None

    def random_crop_and_pad_image_and_labels(self, image, label, crop_h, crop_w, img_channels=3, label_channels=1):
        """
        Randomly crop and pads the input images.

        Args:
          image: Training image to crop/ pad.
          label: Segmentation mask to crop/ pad.
          crop_h: Height of cropped segment.
          crop_w: Width of cropped segment.
        """
        if label != None:
            label = tf.cast(label, dtype=tf.float32)
            combined = tf.concat(axis=2, values=[image, label])
            image_shape = tf.shape(image)
            combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
                                                        tf.maximum(crop_w, image_shape[1]))

            combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, img_channels + label_channels])
            img_crop = combined_crop[:, :, :img_channels]
            label_crop = combined_crop[:, :, img_channels:]
            label_crop = tf.cast(label_crop, dtype=tf.uint8)

            # Set static shape so that tensorflow knows shape at compile time.
            img_crop.set_shape((crop_h, crop_w, img_channels))
            label_crop.set_shape((crop_h, crop_w, label_channels))
            return tf.cast(img_crop, dtype=tf.uint8), tf.cast(label_crop, dtype=tf.uint8)
        else:
            image_pad = tf.image.resize_image_with_crop_or_pad(image, 144 + np.random.randint(0, 12),
                                                               112 + np.random.randint(0, 12))
            return image_pad, None

    def read_image_list(self, data_dir, data_list):
        """Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: None
          data_list: absolute path of the image file with lines of the form '/path/to/image'.

        Returns:
          lists with all file names for images
        """
        f = open(data_list, 'r')
        images = []
        for line in f:
            image = line.strip("\n")
            images.append(data_dir + image)
        return images

    def read_labeled_image_list(self, data_dir, data_list):
        """Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        images = []
        masks = []
        for line in f:
            try:
                image, mask = line.strip("\n").split(' ')
            except ValueError:  # Adhoc for test.
                image = mask = line.strip("\n")
            images.append(data_dir + image)
            masks.append(data_dir + mask)
        return images, masks

    def resize_a_image(self, image, label, target_size, img_channels=3, label_channels=1):
        crop_h, crop_w = target_size[0], target_size[1]
        combined_resize = tf.image.resize_images(image, [crop_h, crop_w],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # label_resize = tf.image.rgb_to_grayscale(label_resize)

        img_resize = combined_resize
        # Set static shape so that tensorflow knows shape at compile time.
        img_resize.set_shape((crop_h, crop_w, img_channels))
        return tf.cast(img_resize, dtype=tf.uint8), None

    def read_images_from_disk(self, input_queue, input_size, random_scale, random_mirror, img_mean, random_crop=False,
                              img_channels=3, label_channels=1):  # optional pre-processing arguments
        """Read one image and its corresponding mask with optional pre-processing.

        Args:
          input_queue: tf queue with paths to the image and its mask.
          input_size: a tuple with (height, width) values.
                      If not given, return images of original size.
          random_scale: whether to randomly scale the images prior
                        to random crop.
          random_mirror: whether to randomly mirror the images prior
                        to random crop.
          img_mean: vector of mean colour values.
          label_channels -1: 没有mask信息

        Returns:
          Two tensors: the decoded image and its mask.
        """

        img_name = tf.string_split([input_queue[0]], delimiter='/').values[-1]
        img_contents = tf.read_file(input_queue[0])
        label_contents = tf.read_file(input_queue[1])

        img = tf.image.decode_jpeg(img_contents, channels=img_channels)
        img = tf.cast(img, dtype=tf.uint8)

        # transpose to bgr and be abandoned in seg-gan
        # img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        # img = tf.cast(tf.concat(axis=2, values=[img_r, img_g, img_b]), dtype=tf.float32)

        if not img_mean is None:
            # Extract mean.
            img -= img_mean

        if label_channels != -1:
            label = tf.image.decode_png(label_contents, channels=label_channels)
        else:
            label = None
        # Randomly scale the images and labels.
        if random_scale:
            img, label = self.image_scaling(img, label)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label = self.image_mirroring(img, label)

        if random_crop:
            # Randomly crops the images and labels.
            # if the task is vcdb ,this process is to 填充小黑边
            img, label = self.random_crop_and_pad_image_and_labels(img, label, input_size[0], input_size[1],
                                                                   img_channels=img_channels,
                                                                   label_channels=label_channels)
        # finally resize the image to the desire size
        img, label = self.resize_a_image(img, label, input_size, img_channels=img_channels,
                                         label_channels=label_channels)

        return img_name, img, label
